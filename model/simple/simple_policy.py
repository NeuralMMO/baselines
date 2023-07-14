from collections import defaultdict

import nmmo
import numpy as np
import pufferlib
import torch
from nmmo.entity.entity import EntityState
from torch import nn

from model.simple import io, subnets
from model.simple.model_architecture import (
    EMBED_DIM,
    HIDDEN_DIM,
    PLAYER_VISION_DIAMETER,
)

EntityId = EntityState.State.attr_name_to_col["id"]


class SimplePolicy(pufferlib.models.Policy):
  def __init__(self, binding, input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM):
    super().__init__(binding, input_size, hidden_size)
    self.observation_space = binding.featurized_single_observation_space
    self.action_space = binding._single_action_space

    self.state_handler_dict = {}
    torch.set_num_threads(1)

    self.input = io.Input(embeddings=io.MixedEmbedding)

    self.entity_net = nn.Linear(2 * EMBED_DIM, EMBED_DIM)
    self.tile_net = nn.Conv2d(EMBED_DIM, EMBED_DIM, 3)
    self.pool = nn.MaxPool2d(2)
    self.fc = nn.Linear(EMBED_DIM * 6 * 6, HIDDEN_DIM)

    self.proj = nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
    self.attend = subnets.SelfAttention(EMBED_DIM, HIDDEN_DIM)

    self.value_head = nn.Linear(HIDDEN_DIM, 1)
    self.policy_head = PolicyHead(self.action_space)

  def critic(self, hidden):
    return self.value_head(hidden)

  def encode_observations(self, env_outputs):
    obs = pufferlib.emulation.unpack_batched_obs(
        self.observation_space,
        # self.single_observation_space,
        env_outputs,
    )
    embedded_obs = self.input(obs)
    embedded_obs["ActionTargets"] = obs["ActionTargets"]

    # Attentional agent embedding
    agentEmb = embedded_obs["Entity"]

    # Pull out rows corresponding to the agent
    my_id = obs["AgentId"][:, 0]
    entity_ids = obs["Entity"][:, :, EntityId]
    mask = (entity_ids == my_id.unsqueeze(1)) & (entity_ids != 0)
    mask = mask.int()
    row_indices = torch.where(
        mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
    )
    selfEmb = agentEmb[torch.arange(agentEmb.shape[0]), row_indices]
    selfEmb = selfEmb.unsqueeze(dim=1).expand_as(agentEmb)

    # Concatenate self and agent embeddings
    agents = torch.cat((selfEmb, agentEmb), dim=-1)
    agents = self.entity_net(agents)
    agents, _ = self.attend(agents)
    # agents = self.ent(selfEmb)

    # Convolutional tile embedding
    tiles = embedded_obs["Tile"]
    self.attn = torch.norm(tiles, p=2, dim=-1)

    w = PLAYER_VISION_DIAMETER
    batch = tiles.size(0)
    hidden = tiles.size(2)
    # Dims correct?
    tiles = tiles.reshape(batch, w, w, hidden).permute(0, 3, 1, 2)
    tiles = self.tile_net(tiles)
    tiles = self.pool(tiles)
    tiles = tiles.reshape(batch, -1)
    tiles = self.fc(tiles)

    hidden = torch.cat((agents, tiles), dim=-1)
    hidden = self.proj(hidden)

    return hidden, embedded_obs

  def decode_actions(self, hidden, embeeded_obs, concat=True):
    return self.policy_head(hidden, embeeded_obs)

  @staticmethod
  def create_policy():
    return pufferlib.frameworks.cleanrl.make_policy(
        SimplePolicy,
        recurrent_args=[HIDDEN_DIM, HIDDEN_DIM],
        recurrent_kwargs={"num_layers": 1},
    )

  @staticmethod
  def env_creator(config, team_helper: TeamHelper):
    return lambda: NMMOEnv(config)

  @staticmethod
  def num_agents(team_helper: TeamHelper):
    return sum(len(t) for t in team_helper.teams.values())


class PolicyHead(nn.Module):
  def __init__(self, action_space):
    super().__init__()

    self.action_space = action_space
    self.proj = None
    if HIDDEN_DIM != EMBED_DIM:
      self.proj = nn.Linear(HIDDEN_DIM, EMBED_DIM)
    self.net = io.DiscreteAction(EMBED_DIM)
    self.arg = nn.Embedding(nmmo.Action.n, EMBED_DIM)

  def names(self, nameMap, args):
    """Lookup argument indices from name mapping"""
    return np.array([nameMap.get(e) for e in args])

  def forward(self, obs, lookup):
    """Populates an IO object with actions in-place

    Args:
        obs    : An IO object specifying observations
        lookup : A fixed size representation of each entity
    """
    if self.proj:
      obs = self.proj(obs)

    batch = obs.shape[0]

    rets = defaultdict(dict)
    for atn in [nmmo.action.Move, nmmo.action.Attack]:
      for arg in atn.edges:
        if arg.argType == nmmo.action.Fixed:
          batch = obs.shape[0]
          idxs = range(len(arg.edges))
          cands = self.arg.weight[idxs]
          cands = cands.repeat(batch, 1, 1)
        elif arg == nmmo.action.Target:
          cands = lookup["Entity"]

        mask = None
        if atn in lookup["ActionTargets"]:
          mask = lookup["ActionTargets"][atn][arg]

        logits = self.net(obs, cands, mask)
        rets[atn][arg] = logits

    return [
        rets[nmmo.action.Attack][nmmo.action.Style],
        rets[nmmo.action.Attack][nmmo.action.Target],
        rets[nmmo.action.Move][nmmo.action.Direction],
    ]
