import torch
import pufferlib

import numpy as np

import torch
from torch import nn
from torch.nn.utils import rnn
from model.simple import io, subnets
from model.simple.model_architecture import EMBED_DIM, HIDDEN_DIM, PLAYER_VISION_DIAMETER

from nmmo.entity.entity import EntityState
EntityId = EntityState.State.attr_name_to_col["id"]

class SimplePolicy(pufferlib.models.Policy):
  def __init__(self, binding, input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM):
    super().__init__(binding, input_size, hidden_size)
    self.observation_space = binding.featurized_single_observation_space

    # xcxc Do these belong here?
    self.state_handler_dict = {}
    torch.set_num_threads(1)

    self.input = io.Input(embeddings=io.MixedEmbedding)

    self.entity_net = nn.Linear(2*EMBED_DIM, EMBED_DIM)
    self.tile_net = nn.Conv2d(EMBED_DIM, EMBED_DIM, 3)
    self.pool = nn.MaxPool2d(2)
    self.fc = nn.Linear(EMBED_DIM*6*6, HIDDEN_DIM)

    self.proj   = nn.Linear(2*HIDDEN_DIM, HIDDEN_DIM)
    self.attend = subnets.SelfAttention(EMBED_DIM, HIDDEN_DIM)

    self.value_head = nn.Linear(HIDDEN_DIM, 1)
    self.policy_head = io.Output()

  def critic(self, hidden):
      return self.value_head(hidden)

  def encode_observations(self, env_outputs):
    obs = pufferlib.emulation.unpack_batched_obs(
      self.observation_space,
      #self.single_observation_space,
      env_outputs
    )
    embedded_obs = self.input(obs)
    embedded_obs["ActionTargets"] = obs["ActionTargets"]

    #Attentional agent embedding
    agentEmb  = embedded_obs['Entity']

    # Pull out rows corresponding to the agent
    my_id = obs["Game"][:,1]
    entity_ids = obs["Entity"][:,:,EntityId]
    mask = (entity_ids == my_id.unsqueeze(1)) & (entity_ids != 0)
    mask = mask.int()
    row_indices = torch.where(mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1)))
    selfEmb = agentEmb[torch.arange(agentEmb.shape[0]), row_indices]
    selfEmb   = selfEmb.unsqueeze(dim=1).expand_as(agentEmb)

    # Concatenate self and agent embeddings
    agents    = torch.cat((selfEmb, agentEmb), dim=-1)
    agents    = self.entity_net(agents)
    agents, _ = self.attend(agents)
    #agents = self.ent(selfEmb)

    #Convolutional tile embedding
    tiles     = embedded_obs['Tile']
    self.attn = torch.norm(tiles, p=2, dim=-1)

    w = PLAYER_VISION_DIAMETER
    batch  = tiles.size(0)
    hidden = tiles.size(2)
    #Dims correct?
    tiles  = tiles.reshape(batch, w, w, hidden).permute(0, 3, 1, 2)
    tiles  = self.tile_net(tiles)
    tiles  = self.pool(tiles)
    tiles  = tiles.reshape(batch, -1)
    tiles  = self.fc(tiles)

    hidden = torch.cat((agents, tiles), dim=-1)
    hidden = self.proj(hidden)

    return hidden, embedded_obs

  def decode_actions(self, hidden, embeeded_obs, concat=True):
    return self.policy_head(hidden, embeeded_obs.to(self.device))

  @staticmethod
  def create_policy():
    return pufferlib.frameworks.cleanrl.make_policy(
      SimplePolicy,
      recurrent_args=[HIDDEN_DIM, HIDDEN_DIM],
      recurrent_kwargs={'num_layers': 1},
    )
