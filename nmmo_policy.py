from argparse import ArgumentParser
from typing import Dict

import pufferlib
import pufferlib.emulation
import pufferlib.models
import torch
import torch.nn.functional as F
from nmmo.entity.entity import EntityState


def add_args(parser: ArgumentParser):
  parser.add_argument(
      "--policy.num_lstm_layers",
      dest="num_lstm_layers",
      type=int,
      default=0,
      help="number of LSTM layers to use (default: 0)",
  )


NUM_ATTRS = 26
EntityId = EntityState.State.attr_name_to_col["id"]
tile_offset = torch.tensor([i * 256 for i in range(3)])
agent_offset = torch.tensor([i * 256 for i in range(3, 26)])


class NmmoPolicy(pufferlib.models.Policy):
  def __init__(self, binding, policy_args: Dict):
    super().__init__(binding)
    self._policy_args = policy_args

    input_size = policy_args.get("input_size", 256)
    hidden_size = policy_args.get("hidden_size", 256)
    # output_size = policy_args.get("output_size", 128)

    self.raw_single_observation_space = binding.raw_single_observation_space

    # observation_size = binding.raw_single_observation_space["Entity"].shape[1]

    self.embedding = torch.nn.Embedding(NUM_ATTRS * 256, 32)
    self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
    self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
    self.tile_fc = torch.nn.Linear(8 * 11 * 11, input_size)

    self.agent_fc = torch.nn.Linear(23 * 32, hidden_size)
    self.my_agent_fc = torch.nn.Linear(23 * 32, input_size)

    self.proj_fc = torch.nn.Linear(2 * input_size, input_size)
    self.attack_fc = torch.nn.Linear(hidden_size, hidden_size)

    action_dims = binding.single_action_space.nvec
    action_dims = [action_dims[0], *action_dims[2:]]
    self.decoders = torch.nn.ModuleList(
        [torch.nn.Linear(hidden_size, n) for n in action_dims]
    )
    self.value_head = torch.nn.Linear(hidden_size, 1)

  def critic(self, hidden):
    return self.value_head(hidden)

  def encode_observations(self, env_outputs):
    # TODO: Change 0 for teams when teams are added
    env_outputs = self.binding.unpack_batched_obs(env_outputs)[0]

    tile = env_outputs["Tile"]
    # Center on player
    # This is cursed without clone??
    tile[:, :, :2] -= tile[:, 112:113, :2].clone()
    tile[:, :, :2] += 7
    tile = self.embedding(tile.long().clip(
        0, 255) + tile_offset.to(tile.device))

    agents, tiles, features, embed = tile.shape
    tile = (
        tile.view(agents, tiles, features * embed)
        .transpose(1, 2)
        .view(agents, features * embed, 15, 15)
    )

    tile = self.tile_conv_1(tile)
    tile = F.relu(tile)
    tile = self.tile_conv_2(tile)
    tile = F.relu(tile)
    tile = tile.contiguous().view(agents, -1)
    tile = self.tile_fc(tile)
    tile = F.relu(tile)

    # Pull out rows corresponding to the agent
    agents = env_outputs["Entity"]
    my_id = env_outputs["AgentId"][:, 0]
    agent_ids = agents[:, :, EntityId]
    mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
    mask = mask.int()
    row_indices = torch.where(
        mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
    )

    agent_embeddings = self.embedding(
        agents.long().clip(0, 255) + agent_offset.to(agents.device)
    )
    batch, agent, attrs, embed = agent_embeddings.shape

    # Embed each feature separately
    agent_embeddings = agent_embeddings.view(batch, agent, attrs * embed)
    my_agent_embeddings = agent_embeddings[
        torch.arange(agents.shape[0]), row_indices
    ]

    # Project to input of recurrent size
    agent_embeddings = self.agent_fc(agent_embeddings)
    my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
    my_agent_embeddings = F.relu(my_agent_embeddings)

    obs = torch.cat([tile, my_agent_embeddings], dim=-1)
    return self.proj_fc(obs), agent_embeddings

  def decode_actions(self, flat_hidden, lookup, concat=True):
    attack_key = self.attack_fc(flat_hidden)
    attack_logits = torch.matmul(lookup, attack_key.unsqueeze(-1)).squeeze(-1)

    actions = [dec(flat_hidden) for dec in self.decoders]
    actions.insert(1, attack_logits)
    if concat:
      return torch.cat(actions, dim=-1)
    return actions

  def policy_args(self):
    return self._policy_args

  @staticmethod
  def create_policy(binding: pufferlib.emulation.Binding, args: Dict):
    args["input_size"] = 128
    args["hidden_size"] = 256 if args["num_lstm_layers"] else 128

    return pufferlib.frameworks.cleanrl.make_policy(
        NmmoPolicy,
        recurrent_args=[args["input_size"], args["hidden_size"]]
        if args["num_lstm_layers"]
        else [],
        recurrent_kwargs={"num_layers": args["num_lstm_layers"]},
    )(binding, args)
