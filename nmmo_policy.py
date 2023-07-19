from argparse import ArgumentParser
from typing import Dict

import pufferlib
import pufferlib.emulation
import pufferlib.models
import torch
import torch.nn.functional as F

from nmmo.entity.entity import EntityState
import nmmo


EntityId = EntityState.State.attr_name_to_col["id"]

def add_args(parser: ArgumentParser):
  parser.add_argument(
      "--policy.num_lstm_layers",
      dest="num_lstm_layers",
      type=int,
      default=0,
      help="number of LSTM layers to use (default: 0)",
  )

  parser.add_argument(
     "--policy.task_size",
      dest="task_size",
      type=int,
      default=1024,
      help="size of task embedding (default: 1024)",
  )

  parser.add_argument(
      "--policy.mask_actions",
      dest="mask_actions",
      type=bool,
      default=True,
      help="mask actions (default: True)",
  )

  parser.add_argument(
      "--policy.encode_task",
      dest="encode_task",
      type=bool,
      default=True,
      help="encode task (default: True)",
  )


class TileEncoder(torch.nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.tile_offset = torch.tensor([i * 256 for i in range(3)])
    self.embedding = torch.nn.Embedding(3 * 256, 32)

    self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
    self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
    self.tile_fc = torch.nn.Linear(8 * 11 * 11, input_size)

  def forward(self, tile):
    tile[:, :, :2] -= tile[:, 112:113, :2].clone()
    tile[:, :, :2] += 7
    tile = self.embedding(tile.long().clip(
        0, 255) + self.tile_offset.to(tile.device))

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

    return tile


class PlayerEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.player_offset = torch.tensor([i * 256 for i in range(23)])
    self.embedding = torch.nn.Embedding(23 * 256, 32)

    self.agent_fc = torch.nn.Linear(23 * 32, hidden_size)
    self.my_agent_fc = torch.nn.Linear(23 * 32, input_size)

  def forward(self, agents, my_id):
    # Pull out rows corresponding to the agent
    agent_ids = agents[:, :, EntityId]
    mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
    mask = mask.int()
    row_indices = torch.where(
        mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
    )

    agent_embeddings = self.embedding(
        agents.long().clip(0, 255) + self.player_offset.to(agents.device)
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

    return agent_embeddings, my_agent_embeddings

class ItemEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.item_offset = torch.tensor([i * 256 for i in range(16)])
    self.embedding = torch.nn.Embedding(32, 32)

    self.fc = torch.nn.Linear(2*32+12, hidden_size)

    self.discrete_idxs = [1, 14]
    self.discrete_offset = torch.Tensor([2, 0])
    self.continuous_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
    self.continuous_scale = torch.Tensor([1/10, 1/10, 1/10, 1/100, 1/100, 1/100, 1/40, 1/40, 1/40, 1/100, 1/100, 1/100])

  def forward(self, items):
    if self.discrete_offset.device != items.device:
      self.discrete_offset = self.discrete_offset.to(items.device)
      self.continuous_scale = self.continuous_scale.to(items.device)

    # Embed each feature separately
    discrete = items[:, :, self.discrete_idxs] + self.discrete_offset
    discrete = self.embedding(discrete.long().clip(0, 255))
    batch, item, attrs, embed = discrete.shape
    discrete = discrete.view(batch, item, attrs * embed)

    continuous = items[:, :, self.continuous_idxs] / self.continuous_scale

    item_embeddings = torch.cat([discrete, continuous], dim=-1)
    item_embeddings = self.fc(item_embeddings)
    return item_embeddings


class InventoryEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.fc = torch.nn.Linear(12*hidden_size, input_size)

  def forward(self, inventory):
    agents, items, hidden = inventory.shape
    inventory = inventory.view(agents, items * hidden)
    return self.fc(inventory)


class MarketEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.fc = torch.nn.Linear(hidden_size, input_size)

  def forward(self, market):
    return self.fc(market).mean(-2)


class TaskEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size, task_size):
    super().__init__()
    self.fc = torch.nn.Linear(task_size, input_size)

  def forward(self, task):
    return self.fc(task.clone())

class ActionDecoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size, mask_actions):
    super().__init__()
    self.mask_actions = mask_actions
    self.layers = torch.nn.ModuleDict({
        'attack_style': torch.nn.Linear(hidden_size, 3),
        'attack_target': torch.nn.Linear(hidden_size, hidden_size),
        'market_buy': torch.nn.Linear(hidden_size, hidden_size),
        'inventory_destroy': torch.nn.Linear(hidden_size, hidden_size),
        'inventory_give_item': torch.nn.Linear(hidden_size, hidden_size),
        'inventory_give_player': torch.nn.Linear(hidden_size, hidden_size),
        'gold_quantity': torch.nn.Linear(hidden_size, 99),
        'gold_target': torch.nn.Linear(hidden_size, hidden_size),
        'move': torch.nn.Linear(hidden_size, 5),
        'inventory_sell': torch.nn.Linear(hidden_size, hidden_size),
        'inventory_price': torch.nn.Linear(hidden_size, 99),
        'inventory_use': torch.nn.Linear(hidden_size, hidden_size),
    })

  def apply_layer(self, layer, embeddings, mask, hidden):
    hidden = layer(hidden)
    if hidden.dim() == 2 and embeddings is not None:
      hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

    if mask is not None:
      mask[~mask.any(dim=1)] = True
      hidden = hidden.masked_fill(~mask.bool(), -1e9)

    return hidden

  def forward(self, hidden, lookup, concat):
    player_embeddings, inventory_embeddings, market_embeddings, action_targets = lookup

    embeddings = {
        'attack_target': player_embeddings,
        'market_buy': market_embeddings,
        'inventory_destroy': inventory_embeddings,
        'inventory_give_item': inventory_embeddings,
        'inventory_give_player': player_embeddings,
        'gold_target': player_embeddings,
        'inventory_sell': inventory_embeddings,
        'inventory_use': inventory_embeddings,
    }

    action_targets = {
        'attack_style': action_targets[nmmo.action.Attack][nmmo.action.Style],
        'attack_target': action_targets[nmmo.action.Attack][nmmo.action.Target],
        'market_buy': action_targets[nmmo.action.Buy][nmmo.action.MarketItem],
        'inventory_destroy': action_targets[nmmo.action.Destroy][nmmo.action.InventoryItem],
        'inventory_give_item': action_targets[nmmo.action.Give][nmmo.action.InventoryItem],
        'inventory_give_player': action_targets[nmmo.action.Give][nmmo.action.Target],
        'gold_quantity': action_targets[nmmo.action.GiveGold][nmmo.action.Price],
        'gold_target': action_targets[nmmo.action.GiveGold][nmmo.action.Target],
        'move': action_targets[nmmo.action.Move][nmmo.action.Direction],
        'inventory_sell': action_targets[nmmo.action.Sell][nmmo.action.InventoryItem],
        'inventory_price': action_targets[nmmo.action.Sell][nmmo.action.Price],
        'inventory_use': action_targets[nmmo.action.Use][nmmo.action.InventoryItem],
    }

    actions = []
    for key, layer in self.layers.items():
      mask = None
      if self.mask_actions:
        mask = action_targets[key]
      action = self.apply_layer(layer, embeddings.get(key), mask, hidden)
      actions.append(action)

    if concat:
      return torch.cat(actions, dim=-1)

    return actions


class NmmoPolicy(pufferlib.models.Policy):
  def __init__(self, binding, policy_args: Dict):
    super().__init__(binding)
    """Simple custom PyTorch policy subclassing the pufferlib BasePolicy

    This requires only that you structure your network as an observation encoder,
    an action decoder, and a critic function. If you use our LSTM support, it will
    be added between the encoder and the decoder.
    """
    super().__init__(binding)
    self.raw_single_observation_space = binding.raw_single_observation_space
    input_size = policy_args.get("input_size", 256)
    hidden_size = policy_args.get("hidden_size", 256)
    task_size = policy_args.get("task_size", 1024)
    mask_actions = policy_args.get("mask_actions", True)
    self.encode_task = policy_args.get("encode_task", True)
    self._policy_args = policy_args

    self.tile_encoder = TileEncoder(input_size)
    self.player_encoder = PlayerEncoder(input_size, hidden_size)
    self.item_encoder = ItemEncoder(input_size, hidden_size)
    self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
    self.market_encoder = MarketEncoder(input_size, hidden_size)

    if self.encode_task:
      self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
      self.proj_fc = torch.nn.Linear(5 * input_size, input_size)
    else:
      self.proj_fc = torch.nn.Linear(4 * input_size, input_size)

    self.action_decoder = ActionDecoder(input_size, hidden_size, mask_actions)

    self.value_head = torch.nn.Linear(hidden_size, 1)

  def critic(self, hidden):
    return self.value_head(hidden)

  def encode_observations(self, flat_observations):
    env_outputs = self.binding.unpack_batched_obs(flat_observations)[0]
    tile = self.tile_encoder(env_outputs["Tile"])
    player_embeddings, my_agent = self.player_encoder(env_outputs["Entity"], env_outputs["AgentId"][:, 0])

    inventory_embeddings = self.item_encoder(env_outputs["Inventory"])
    market_embeddings = self.item_encoder(env_outputs["Market"])

    inventory = self.inventory_encoder(inventory_embeddings)
    market = self.market_encoder(market_embeddings)

    if self.encode_task:
      task = self.task_encoder(env_outputs["Task"])
      obs = torch.cat([tile, my_agent, inventory, market, task], dim=-1)
    else:
      obs = torch.cat([tile, my_agent, inventory, market], dim=-1)

    return self.proj_fc(obs), (player_embeddings, inventory_embeddings, market_embeddings, env_outputs['ActionTargets'])

  def decode_actions(self, flat_hidden, lookup, concat=True):
    return self.action_decoder(flat_hidden, lookup, concat)

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
