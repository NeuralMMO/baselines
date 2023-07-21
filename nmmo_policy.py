import argparse
from typing import Dict, Optional, Tuple

import nmmo
import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.models
import torch
import torch.nn.functional as F
from nmmo.entity.entity import EntityState
from torch import Tensor, nn

EntityId = EntityState.State.attr_name_to_col["id"]


class ScaledDotProductAttention(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.sqrt_dim = np.sqrt(dim)

  def forward(
      self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
  ) -> Tuple[Tensor, Tensor]:
    score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

    if mask is not None:
      score.masked_fill_(mask.view(score.size()), -float("Inf"))

    attn = F.softmax(score, -1)
    context = torch.bmm(attn, value)
    return context, attn


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int = 512, num_heads: int = 8):
    super().__init__()

    assert d_model % num_heads == 0, "d_model % num_heads should be zero."

    self.d_head = int(d_model / num_heads)
    self.num_heads = num_heads
    self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
    self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
    self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
    self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

  def forward(
      self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
  ) -> Tuple[Tensor, Tensor]:
    batch_size = value.size(0)

    query = self.query_proj(query).view(
        batch_size, -1, self.num_heads, self.d_head
    )  # BxQ_LENxNxD
    key = self.key_proj(key).view(
        batch_size, -1, self.num_heads, self.d_head
    )  # BxK_LENxNxD
    value = self.value_proj(value).view(
        batch_size, -1, self.num_heads, self.d_head
    )  # BxV_LENxNxD

    query = (
        query.permute(2, 0, 1, 3)
        .contiguous()
        .view(batch_size * self.num_heads, -1, self.d_head)
    )  # BNxQ_LENxD
    key = (
        key.permute(2, 0, 1, 3)
        .contiguous()
        .view(batch_size * self.num_heads, -1, self.d_head)
    )  # BNxK_LENxD
    value = (
        value.permute(2, 0, 1, 3)
        .contiguous()
        .view(batch_size * self.num_heads, -1, self.d_head)
    )  # BNxV_LENxD

    if mask is not None:
      mask = mask.unsqueeze(1).repeat(
          1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

    context, attn = self.scaled_dot_attn(query, key, value, mask)

    context = context.view(self.num_heads, batch_size, -1, self.d_head)
    context = (
        context.permute(1, 2, 0, 3)
        .contiguous()
        .view(batch_size, -1, self.num_heads * self.d_head)
    )  # BxTxND

    return context, attn


def str_to_bool(s):
  if s.lower() in ("yes", "true", "t", "y", "1"):
    return True
  if s.lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError("Boolean value expected.")


def add_args(parser: argparse.ArgumentParser):
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
      type=str,
      default="none",
      choices=["none", "move", "all", "exclude-attack"],
      help="mask actions - none, move, all, or exclude-attack (default: none)",
  )

  parser.add_argument(
      "--policy.encode_task",
      dest="encode_task",
      type=str_to_bool,
      default=True,
      help="encode task (default: False)",
  )

  parser.add_argument(
      "--policy.attend_task",
      dest="attend_task",
      type=str,
      default="none",
      choices=["none", "pytorch", "nikhil"],
      help="attend task - none, pytorch, or nikhil (default: none)",
  )

  parser.add_argument(
      "--policy.attentional_decode",
      dest="attentional_decode",
      type=str_to_bool,
      default=True,
      help="use attentional action decoder (default: False)",
  )

  parser.add_argument(
      "--policy.extra_encoders",
      dest="extra_encoders",
      type=str_to_bool,
      default=True,
      help="use inventory and market encoders (default: False)",
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
    tile = self.embedding(
        tile.long().clip(0, 255) + self.tile_offset.to(tile.device)
    )

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

    self.fc = torch.nn.Linear(2 * 32 + 12, hidden_size)

    self.discrete_idxs = [1, 14]
    self.discrete_offset = torch.Tensor([2, 0])
    self.continuous_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
    self.continuous_scale = torch.Tensor(
        [
            1 / 10,
            1 / 10,
            1 / 10,
            1 / 100,
            1 / 100,
            1 / 100,
            1 / 40,
            1 / 40,
            1 / 40,
            1 / 100,
            1 / 100,
            1 / 100,
        ]
    )

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
    self.fc = torch.nn.Linear(12 * hidden_size, input_size)

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
    self.layers = torch.nn.ModuleDict(
        {
            "attack_style": torch.nn.Linear(hidden_size, 3),
            "attack_target": torch.nn.Linear(hidden_size, hidden_size),
            "market_buy": torch.nn.Linear(hidden_size, hidden_size),
            "inventory_destroy": torch.nn.Linear(hidden_size, hidden_size),
            "inventory_give_item": torch.nn.Linear(hidden_size, hidden_size),
            "inventory_give_player": torch.nn.Linear(hidden_size, hidden_size),
            "gold_quantity": torch.nn.Linear(hidden_size, 99),
            "gold_target": torch.nn.Linear(hidden_size, hidden_size),
            "move": torch.nn.Linear(hidden_size, 5),
            "inventory_sell": torch.nn.Linear(hidden_size, hidden_size),
            "inventory_price": torch.nn.Linear(hidden_size, 99),
            "inventory_use": torch.nn.Linear(hidden_size, hidden_size),
        }
    )

  def apply_layer(self, layer, embeddings, mask, hidden):
    hidden = layer(hidden)
    if hidden.dim() == 2 and embeddings is not None:
      hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

    if mask is not None:
      hidden = hidden.masked_fill(mask == 0, -1e9)

    return hidden

  def forward(self, hidden, lookup):
    (
        player_embeddings,
        inventory_embeddings,
        market_embeddings,
        action_targets,
    ) = lookup

    embeddings = {
        "attack_target": player_embeddings,
        "market_buy": market_embeddings,
        "inventory_destroy": inventory_embeddings,
        "inventory_give_item": inventory_embeddings,
        "inventory_give_player": player_embeddings,
        "gold_target": player_embeddings,
        "inventory_sell": inventory_embeddings,
        "inventory_use": inventory_embeddings,
    }

    action_targets = {
        "attack_style": action_targets[nmmo.action.Attack][nmmo.action.Style],
        "attack_target": action_targets[nmmo.action.Attack][nmmo.action.Target],
        "market_buy": action_targets[nmmo.action.Buy][nmmo.action.MarketItem],
        "inventory_destroy": action_targets[nmmo.action.Destroy][
            nmmo.action.InventoryItem
        ],
        "inventory_give_item": action_targets[nmmo.action.Give][
            nmmo.action.InventoryItem
        ],
        "inventory_give_player": action_targets[nmmo.action.Give][
            nmmo.action.Target
        ],
        "gold_quantity": action_targets[nmmo.action.GiveGold][nmmo.action.Price],
        "gold_target": action_targets[nmmo.action.GiveGold][nmmo.action.Target],
        "move": action_targets[nmmo.action.Move][nmmo.action.Direction],
        "inventory_sell": action_targets[nmmo.action.Sell][
            nmmo.action.InventoryItem
        ],
        "inventory_price": action_targets[nmmo.action.Sell][nmmo.action.Price],
        "inventory_use": action_targets[nmmo.action.Use][nmmo.action.InventoryItem],
    }

    actions = []
    for key, layer in self.layers.items():
      mask = None
      if self.mask_actions == "all":
        mask = action_targets[key]
      elif self.mask_actions == "move" and key == "move":
        mask = action_targets[key]
      elif self.mask_actions == "exclude-attack" and key != "attack_target":
        mask = action_targets[key]
      action = self.apply_layer(layer, embeddings.get(key), mask, hidden)
      actions.append(action)

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
    self.attend_task = policy_args.get("attend_task", "none")
    self.attentional_decode = policy_args.get("attentional_decode", True)
    self.extra_encoders = policy_args.get("extra_encoders", True)
    self._policy_args = policy_args

    self.tile_encoder = TileEncoder(input_size)
    self.player_encoder = PlayerEncoder(input_size, hidden_size)

    if self.extra_encoders:
      self.item_encoder = ItemEncoder(input_size, hidden_size)
      self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
      self.market_encoder = MarketEncoder(input_size, hidden_size)

    num_encode = 2
    if self.encode_task:
      self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
      if self.attend_task == "nikhil":
        self.task_attention = MultiHeadAttention(input_size, input_size)
      elif self.attend_task == "pytorch":
        pass
      else:
        num_encode += 1
    if self.extra_encoders:
      num_encode += 2

    self.proj_fc = torch.nn.Linear(num_encode * input_size, input_size)

    if self.attentional_decode:
      self.action_decoder = ActionDecoder(
          input_size, hidden_size, mask_actions)
    else:
      self.action_decoder = torch.nn.ModuleList(
          [
              torch.nn.Linear(hidden_size, n)
              for n in binding.single_action_space.nvec
          ]
      )

    self.value_head = torch.nn.Linear(hidden_size, 1)

  def critic(self, hidden):
    return self.value_head(hidden)

  def encode_observations(self, flat_observations):
    env_outputs = self.binding.unpack_batched_obs(flat_observations)[0]
    tile = self.tile_encoder(env_outputs["Tile"])
    player_embeddings, my_agent = self.player_encoder(
        env_outputs["Entity"], env_outputs["AgentId"][:, 0]
    )

    if self.extra_encoders:
      inventory_embeddings = self.item_encoder(env_outputs["Inventory"])
      market_embeddings = self.item_encoder(env_outputs["Market"])

      inventory = self.inventory_encoder(inventory_embeddings)
      market = self.market_encoder(market_embeddings)

    obs = [tile, my_agent]
    lookup = []

    if self.extra_encoders:
      obs.extend([inventory, market])
      lookup.extend(
          [player_embeddings, inventory_embeddings, market_embeddings])

    if self.encode_task:
      task = self.task_encoder(env_outputs["Task"])
      if self.attend_task == "none":
        obs.append(task)
      lookup.append(task)

    obs = torch.cat(obs, dim=-1)
    obs = self.proj_fc(obs)

    if self.attend_task == "nikhil":
      obs, _ = self.task_attention(
          task.unsqueeze(0), obs.unsqueeze(0), obs.unsqueeze(0)
      )
      obs = obs.squeeze(0)
    elif self.attend_task == "pytorch":
      obs = torch.nn.functional.scaled_dot_product_attention(
          task.unsqueeze(0), obs.unsqueeze(0), obs.unsqueeze(0)
      )
      obs = obs.squeeze(0)

    return obs, (
        player_embeddings,
        inventory_embeddings,
        market_embeddings,
        env_outputs["ActionTargets"],
    )

  def decode_actions(self, hidden, lookup, concat=True):
    if self.attentional_decode:
      actions = self.action_decoder(hidden, lookup)
    else:
      actions = [decoder(hidden) for decoder in self.action_decoder]

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
