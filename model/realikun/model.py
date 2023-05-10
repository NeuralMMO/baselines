# TODO: remove the below line, eventually...
# pylint: disable=all
from attr import dataclass
from collections import OrderedDict

import torch
import torch.nn as nn

from lib.model.mlp import MLPEncoder
from lib.model.resnet import ResNet

def sort_dict_by_key(dict):
  return OrderedDict((key, dict[key]) for key in sorted(dict.keys()))

# Gather the model-related constants here
@dataclass
class ModelArchitecture:

  NUM_TEAMS = 16
  NUM_PLAYERS_PER_TEAM = 8
  INVENTORY_CAPACITY = 12

  # Observations
  TILE_NUM_CHANNELS = 7
  TILE_IMG_SIZE = [25, 25]

  ITEM_NUM_TYPES = 17 + 1 # 17 items + 1 for no item
  ITEM_NUM_FEATURES = 11

  ENTITY_NUM_FEATURES = 30
  ENTITY_NUM_NPCS_CONSIDERED = 9
  ENTITY_NUM_ENEMIES_CONSIDERED = 9

  # TODO: let the policy consider what to buy
  #MARKET_NUM_LISTINGS_CONSIDERED = 20

  # Actions
  # NOTE: The order of policy heads are the same as here,
  #   but gym.spaces.Dict sorts the keys, so the orders can be different
  #   So, sort_dict_by_key func was used to match these.
  ACTION_NUM_DIM = sort_dict_by_key({
    'move': 5, # 4 dirs + 1 for no move
    'style' : 3,
    # 9 npcs 9 enemies + 1 for no target
    'target': ENTITY_NUM_NPCS_CONSIDERED + ENTITY_NUM_ENEMIES_CONSIDERED + 1, # for no attack
    'use': INVENTORY_CAPACITY + 1, # for no use
    'destroy': INVENTORY_CAPACITY + 1, # for no destroy
    'sell': INVENTORY_CAPACITY + 1, # for no sell
  })

  # the game progress is encoded with multi-hot-generator
  PROGRESS_NUM_FEATURES = 16 # index=int(1+16*curr_step/config.HORIZON)
  # game_progress (1), n_alive/team_size (1), n_progress_feat, team_size
  GAME_NUM_FEATURES = 1 + 1 + PROGRESS_NUM_FEATURES + NUM_PLAYERS_PER_TEAM

  NEARBY_NUM_FEATURES = 246

  # Melee, Ranged, Magic - only considering combat types
  NUM_PROFESSIONS = 3

  TEAM_NUM_FEATURES = (
    ENTITY_NUM_FEATURES +
    NUM_TEAMS +
    NUM_PLAYERS_PER_TEAM +
    NUM_PROFESSIONS +
    NEARBY_NUM_FEATURES
  )

  SELF_NUM_FEATURES = (
    TEAM_NUM_FEATURES +
    GAME_NUM_FEATURES +
    sum(ACTION_NUM_DIM.values())
  )

  # Model
  SELF_EMBED = 512
  SELF_AS_ALLY_EMBED = 256
  LSTM_HIDDEN = 512
  ATTENTION_HIDDEN = 256


class TileEncoder(nn.Module):
  def __init__(self, in_ch, in_size):
    super().__init__()

    self.tile_net = ResNet(
      in_ch, in_size, channel_and_blocks=[[32, 2], [32, 2], [64, 2]])

    sample_in = torch.zeros(1, in_ch, *in_size)
    with torch.no_grad():
      self.h_size = len(self.tile_net(sample_in).flatten())

  def forward(self, x):
    bs, na = x['tile'].shape[:2]
    x_tile = x['tile'].contiguous().view(-1, *x['tile'].shape[2:])
    h_tile = self.tile_net(x_tile)
    h_tile = h_tile.view(bs, na, -1)  # flatten
    return h_tile


class ItemEncoder(nn.Module):
  def __init__(self, n_item_hidden):
    super().__init__()

    n_item_type = ModelArchitecture.ITEM_NUM_TYPES
    n_item_feature = ModelArchitecture.ITEM_NUM_FEATURES
    n_type_hidden = 32
    n_fc_in = n_item_feature + n_type_hidden
    self.type_embedding = nn.Embedding(n_item_type, n_type_hidden)
    self.post_fc = MLPEncoder(n_fc_in, n_hiddens=[n_item_hidden])

  def forward(self, x):
    bs, na, ni = x['item_type'].shape  # batch size, num agent, num item
    x_item_type = x['item_type'].long().view(-1, ni)
    h_type_embed = self.type_embedding(x_item_type).view(bs, na, ni, -1)
    h_item = torch.cat([h_type_embed, x['item']], dim=-1)
    h_item = self.post_fc(h_item)
    h_item = h_item.max(dim=2)[0]
    return h_item


class PrevActionEncoder(nn.Module):
  def __init__(self, n_legal, n_embed):
    super().__init__()

    self.n_legal = n_legal
    self.embedding = nn.ModuleDict({
      name: nn.Embedding(n_input, n_embed)
      for name, n_input in n_legal.items()
    })

  def forward(self, x):
    bs, na, nh = x['prev_act'].shape  # batch size, num actions, num action heads
    x_prev_act = x['prev_act'].long().view(-1, nh)
    h_embed = [
      self.embedding[name](x_prev_act[:, i])
      for i, name in enumerate(self.n_legal)
    ]
    h_embed = torch.cat(h_embed, dim=-1).view(bs, na, -1)
    return h_embed


class SelfEncoder(nn.Module):
  def __init__(self, in_img_ch, in_img_size, n_self_feat,
               n_legal, n_self_hidden):
    super().__init__()

    prev_act_embed_size = 16
    prev_act_hidden_size = len(n_legal) * prev_act_embed_size
    item_hidden_size = 128
    self.prev_act_embed = PrevActionEncoder(n_legal, prev_act_embed_size)
    self.item_net = ItemEncoder(item_hidden_size)
    self.img_net = TileEncoder(in_img_ch, in_img_size)
    mlp_input_size = (
      self.img_net.h_size +        # 1024
        n_self_feat +            # 318
        prev_act_hidden_size +   # 64
        item_hidden_size         # 128
      )

    self.mlp_net = MLPEncoder(mlp_input_size, n_hiddens=[n_self_hidden])

  def forward(self, x):
    batch_size, num_agents, _ = x['team'].shape
    h_tile = self.img_net(x)
    h_pre_act = self.prev_act_embed(x)
    h_item = self.item_net(x)
    h_self = torch.cat([
      h_tile,    # 1024
      x['team'], # 262
      h_item,    # 128
      h_pre_act, # 64
      *x['legal'].values(),
      x['game'].unsqueeze(1).expand(-1, num_agents, -1), # 26
    ], dim=-1)
    h_self = self.mlp_net(h_self)
    return h_self


class EntityEncoder(nn.Module):
  def __init__(self, entity_type, n_entity_feat, n_attn_hidden):
    super().__init__()

    self.entity_type = entity_type
    self.n_self_hidden_used = n_attn_hidden
    self.net = MLPEncoder(n_attn_hidden + n_entity_feat,
                          n_hiddens=[n_attn_hidden])

  def forward(self, x, h_self):
    x_entity = x[self.entity_type]
    bs, na, ne, nf = x_entity.shape

    h_self = h_self.unsqueeze(2).expand(-1, -1, ne, -1)
    h_self = h_self[:, :, :, :self.n_self_hidden_used]
    h_entity = torch.cat([h_self, x_entity], dim=-1)
    h_entity = self.net(h_entity)
    return h_entity


class InteractionBlock(nn.Module):
  def __init__(self, n_attn_hidden):
    super().__init__()

    self.n_attn_hidden = n_attn_hidden
    encoder_block = nn.TransformerEncoderLayer(
      d_model=n_attn_hidden, nhead=2, dim_feedforward=512, dropout=0.)
    self.transformer = nn.TransformerEncoder(encoder_block, num_layers=3)

  def forward(self, x, h_self, *hs_other):
    h_self = h_self.unsqueeze(2)[:, :, :, :self.n_attn_hidden]
    h = torch.cat([h_self, *hs_other], dim=2)
    bs, na, ne, nf = h.shape  # batch_size, num_agent, num_entity, num_feature
    mask = torch.cat([x['team_mask'], x['npc_mask'], x['enemy_mask']], dim=-1)
    mask = mask.to(bool).view(bs * na, ne)
    h = h.view(bs * na, ne, nf).transpose(0, 1)
    h = self.transformer(h, src_key_padding_mask=mask)
    h = h.transpose(0, 1).view(bs, na, ne, nf)
    h = h.max(dim=2)[0]
    return h


class MemoryBlock(nn.Module):
  def __init__(self, n_attn_hidden, n_lstm_hidden, num_layers):
    super().__init__()

    self.num_layers = num_layers
    self.hidden_size = n_lstm_hidden

    # Hardcoded for compatibility with CleanRL wrapper shape checks
    n_attn_hidden = 256
    n_lstm_hidden = 512

    self.n_attn_hidden = n_attn_hidden
    self.lstm = nn.LSTMCell(n_lstm_hidden, n_lstm_hidden, num_layers)

  def forward(self, x, state):
    # @daveey - Any idea on where seq_len comes from?
    seq_len, teams, _ = x.shape
    hxs, cxs = state
    h_self = self.h_self
    h_inter = self.h_inter

    hxs = hxs.view(-1, teams*8, 512)
    cxs = cxs.view(-1, teams*8, 512)

    h = torch.cat([h_self[:, :, self.n_attn_hidden:], h_inter], dim=-1)
    bs, na, nf = h.shape  # batch_size, num_agent, num_feature
    nt = bs // seq_len  # num of truncation
    resets = self.reset.repeat(1, na)
    h = h.view(nt, seq_len, na, -1).transpose(1, 0).reshape(seq_len, nt * na, -1)
    hys = [hxs[::seq_len].reshape(-1, nf)]  # [nt * na, nf]
    cys = [cxs[::seq_len].reshape(-1, nf)]
    for i in range(seq_len):
        hys[i] = hys[i] * (1 - resets[i::seq_len]).reshape(nt * na, 1)
        cys[i] = cys[i] * (1 - resets[i::seq_len]).reshape(nt * na, 1)
        hy, cy = self.lstm(h[i], (hys[i], cys[i]))
        hys.append(hy)
        cys.append(cy)
    hys = torch.stack(hys[1:], dim=0)  # [seq_len, nt * na, nf]
    cys = torch.stack(cys[1:], dim=0)
    hys = hys.view(seq_len, nt, na, -1).transpose(1, 0).reshape(bs, na, -1)
    cys = cys.view(seq_len, nt, na, -1).transpose(1, 0).reshape(bs, na, -1)

    hys = hys.view(seq_len, nt, -1)
    cys = cys.view(seq_len, nt, -1)

    #bs, na, nf = h.shape # batch_size * num_agent, num_feature
    return hys, (hys, cys)


class PolicyHead(nn.Module):
  def __init__(self, n_hidden, n_legal):
    super().__init__()

    self.heads = nn.ModuleList([
        nn.Linear(n_hidden, n_output) for n_output in n_legal.values()
    ])

  def forward(self, h):
    return [head(h) for head in self.heads]
