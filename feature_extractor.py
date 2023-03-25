from typing import Any, Dict, List

import nmmo
from team_env import TeamEnv
import collections
import copy

import nmmo.io.action as nmmo_act
import numpy as np

from model.const import *
from model.util import one_hot_generator, multi_hot_generator
from pettingzoo.utils.env import AgentID, ParallelEnv
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.lib import material
from nmmo.systems import item
from nmmo.systems.item import ItemState
from team_helper import TeamHelper

DEFOGGING_VALUE = 16
VISITATION_MEMORY = 100

N_CH = 7
IMG_SIZE = 25
DUMMY_IMG_FEAT = np.zeros((N_CH, IMG_SIZE, IMG_SIZE))
X_IMG = np.arange(TERRAIN_SIZE+1).repeat(TERRAIN_SIZE+1).reshape(TERRAIN_SIZE+1, TERRAIN_SIZE+1)
Y_IMG = X_IMG.transpose(1, 0)
TEAMMATE_REPR = 1 / 5.
ENEMY_REPR = 2 / 5.
NEGATIVE_REPR = 3 / 5.
NEUTRAL_REPR = 4 / 5.
HOSTILE_REPR = 1.

N_NPC_CONSIDERED = 9
N_ENEMY_CONSIDERED = 9
AWARE_RANGE = 15
PER_ITEM_FEATURE = 11
N_SELF_FEATURE = 262
PER_ENTITY_FEATURE = 30
DUMMY_ITEM_TYPES = np.zeros(N_ITEM_SLOT)
DUMMY_ITEM_ARRS = np.zeros((N_ITEM_SLOT, PER_ITEM_FEATURE))

N_MOVE = 4
N_ATK_TARGET = N_NPC_CONSIDERED + N_ENEMY_CONSIDERED
N_ATK_TYPE = 3
N_USE = 2
N_SELL = 2

USE_POULTICE = 0
USE_RATION = 1
SELL_POULTICE = 0
SELL_RATION = 1

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

ITEM_TO_PROF_IDX = {
    item.Sword.ITEM_TYPE_ID: EntityAttr["melee_level"],
    item.Bow.ITEM_TYPE_ID: EntityAttr["range_level"],
    item.Wand.ITEM_TYPE_ID: EntityAttr["mage_level"],
    item.Scrap.ITEM_TYPE_ID: EntityAttr["melee_level"],
    item.Shaving.ITEM_TYPE_ID:EntityAttr["range_level"],
    item.Shard.ITEM_TYPE_ID: EntityAttr["mage_level"],
    item.Rod.ITEM_TYPE_ID: EntityAttr["fishing_level"],
    item.Gloves.ITEM_TYPE_ID: EntityAttr["herbalism_level"],
    item.Pickaxe.ITEM_TYPE_ID: EntityAttr["prospecting_level"],
    item.Chisel.ITEM_TYPE_ID: EntityAttr["carving_level"],
    item.Arcane.ITEM_TYPE_ID: EntityAttr["alchemy_level"],
}

DEPLETION_MAP = {
    material.Forest.index: material.Scrub.index,
    material.Tree.index: material.Stump.index,
    material.Ore.index: material.Slag.index,
    material.Crystal.index: material.Fragment.index,
    material.Herb.index: material.Weeds.index,
    material.Fish.index: material.Ocean.index,
}


ARMORS = {
  item.Hat.ITEM_TYPE_ID,
  item.Top.ITEM_TYPE_ID,
  item.Bottom.ITEM_TYPE_ID,
}

WEAPONS = {
  item.Sword.ITEM_TYPE_ID,
  item.Wand.ITEM_TYPE_ID,
  item.Bow.ITEM_TYPE_ID,
}

TOOLS = {
  item.Rod.ITEM_TYPE_ID,
  item.Gloves.ITEM_TYPE_ID,
  item.Pickaxe.ITEM_TYPE_ID,
  item.Chisel.ITEM_TYPE_ID,
  item.Arcane.ITEM_TYPE_ID,
}

AMMOS = {
  item.Scrap.ITEM_TYPE_ID,
  item.Shaving.ITEM_TYPE_ID,
  item.Shard.ITEM_TYPE_ID,
}

CONSUMABLES = {
  item.Ration.ITEM_TYPE_ID,
  item.Poultice.ITEM_TYPE_ID,
}

ATK_TO_WEAPON = {
  'Melee': item.Sword.ITEM_TYPE_ID,
  'Range': item.Bow.ITEM_TYPE_ID,
  'Mage': item.Wand.ITEM_TYPE_ID
}

ATK_TO_TOOL = {
  'Melee': item.Pickaxe.ITEM_TYPE_ID,
  'Range': item.Chisel.ITEM_TYPE_ID,
  'Mage': item.Arcane.ITEM_TYPE_ID
}

ATK_TO_TILE = {
  'Melee': material.Ore.index,
  'Range': material.Tree.index,
  'Mage': material.Crystal.index
}

PROF_TO_ATK_TYPE = {
  'Melee': 0,
  'Range': 1,
  'Mage': 2,
}

class FeatureExtractor():
  def __init__(self, config: nmmo.config.Config, team_helper: TeamHelper, team_id: int):
    self.config = config
    self.team_id = team_id
    self.team_size = team_helper.team_size[team_id]
    self.num_teams = team_helper.num_teams

    self.curr_step = None
    self.curr_obs = None
    self.prev_obs = None

    self.tile_map = None
    self.fog_map = None
    self.visit_map = None
    self.poison_map = None
    self.entity_map = None

    self.npc_tgt = None
    self.enemy_tgt = None

    self.target_entity_id = None
    self.target_entity_pop = None
    self.player_kill_num = None  # for comparing with playerDefeat stat only
    self.npc_kill_num = None  # statistics for reward only
    self.step_onto_herb_cnt = None  # statistics for reward only

    self.prof = None
    self.best_hats = None
    self.best_tops = None
    self.best_bottoms = None
    self.best_weapons = None
    self.best_tools = None
    self.force_use_idx = None
    self.force_sell_idx = None
    self.force_sell_price = None
    self.force_buy_idx = None
    self.rescue_cooldown = None  # cooldown rounds of sell poultice to teammates

    self.prev_actions = None

  def reset(self, init_obs):
    self.curr_step = 0
    self.prev_obs = init_obs
    self.rescue_cooldown = 0

    self.tile_map = get_init_tile_map()
    self.fog_map = np.zeros((TERRAIN_SIZE+1, TERRAIN_SIZE+1))
    self.visit_map = np.zeros((self.team_size, TERRAIN_SIZE+1, TERRAIN_SIZE+1))
    self.poison_map = get_init_poison_map()
    self.entity_map = None

    self.target_entity_id = [None] * self.team_size
    self.target_entity_pop = [None] * self.team_size
    self.player_kill_num = [0] * self.team_size

    # p: passive, n: neutral, h: hostile
    self.npc_kill_num = {kind: [0] * self.team_size for kind in 'pnh'}

    self.step_onto_herb_cnt = [0] * self.team_size

    self.prof = self._choose_prof()

    self.prev_actions = np.array([N_MOVE, N_ATK_TARGET, N_USE, N_SELL])[None, :] \
                          .repeat(self.team_size, axis=0)  # init as idle

  def trans_obs(self, obs):
    self.curr_obs = obs
    self.rescue_cooldown = max(0, self.rescue_cooldown - 1)

    self._update_global_maps(obs)
    self._update_kill_num(obs)
    tile = self._extract_tile_feature(obs)
    self._process_items(obs)  # use & sell
    self._process_market(obs)  # buy
    item_type, item = self._extract_item_feature(obs)
    team, npc, enemy, *masks, self.npc_tgt, self.enemy_tgt = self._extract_entity_feature(obs)
    game = self._extract_game_feature(obs)
    # legal = self._extract_legal_action(obs, self.npc_tgt, self.enemy_tgt)
    reset_flag = np.array([self.curr_step == 0])  # for resetting RNN hidden

    state = {
      'tile': tile,
      'item_type': item_type,
      'item': item,
      'team': team,
      'npc': npc,
      'enemy': enemy,
      'team_mask': masks[0],
      'npc_mask': masks[1],
      'enemy_mask': masks[2],
      'game': game,
      # 'legal': legal,
      'prev_act': self.prev_actions,
      'reset': reset_flag,
    }
    return state

  def _update_global_maps(self, obs):
    if self.curr_step % 16 == 15:
      self.poison_map += 1  # poison shrinking

    self.fog_map = np.clip(self.fog_map - 1, 0, DEFOGGING_VALUE)  # decay
    self.visit_map = np.clip(self.visit_map - 1, 0, VISITATION_MEMORY)  # decay

    entity_map = np.zeros((5, TERRAIN_SIZE+1, TERRAIN_SIZE+1))

    for player_id, player_obs in obs.items():
      # mark tile
      tile_obs = player_obs['Tile']
      tile_pos = tile_obs[:, TileAttr["row"]:TileAttr["col"]+1].astype(int)
      tile_type = tile_obs[:, TileAttr["material_id"]].astype(int)
      mark_point(self.fog_map, tile_pos, DEFOGGING_VALUE)
      x, y = tile_pos[0]
      self.tile_map[x:x+WINDOW, y:y+WINDOW] = tile_type.reshape(WINDOW, WINDOW)

      # mark team/enemy/npc
      entity_obs = player_obs['Entity']
      entity_pos = entity_obs[:, EntityAttr["row"]:EntityAttr["col"]+1].astype(int)
      entity_pop = entity_obs[:, EntityAttr["population_id"]].astype(int)

      mark_point(entity_map[0], entity_pos, entity_pop == self.team_id)  # team
      mark_point(entity_map[1], entity_pos, np.logical_and(entity_pop != self.team_id, entity_pop > 0))  # enemy
      mark_point(entity_map[2], entity_pos, entity_pop == -1)  # negative
      mark_point(entity_map[3], entity_pos, entity_pop == -2)  # neutral
      mark_point(entity_map[4], entity_pos, entity_pop == -3)  # hostile

      # update visit map
      # xcxc my position
      mark_point(self.visit_map[player_id], entity_pos[:1], VISITATION_MEMORY)

      # update herb gathering count
      my_curr_pos = entity_obs[0, 7:9].astype(int)
      my_prev_pos = self.prev_obs[player_id]['Entity'][0, 7:9].astype(int)

      if self.tile_map[my_curr_pos[0], my_curr_pos[1]] == material.Herb.index and \
          (my_curr_pos[0] != my_prev_pos[0] or my_curr_pos[1] != my_prev_pos[1]):
        self.step_onto_herb_cnt[player_id] += 1

      # change tile in advance
      for pos, pop in zip(entity_pos, entity_pop):
        if pop >= 0:  # is player
          new_tile = DEPLETION_MAP.get(self.tile_map[pos[0], pos[1]])
          if new_tile is not None:
            self.tile_map[pos[0], pos[1]] = new_tile

            for row_offset in range(-1, 2):
              for col_offset in range(-1, 2):
                if self.tile_map[pos[0]+row_offset, pos[1]+col_offset] == material.Fish.index:
                  self.tile_map[pos[0]+row_offset, pos[1]+col_offset] = material.Ocean.index

    self.entity_map = entity_map[0] * TEAMMATE_REPR + entity_map[1] * ENEMY_REPR + \
      entity_map[2] * NEGATIVE_REPR + entity_map[3] * NEUTRAL_REPR + entity_map[4] * HOSTILE_REPR

  def _update_kill_num(self, obs):
    for player_id, player_obs in obs.items():
      if self.target_entity_id[player_id] is None:  # no target
          continue
      entity_obs = player_obs['Entity']
      entity_in_sight = entity_obs[:, EntityAttr["id"]]
      if self.target_entity_id[player_id] not in entity_in_sight:
        if self.target_entity_id[player_id] > 0:
          self.player_kill_num[player_id] += 1
        elif self.target_entity_id[player_id] < 0:
          if self.target_entity_pop[player_id] == -1:
            self.npc_kill_num['p'][player_id] += 1
          elif self.target_entity_pop[player_id] == -2:
            self.npc_kill_num['n'][player_id] += 1
          elif self.target_entity_pop[player_id] == -3:
            self.npc_kill_num['h'][player_id] += 1
          else:
            raise ValueError('Unknown npc pop:', self.target_entity_pop[player_id])

  def _extract_tile_feature(self, obs):
    imgs = []
    for i in range(self.team_size):
      # replace with dummy feature if dead
      if i not in obs:
          imgs.append(DUMMY_IMG_FEAT)
          continue

      curr_pos = obs[i]['Entity'][0, 7:9].astype(int)
      l, r = curr_pos[0] - IMG_SIZE // 2, curr_pos[0] + IMG_SIZE // 2 + 1
      u, d = curr_pos[1] - IMG_SIZE // 2, curr_pos[1] + IMG_SIZE // 2 + 1
      tile_img = self.tile_map[l:r, u:d] / N_TILE_TYPE
      # obstacle_img = np.sum([self.tile_map[l:r, u:d] == t for t in OBSTACLE_TILES], axis=0)
      entity_img = self.entity_map[l:r, u:d]
      poison_img = np.clip(self.poison_map[l:r, u:d], 0, np.inf) / 20.
      fog_img = self.fog_map[l:r, u:d] / DEFOGGING_VALUE
      # view_img = (fog_img == 1.).astype(np.float32)
      visit_img = self.visit_map[i][l:r, u:d] / VISITATION_MEMORY
      coord_imgs = [X_IMG[l:r, u:d] / TERRAIN_SIZE, Y_IMG[l:r, u:d] / TERRAIN_SIZE]
      img = np.stack([tile_img, entity_img, poison_img, fog_img, visit_img, *coord_imgs])
      imgs.append(img)
    imgs = np.stack(imgs)
    return imgs

  def _process_items(self, obs):
    # reset
    self.best_hats = [None] * self.team_size
    self.best_tops = [None] * self.team_size
    self.best_bottoms = [None] * self.team_size
    self.best_weapons = [None] * self.team_size
    self.best_tools = [None] * self.team_size
    self.force_use_idx = [None] * self.team_size
    self.force_sell_idx = [None] * self.team_size
    self.force_sell_price = [None] * self.team_size

    for i in range(self.team_size):
      if i not in obs:
        continue
      item_obs = obs[i]['Inventory']
      my_obs = obs[i]['Entity'][0]

      # force use weapons and armors
      usable_types = [  # reflect priority
          ATK_TO_WEAPON[self.prof[i]],
          item.Hat.ITEM_TYPE_ID,
          item.Top.ITEM_TYPE_ID,
          item.Bottom.ITEM_TYPE_ID,
      ]
      best_savers = [
          self.best_weapons,
          self.best_hats,
          self.best_tops,
          self.best_bottoms,
      ]
      for item_type, saver in zip(usable_types, best_savers):
          if item_type in ARMORS:
              max_equipable_lvl = max(my_obs[-N_PROF:])  # maximum of all levels
          else:
              max_equipable_lvl = my_obs[ITEM_TO_PROF_IDX[item_type]]
          items = item_obs[item_obs[:, ItemAttr["type_id"]] == item_type]  # those with the target type
          items = items[items[:, ItemAttr["level"]] <= max_equipable_lvl]  # those within the equipable level
          if len(items) > 0:
              max_item_lvl = max(items[:, ItemAttr["level"]])
              items = items[items[:, ItemAttr["level"]] == max_item_lvl]  # those with highest level
              min_id = min(items[:, ItemAttr["id"]])  # always choose the one with the minimal item id as the best
              idx = np.argwhere(item_obs[:, ItemAttr["id"]] == min_id).item()
              saver[i] = item_obs[idx]
              if not item_obs[idx][ItemAttr["equipped"]] and self.force_use_idx[i] is None:
                  self.force_use_idx[i] = idx  # save for later translation

      # force use tools
      if self.best_weapons[i] is None:
        tools = []
        for tool_type in TOOLS:
          tools_ = item_obs[item_obs[:, ItemAttr["type_id"]] == tool_type]
          max_equipable_lvl = my_obs[ITEM_TO_PROF_IDX[tool_type]]
          tools_ = tools_[tools_[:, ItemAttr["level"]] <= max_equipable_lvl]
          tools.append(tools_)
        tools = np.concatenate(tools)
        if len(tools) > 0:
          max_tool_lvl = max(tools[:, ItemAttr["level"]])
          tools = tools[tools[:, ItemAttr["level"]] == max_tool_lvl]  # those with highest level
          min_id = min(tools[:, ItemAttr["id"]])  # always choose the one with the minimal item id as the best
          idx = np.argwhere(item_obs[:, ItemAttr["id"]] == min_id).item()
          self.best_tools[i] = item_obs[idx]
          if not item_obs[idx][ItemAttr["equipped"]] and self.force_use_idx[i] is None:
            self.force_use_idx[i] = idx  # save for later translation

      # directly sell ammo
      items = np.concatenate([
        item_obs[item_obs[:, ItemAttr["type_id"]] == ammo_type]
        for ammo_type in AMMOS
      ], axis=0)
      if len(items) > 0:
        item_id = items[0][ItemAttr["id"]]
        item_lvl = items[0][ItemAttr["level"]]
        idx = np.argwhere(item_obs[:, ItemAttr["id"]] == item_id).item()
        self.force_sell_idx[i] = idx
        self.force_sell_price[i] = max(1, int(item_lvl) - 1)
        continue

      # directly sell weapons not belong to my profession
      other_weapon_types = [w for w in WEAPONS
                            if w != ATK_TO_WEAPON[self.prof[i]]]
      items = np.concatenate([
        item_obs[item_obs[:, ItemAttr["type_id"]] == weapon_type]
        for weapon_type in other_weapon_types
      ], axis=0)
      if len(items) > 0:
        item_id = items[0][ItemAttr["id"]]
        item_lvl = items[0][ItemAttr["level"]]
        idx = np.argwhere(item_obs[:, ItemAttr["id"]] == item_id).item()
        self.force_sell_idx[i] = idx
        self.force_sell_price[i] = min(99, np.random.randint(10) + int(item_lvl) * 25 - 10)
        continue

      # sell tools
      items = np.concatenate([
        item_obs[item_obs[:, ItemAttr["type_id"]] == tool_type]
        for tool_type in TOOLS
      ], axis=0)
      if self.best_weapons[i] is None and self.best_tools[i] is not None:
        best_idx = self.best_tools[i][ItemAttr["id"]]
        items = items[items[:, ItemAttr["id"]] != best_idx]  # filter out the best
      if len(items) > 0:
        to_sell = sorted(items, key=lambda x: x[ItemAttr["level"]])[0]  # sell the worst first
        idx = np.argwhere(item_obs[:, ItemAttr["id"]] == to_sell[ItemAttr["id"]]).item()
        lvl = to_sell[ItemAttr["level"]]
        self.force_sell_idx[i] = idx
        self.force_sell_price[i] = int(lvl) * 6 + np.random.randint(3)
        continue

      # sell armors and weapons of my profession
      sell_not_best_types = [
        item.Hat.ITEM_TYPE_ID,
        item.Top.ITEM_TYPE_ID,
        item.Bottom.ITEM_TYPE_ID,
        ATK_TO_WEAPON[self.prof[i]],
      ]  # reflect priority
      best_savers = [
        self.best_hats,
        self.best_tops,
        self.best_bottoms,
        self.best_weapons,
      ]
      for item_type, saver in zip(sell_not_best_types, best_savers):
        items = item_obs[item_obs[:, ItemAttr["type_id"]] == item_type]  # those with the target type
        if saver[i] is not None:
          items = items[items[:, ItemAttr["id"]] != saver[i][ItemAttr["id"]]]  # filter out the best
          best_lvl = saver[i][ItemAttr["level"]]
          if best_lvl < 6:
            # reserve items no more than level 6 for future use
            reserves = items[items[:, ItemAttr["level"]] > best_lvl]
            reserves = reserves[reserves[:, ItemAttr["level"]] <= 6]
            if len(reserves) > 0:
              reserve = sorted(reserves, key=lambda x: x[ItemAttr["level"]])[-1]  # the best one to reserve
              items = items[items[:, ItemAttr["id"]] != reserve[ItemAttr["id"]]]  # filter out the reserved
        if len(items) > 0:
          to_sell = sorted(items, key=lambda x: x[ItemAttr["level"]])[0]  # sell the worst first
          idx = np.argwhere(item_obs[:, ItemAttr["id"]] == to_sell[ItemAttr["id"]]).item()
          lvl = to_sell[ItemAttr["level"]]
          self.force_sell_idx[i] = idx
          if item_type in WEAPONS:
            self.force_sell_price[i] = min(99, np.random.randint(10) + int(lvl) * 25 - 10)
          else:  # ARMORS
            self.force_sell_price[i] = 3 + np.random.randint(3) + int(lvl - 1) * 4
          break

  def _process_market(self, obs):
      # reset
      self.force_buy_idx = [None] * self.team_size

      alive_ids = np.array([i for i in range(self.team_size) if i in obs])
      raw_market_obs = obs[alive_ids[0]]['Market']
      market_obs = copy.deepcopy(raw_market_obs)  # will be modified later

      # combat rating
      ratings = []
      for i in alive_ids:
          rating = 0
          if self.best_weapons[i] is not None:
              rating += self.best_weapons[i][ItemAttr["level"]] * 10
          elif self.best_tools[i] is not None:
              rating += self.best_tools[i][ItemAttr["level"]] * 4
          if self.best_hats[i] is not None:
              rating += self.best_hats[i][ItemAttr["level"]] * 4
          if self.best_tops[i] is not None:
              rating += self.best_tops[i][ItemAttr["level"]] * 4
          if self.best_bottoms[i] is not None:
              rating += self.best_bottoms[i][ItemAttr["level"]] * 4
          ratings.append(rating)
      ratings = np.array(ratings)

      # buy weapons & armors
      agent_order = np.argsort(ratings)
      alive_ids = alive_ids[agent_order]  # reorder, low rating buy first
      for i in alive_ids:
          my_obs = obs[i]['Entity'][0]
          my_items = obs[i]['Inventory']
          my_gold = my_obs[EntityAttr["gold"]]

          care_types = [ATK_TO_WEAPON[self.prof[i]], *ARMORS]
          savers = [
              self.best_weapons,
              self.best_hats,
              self.best_tops,
              self.best_bottoms,
          ]

          wishlist = []
          enhancements = []
          for typ, saver in zip(care_types, savers):
              if typ in ARMORS:
                  max_equipable_lvl = max(my_obs[-N_PROF:])  # maximum of all levels
              else:
                  max_equipable_lvl = my_obs[ITEM_TO_PROF_IDX[typ]]
              curr_best_lvl = 0
              if saver[i] is not None:
                  curr_best_lvl = saver[i][ItemAttr["level"]]
              mkt_comds = market_obs[market_obs[:, ItemAttr["type_id"]] == typ]
              mkt_comds = mkt_comds[mkt_comds[:, ItemAttr["listed_price"]] <= my_gold]
              mkt_comds = mkt_comds[mkt_comds[:, ItemAttr["level"]] <= max_equipable_lvl]
              mkt_comds = mkt_comds[mkt_comds[:, ItemAttr["level"]] > curr_best_lvl]
              if len(mkt_comds) > 0:
                  best_comd = sorted(mkt_comds, key=lambda x: x[ItemAttr["level"]])[-1]
                  wishlist.append(best_comd)
                  best_lvl = best_comd[ItemAttr["level"]]
                  delta_per_lvl = 4 if typ not in WEAPONS else 10
                  delta = delta_per_lvl * (best_lvl - curr_best_lvl)
                  enhancements.append(delta)
              else:
                  wishlist.append(None)
                  enhancements.append(0)

          if max(enhancements) > 0:
              to_buy = wishlist[enhancements.index(max(enhancements))]
              self.force_buy_idx[i] = np.argwhere(raw_market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
              idx = np.argwhere(market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
              market_obs[idx][ItemAttr["quantity"]] -= 1
              if market_obs[idx][ItemAttr["quantity"]] == 0:
                  # remove from market obs to prevent competition among teammates
                  market_obs = np.concatenate([market_obs[:idx], market_obs[idx+1:]])

      # survival rating
      alive_ids = np.array([i for i in range(self.team_size) if i in obs])
      ratings = []
      for i in alive_ids:
          rating = 0
          my_items = obs[i]['Inventory']
          n_poultice = len(my_items[my_items[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID])
          n_ration = len(my_items[my_items[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID])
          if n_poultice == 1:
              rating += 4
          elif n_poultice >= 2:
              rating += 8
          if n_ration >= 1:
              rating += 2
          ratings.append(rating)
      ratings = np.array(ratings)

      # emergent case to buy poultice
      will_die_id = None
      will_die_gold = None
      team_n_poultice = [None] * self.team_size
      for i in alive_ids:
          my_obs = obs[i]['Entity'][0]
          my_items = obs[i]['Inventory']
          my_poultices = my_items[my_items[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
          team_n_poultice[i] = len(my_poultices)
          if len(my_poultices) > 0:  # not emergent
              continue
          if my_obs[EntityAttr["health"]] > 35:  # not emergent
              continue
          # if my_obs[EntityAttr["food"]] >= 50 and my_obs[EntityAttr["water"]] >= 50:
          #     my_pop = my_obs[EntityAttr["population_id"]]
          #     entity_obs = obs[i]['Entity']
          #     n_ent_observed = obs[i]['Entity']['N'][0]
          #     other_entities = entity_obs[1:n_ent_observed]
          #     other_enemies =
          my_gold = my_obs[EntityAttr["gold"]]
          mkt_ps = market_obs[market_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
          mkt_ps = mkt_ps[mkt_ps[:, ItemAttr["listed_price"]] <= my_gold]
          if len(mkt_ps) > 0:
              to_buy = sorted(mkt_ps, key=lambda x: x[ItemAttr["listed_price"]])[0]  # cheapest
              self.force_buy_idx[i] = np.argwhere(raw_market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
              idx = np.argwhere(market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
              market_obs[idx][ItemAttr["quantity"]] -= 1
              if market_obs[idx][ItemAttr["quantity"]] == 0:
                  # remove from market obs to prevent repeatedly buying among teammates
                  market_obs = np.concatenate([market_obs[:idx], market_obs[idx+1:]])
          else:
              will_die_id = i
              will_die_gold = my_gold

      # sell poultice for emergent rescue
      if will_die_id is not None and self.rescue_cooldown == 0 and will_die_gold > 0:
          for i in reversed(alive_ids):
              if team_n_poultice[i] > 0 and self.force_sell_idx[i] is not None:
                  my_items = obs[i]['Inventory']
                  my_poultices = my_items[my_items[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
                  team_pop = next(iter(obs.values()))['Entity'][0][EntityAttr["population_id"]]
                  to_sell = sorted(my_poultices, key=lambda x: x[ItemAttr["level"]])[-1]  # sell the best
                  idx = np.argwhere(my_items[:, ItemAttr["id"]] == to_sell[ItemAttr["id"]]).item()
                  self.force_sell_idx[i] = idx
                  self.force_sell_price[i] = max(int(will_die_gold // 2), 1) if team_pop > 0 else 1
                  self.rescue_cooldown = 3
                  break

      # normal case to buy at least two poultices, at least one ration
      agent_order = np.argsort(ratings)
      alive_ids = alive_ids[agent_order]  # reorder, low rating buy first
      for cons_type in [item.Poultice.ITEM_TYPE_ID, item.Ration.ITEM_TYPE_ID]:
          for i in alive_ids:
              if self.force_buy_idx[i] is not None:
                  continue
              my_items = obs[i]['Inventory']
              if cons_type == item.Ration.ITEM_TYPE_ID and len(my_items[my_items[:, ItemAttr["type_id"]] == cons_type]) >= 1:
                  continue
              if cons_type == item.Poultice.ITEM_TYPE_ID and len(my_items[my_items[:, ItemAttr["type_id"]] == cons_type]) >= 2:
                  continue
              mkt_cons = market_obs[market_obs[:, ItemAttr["type_id"]] == cons_type]
              acceptable_price = 2 + self.curr_step // 300
              mkt_cons = mkt_cons[mkt_cons[:, ItemAttr["listed_price"]] <= acceptable_price]
              if len(mkt_cons) > 0:
                  to_buy = sorted(mkt_cons, key=lambda x: x[ItemAttr["listed_price"]])[0]  # cheapest
                  self.force_buy_idx[i] = np.argwhere(raw_market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
                  idx = np.argwhere(market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
                  market_obs[idx][ItemAttr["quantity"]] -= 1
                  if market_obs[idx][ItemAttr["quantity"]] == 0:
                      # remove from market obs to prevent repeatedly buying among teammates
                      market_obs = np.concatenate([market_obs[:idx], market_obs[idx + 1:]])

  def _extract_item_feature(self, obs):
      items_arrs = []
      items_types = []
      for i in range(self.team_size):
          # replace with dummy feature if dead
          if i not in obs:
              items_types.append(DUMMY_ITEM_TYPES)
              items_arrs.append(DUMMY_ITEM_ARRS)
              continue

          item_obs = obs[i]['Inventory']
          item_arrs = []
          item_types = []
          for j in range(N_ITEM_SLOT):
              o = item_obs[j]
              item_types.append(o[ItemAttr["type_id"]])  # type is 0 if j < n
              arr = self._extract_per_item_feature(o)
              item_arrs.append(arr)
          item_types = np.array(item_types)
          item_arrs = np.stack(item_arrs)

          items_types.append(item_types)
          items_arrs.append(item_arrs)
      items_types = np.stack(items_types)
      items_arrs = np.stack(items_arrs)

      return items_types, items_arrs

  @staticmethod
  def _extract_per_item_feature(o):
    if o is not None:
      arr = np.array([
        o[ItemAttr["level"]] / 10.,
        o[ItemAttr["quantity"]] / 10.,
        o[ItemAttr["melee_attack"]] / 100.,
        o[ItemAttr["range_attack"]] / 100.,
        o[ItemAttr["mage_attack"]] / 100.,
        o[ItemAttr["melee_defense"]] / 40.,
        o[ItemAttr["range_defense"]] / 40.,
        o[ItemAttr["mage_defense"]] / 40.,
        o[ItemAttr["health_restore"]] / 100.,
        o[ItemAttr["listed_price"]] / 100.,
        o[ItemAttr["equipped"]],
      ])
    else:
      arr = np.zeros(PER_ITEM_FEATURE)
    return arr

  def _extract_entity_feature(self, obs):
    team_pop = next(iter(obs.values()))['Entity'][0][EntityAttr["population_id"]]
    team_members_idx = np.arange(8) + team_pop * self.team_size + 1

    # merge obs from all the 8 agents
    team_members = {}  # 0~7 -> raw_arr
    enemies = {}  # entity_id -> raw_arr
    npcs = {}  # entity_id -> raw_arr
    for i in range(self.team_size):
        if i not in obs:
            continue
        entity_obs = obs[i]['Entity']
        team_members[i] = entity_obs[0]
        for j in range(1, self.team_size):
            if entity_obs[j][EntityAttr["id"]] < 0:
                npcs[entity_obs[j][EntityAttr["id"]]] = entity_obs[j]
            elif entity_obs[j][EntityAttr["id"]] not in team_members_idx:
                enemies[entity_obs[j][EntityAttr["id"]]] = entity_obs[j]

    # extract feature of each team member itself
    team_members_arr = np.zeros((self.team_size, N_SELF_FEATURE))
    team_mask = np.array([i not in obs for i in range(self.team_size)])
    for i in range(self.team_size):
        team_members_arr[i] = self._extract_per_entity_feature(team_members.get(i, None), team_pop, i)

    # assign the features of npcs and enemies to each member
    others_arrs = [np.zeros((self.team_size, n, PER_ENTITY_FEATURE))
                    for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
    entity_mask = [np.ones((self.team_size, n))
                    for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
    ids_as_target = [np.zeros((self.team_size, n))
                      for n in (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)]
    for k in range(2):
        n_considered = (N_NPC_CONSIDERED, N_ENEMY_CONSIDERED)[k]
        entities = (npcs, enemies)[k]
        # first extract all the features along with entity's idx & position
        features = [{
            'idx': idx,
            'row': raw_arr[EntityAttr["row"]],
            'col': raw_arr[EntityAttr["col"]],
            'pop': raw_arr[EntityAttr["population_id"]],
            'arr': self._extract_per_entity_feature(raw_arr, team_pop),
        } for idx, raw_arr in entities.items()]

        for i in range(self.team_size):
            if i not in team_members:  # dead
                continue
            my_row = team_members[i][EntityAttr["row"]]
            my_col = team_members[i][EntityAttr["col"]]

            def l1_to_me(f):
                return max(abs(f['row'] - my_row), abs(f['col'] - my_col))

            nearests = sorted(features, key=l1_to_me)[:n_considered]
            for j, feat in enumerate(nearests):
                if l1_to_me(feat) <= ATK_RANGE and feat['pop'] != NEUTRAL_POP:  # as target
                    ids_as_target[k][i][j] = feat['idx']
                if l1_to_me(feat) <= AWARE_RANGE:  # as visible entity
                    others_arrs[k][i][j] = feat['arr']
                    entity_mask[k][i][j] = 0

    npcs_arrs, enemies_arrs = others_arrs
    target_npcs_ids, target_enemies_ids = ids_as_target
    return team_members_arr, npcs_arrs, enemies_arrs, team_mask, entity_mask[0], entity_mask[1], target_npcs_ids, target_enemies_ids

  def _extract_per_entity_feature(self, o, team_pop=None, my_index=None):
      if o is not None:
          arr = np.array([
              1.,  # alive mark
              o[EntityAttr["id"]] in self.target_entity_id,  # attacked by my team
              o[EntityAttr["attacker_id"]] < 0,  # attacked by npc
              o[EntityAttr["attacker_id"]] > 0,  # attacked by player
              o[EntityAttr["item_level"]] / 10., # xcxc ent_level is no more
              o[EntityAttr["item_level"]] / 20.,
              (o[EntityAttr["row"]] - HALF_TERRAIN_SIZE) / HALF_TERRAIN_SIZE,
              (o[EntityAttr["col"]] - HALF_TERRAIN_SIZE) / HALF_TERRAIN_SIZE,
              o[EntityAttr["time_alive"]] / MAX_STEP,
              (o[EntityAttr["row"]] - MAP_LEFT) / MAP_SIZE,
              (o[EntityAttr["col"]] - MAP_LEFT) / MAP_SIZE,
              o[EntityAttr["id"]] >= 0,  # player
              o[EntityAttr["population_id"]] == team_pop,  # is teammate
              o[EntityAttr["population_id"]] == -1,  # passive npc
              o[EntityAttr["population_id"]] == -2,  # neutral npc
              o[EntityAttr["population_id"]] == -3,  # hostile npc
              o[EntityAttr["damage"]] / 10.,
              o[EntityAttr["time_alive"]] / MAX_STEP,
              o[EntityAttr["gold"]] / 100.,
              o[EntityAttr["health"]] / 100.,
              o[EntityAttr["food"]] / 100.,
              o[EntityAttr["water"]] / 100.,
              o[EntityAttr["melee_level"]] / 10.,
              o[EntityAttr["range_level"]] / 10.,
              o[EntityAttr["mage_level"]] / 10.,
              o[EntityAttr["fishing_level"]] / 10.,
              o[EntityAttr["herbalism_level"]] / 10.,
              o[EntityAttr["prospecting_level"]] / 10.,
              o[EntityAttr["carving_level"]] / 10.,
              o[EntityAttr["alchemy_level"]] / 10.,
          ])
      else:
          arr = np.zeros(PER_ENTITY_FEATURE)

      if my_index is not None:
          population_arr = one_hot_generator(self.num_teams, int(team_pop))
          index_arr = one_hot_generator(self.team_size, my_index)
          prof_idx = ATK_TYPE.index(self.prof[my_index])
          prof_arr = one_hot_generator(N_ATK_TYPE, prof_idx)
          if o is not None:
              row = o[EntityAttr["row"]].astype(int)
              col = o[EntityAttr["col"]].astype(int)
              near_tile_map = self.tile_map[row-4:row+5, col-4:col+5]
              food_arr = []
              water_arr = []
              herb_arr = []
              fish_arr = []
              obstacle_arr = []
              for i in range(9):
                for j in range(9):
                  if abs(i-4) + abs(j-4) <= 4:
                    food_arr.append(near_tile_map[i, j] == material.Forest.index)
                    water_arr.append(near_tile_map[i, j] == material.Water.index)
                    herb_arr.append(near_tile_map[i, j] == material.Herb.index)
                    fish_arr.append(near_tile_map[i, j] == material.Fish.index)
                    obstacle_arr.append(near_tile_map[i, j] in OBSTACLE_TILES)
              food_arr[-1] = max(0, self.poison_map[row, col]) / 20.  # patch after getting trained
              water_arr[-1] = max(0, self.poison_map[row+1, col]) / 20.  # patch after getting trained
              herb_arr[-1] = max(0, self.poison_map[row, col+1]) / 20.  # patch after getting trained
              fish_arr[-1] = max(0, self.poison_map[row-1, col]) / 20.  # patch after getting trained
              obstacle_arr[-1] = max(0, self.poison_map[row, col-1]) / 20.  # patch after getting trained
          else:
              food_arr = water_arr = herb_arr = fish_arr = obstacle_arr = np.zeros(41)
          arr = np.concatenate([
              arr, population_arr, index_arr, prof_arr,
              food_arr, water_arr, herb_arr, fish_arr, obstacle_arr,
          ])
      return arr

  def _extract_game_feature(self, obs):
      game_progress = self.curr_step / MAX_STEP
      n_alive = sum([i in obs for i in range(self.team_size)])
      arr = np.array([
          game_progress,
          n_alive / self.team_size,
          *multi_hot_generator(n_feature=16, index=int(game_progress*16)+1),
          *multi_hot_generator(n_feature=self.team_size, index=n_alive),
      ])
      return arr

  def _extract_legal_action(self, obs, npc_target, enemy_target):
      # --- move ---
      team_pos = np.zeros((self.team_size, 2), dtype=int)
      team_food = np.ones(self.team_size) * 100
      team_stuck = [False] * self.team_size

      # first filter out obstacles
      legal_move = np.zeros((self.team_size, N_MOVE + 1))
      for i in range(self.team_size):
          if i not in obs:
              legal_move[i][-1] = 1
              continue
          tiles_type = obs[i]['Tile'][:, 1].reshape((WINDOW, WINDOW))
          entity_pos = obs[i]['Entity'][1:, 7:9].astype(int).tolist()
          center = np.array([WINDOW_CENTER, WINDOW_CENTER])
          for j, e in enumerate(nmmo_act.Direction.edges):
            next_pos = center + e.delta
            if tiles_type[next_pos[0]][next_pos[1]] in PASSABLE_TILES:
              if next_pos.tolist() not in entity_pos:
                legal_move[i][j] = 1
              else:
                ent_on_next_pos_can_move = False
                for ee in nmmo_act.Direction.edges:  # a rough secondary judgement
                  next_next_pos = next_pos + ee.delta
                  if tiles_type[next_next_pos[0]][next_next_pos[1]] in PASSABLE_TILES:
                    if next_next_pos.tolist() not in entity_pos:
                      ent_on_next_pos_can_move = True
                      break
                if ent_on_next_pos_can_move:
                  legal_move[i][j] = 1
          # save something for later use, and detect whether it is stuck
          my_obs = obs[i]['Entity'][0]
          my_pos = my_obs[7:9].astype(int)
          team_pos[i] = my_pos
          team_food[i] = my_obs[EntityAttr["food"]]
          stuck = []
          for e in nmmo_act.Direction.edges:
            d = np.array(e.delta).astype(int)
            near_pos = my_pos + d
            tile_type = self.tile_map[near_pos[0], near_pos[1]]
            entity_type = self.entity_map[near_pos[0], near_pos[1]]
            st = tile_type in OBSTACLE_TILES or entity_type == TEAMMATE_REPR
            stuck.append(st)
          if sum(stuck) == 4:
            team_stuck[i] = True

      # then prevent blocking out from teammates
      for i in range(self.team_size):
        if i not in obs:
          continue
        for j, e in enumerate(nmmo_act.Direction.edges):  # [North, South, East, West]
          d = np.array(e.delta).astype(int)
          near_pos = (team_pos[i] + d).tolist()
          if near_pos in team_pos.tolist():
            teammate_idx = team_pos.tolist().index(near_pos)
            counter_dir = [1, 0, 3, 2][j]
            if team_stuck[i]:
              legal_move[teammate_idx][counter_dir] = 0
            else:
              my_food = team_food[i]
              teammate_food = team_food[teammate_idx]
              if my_food < teammate_food:
                legal_move[teammate_idx][counter_dir] = 0

      for i in range(self.team_size):
        if sum(legal_move[i][:N_MOVE]) == 0:
          legal_move[i][-1] = 1

      # --- attack ---
      target_attackable = np.concatenate([npc_target != 0, enemy_target != 0], axis=-1)  # first npc, then enemy
      no_target = np.sum(target_attackable, axis=-1, keepdims=True) == 0
      legal_target = np.concatenate([target_attackable, no_target], axis=-1)

      # --- use & sell ---
      legal_use = np.zeros((self.team_size, N_USE + 1))
      legal_sell = np.zeros((self.team_size, N_SELL + 1))
      legal_use[:, -1] = 1
      legal_sell[:, -1] = 1
      for i in range(self.team_size):
          if i not in obs:
              continue
          item_obs = obs[i]['Inventory']
          my_obs = obs[i]['Entity'][0]

          if self.force_use_idx[i] is None:
              poultices = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
              if my_obs[EntityAttr["health"]] <= 60 and len(poultices) > 0:
                  legal_use[i][USE_POULTICE] = 1
              rations = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID]
              if (my_obs[EntityAttr["food"]] < 50 or my_obs[EntityAttr["water"]] < 50) and len(rations) > 0:
                  legal_use[i][USE_RATION] = 1

          if n > N_ITEM_LIMIT and self.force_sell_idx[i] is None:
              poultices = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
              rations = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID]
              if len(poultices) > 1:
                  legal_sell[i][SELL_POULTICE] = 1
                  legal_sell[i][-1] = 0
              if len(rations) > 1:
                  legal_sell[i][SELL_RATION] = 1
                  legal_sell[i][-1] = 0

      legal = {
          'move': legal_move,
          'target': legal_target,
          'use': legal_use,
          'sell': legal_sell,
      }
      return legal

  def trans_action(self, actions):
      actions = np.array(actions)
      self.prev_actions = actions.T.copy()

      raw_actions = collections.defaultdict(dict)
      for i in range(self.team_size):
          move, target, use, sell = actions[:, i]
          self._trans_move(i, raw_actions, move)
          self._trans_attack(i, raw_actions, target)
          self._trans_use(i, raw_actions, use)
          self._trans_sell(i, raw_actions, sell)
          self._trans_buy(i, raw_actions)

      self.curr_step += 1
      self.prev_obs = self.curr_obs
      return raw_actions

  @staticmethod
  def _trans_move(i, raw_actions, move):
    if move != N_MOVE:  # is not idle
      raw_actions[i][nmmo_act.Move] = {nmmo_act.Direction: move}

  def _trans_attack(self, i, raw_actions, target):
      if target != N_ATK_TARGET:  # exist some target to attack
          if target < N_NPC_CONSIDERED:
              self.target_entity_id[i] = int(self.npc_tgt[i][target])
              self.target_entity_pop[i] = 1
          else:
              self.target_entity_id[i] = int(self.enemy_tgt[i][target - N_NPC_CONSIDERED])
          # change the id from entity_id to index in obs
          entity_obs = self.curr_obs[i]['Entity']
          target_row_id = np.argwhere(
              entity_obs[:, EntityAttr["id"]] == self.target_entity_id[i]).item()
          self.target_entity_pop[i] = entity_obs[target_row_id, EntityAttr["population_id"]]
          atk_type = PROF_TO_ATK_TYPE[self.prof[i]]
          raw_actions[i][nmmo_act.Attack] = {
              nmmo_act.Style: atk_type,
              nmmo_act.Target: target_row_id,
          }
      else:
          self.target_entity_id[i] = None
          self.target_entity_pop[i] = None

  def _trans_use(self, i, raw_actions, use):
      if i not in self.curr_obs:  # dead
          return

      if self.force_use_idx[i] is not None:
          raw_actions[i][nmmo_act.Use] = {nmmo_act.Item: self.force_use_idx[i]}
      elif use != N_USE:
          item_obs = self.curr_obs[i]['Inventory']
          if use == USE_POULTICE:
              poultices = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
              min_lvl = min(poultices[:, ItemAttr["level"]])
              poultices = poultices[poultices[:, ItemAttr["level"]] == min_lvl]  # those with lowest level
              min_id = min(poultices[:, ItemAttr["id"]])
              idx = np.argwhere(item_obs[:, ItemAttr["id"]] == min_id).item()
          else:  # USE_RATION
              rations = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID]
              min_lvl = min(rations[:, ItemAttr["level"]])
              rations = rations[rations[:, ItemAttr["level"]] == min_lvl]  # those with lowest level
              min_id = min(rations[:, ItemAttr["id"]])
              idx = np.argwhere(item_obs[:, ItemAttr["id"]] == min_id).item()

          raw_actions[i][nmmo_act.Use] = {nmmo_act.Item: idx}

  def _trans_sell(self, i, raw_actions, sell):
      if i not in self.curr_obs:  # dead
          return

      if self.force_sell_idx[i] is not None:
          raw_actions[i][nmmo_act.Sell] = {
              nmmo_act.Item: self.force_sell_idx[i],
              nmmo_act.Price: int(self.force_sell_price[i]),
          }
      elif sell != N_SELL:
          item_obs = self.curr_obs[i]['Inventory']
          if sell == SELL_POULTICE:
              poultices = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
              min_lvl = min(poultices[:, ItemAttr["level"]])
              poultices = poultices[poultices[:, ItemAttr["level"]] == min_lvl]  # those with lowest level
              min_id = min(poultices[:, ItemAttr["id"]])
              idx = np.argwhere(item_obs[:, ItemAttr["id"]] == min_id).item()
          else:  # SELL_RATION
              rations = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID]
              min_lvl = min(rations[:, ItemAttr["level"]])
              rations = rations[rations[:, ItemAttr["level"]] == min_lvl]  # those with lowest level
              min_id = min(rations[:, ItemAttr["id"]])
              idx = np.argwhere(item_obs[:, ItemAttr["id"]] == min_id).item()

          raw_actions[i][nmmo_act.Sell] = {
              nmmo_act.Item: idx,
              nmmo_act.Price: int(2 + self.curr_step // 300),
          }

  def _trans_buy(self, i, raw_actions):
      if i not in self.curr_obs:  # dead
          return

      if self.force_buy_idx[i] is not None:
          raw_actions[i][nmmo_act.Buy] = {
              nmmo_act.Item: self.force_buy_idx[i],
          }

  def _choose_prof(self):
    seed = np.random.randint(N_ATK_TYPE)
    profs = [ATK_TYPE[(seed + i) % N_ATK_TYPE]
              for i in range(self.team_size)]
    np.random.shuffle(profs)
    return profs

def get_init_tile_map():
    arr = np.zeros((TERRAIN_SIZE+1, TERRAIN_SIZE+1))
    # mark the most outside circle of grass
    arr[MAP_LEFT:MAP_RIGHT+1, MAP_LEFT:MAP_RIGHT+1] = 2
    # mark the unseen tiles
    arr[MAP_LEFT+1:MAP_RIGHT, MAP_LEFT+1:MAP_RIGHT] = N_TILE_TYPE
    return arr

def get_init_poison_map():
  arr = np.ones((TERRAIN_SIZE + 1, TERRAIN_SIZE + 1))
  for i in range(TERRAIN_SIZE // 2):
    l, r = i + 1, TERRAIN_SIZE - i
    arr[l:r, l:r] = -i
  # positive value represents the poison strength
  # negative value represents the shortest distance to poison area
  return arr

def mark_point(arr_2d, index_arr, value, clip=False):
    arr_2d[index_arr[:, 0], index_arr[:, 1]] = \
        np.clip(value, 0., 1.) if clip else value
