import collections
import copy

import nmmo
import nmmo.io.action as nmmo_act
import numpy as np
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.io import action
from nmmo.lib import material
from nmmo.systems import item
from nmmo.systems.item import ItemState

from model.util import multi_hot_generator, one_hot_generator
from team_helper import TeamHelper

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

class Inventory:
  def __init__(self) -> None:
    self._legal_use = None
    self._legal_sell = None

  def reset(self):
    pass

  def update(self, obs):
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
      my_entity = self._my_entity(obs, i)

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
          max_equipable_lvl = max(my_entity[-N_PROF:])  # maximum of all levels
        else:
          max_equipable_lvl = my_entity[ITEM_TO_PROF_IDX[item_type]]
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
          max_equipable_lvl = my_entity[ITEM_TO_PROF_IDX[tool_type]]
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

    legal_use = np.zeros((self.team_size, N_USE + 1))
    legal_sell = np.zeros((self.team_size, N_SELL + 1))
    legal_use[:, -1] = 1
    legal_sell[:, -1] = 1
    for i in range(self.team_size):
      if i not in obs:
        continue

      item_obs = obs[i]['Inventory'][obs[i]['Inventory'][:, ItemAttr["type_id"]] > 0]
      my_obs = self._my_entity(obs, i)

      if self.force_use_idx[i] is None:
        poultices = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
        if my_obs[EntityAttr["health"]] <= 60 and len(poultices) > 0:
            legal_use[i][USE_POULTICE] = 1
        rations = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID]
        if (my_obs[EntityAttr["food"]] < 50 or my_obs[EntityAttr["water"]] < 50) and len(rations) > 0:
            legal_use[i][USE_RATION] = 1

      # xcxc only include real items
      n = len(item_obs)
      if n > self.config.ITEM_INVENTORY_CAPACITY - 1 and self.force_sell_idx[i] is None:
        poultices = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
        rations = item_obs[item_obs[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID]
        if len(poultices) > 1:
          legal_sell[i][SELL_POULTICE] = 1
          legal_sell[i][-1] = 0
        if len(rations) > 1:
          legal_sell[i][SELL_RATION] = 1
          legal_sell[i][-1] = 0

  def legal_use(self):
    # 'use_t': {i: obs[i]["ActionTargets"][action.Use][action.InventoryItem] for i in range(self.team_size)},
    return self._legal_use

  def legal_sell(self):
    # 'sell_t': {i: obs[i]["ActionTargets"][action.Sell][action.InventoryItem] for i in range(self.team_size)},
    return self._legal_sell

