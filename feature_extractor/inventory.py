from typing import Callable, Dict, Tuple

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
from feature_extractor.item import Item, ATK_TO_WEAPON, ARMORS, N_PROF, ITEM_TO_PROF_IDX, AMMOS, WEAPONS, TOOLS
from feature_extractor.action import N_USE, USE_POULTICE, SELL_POULTICE, SELL_RATION, N_USE, N_SELL, USE_RATION
from feature_extractor.entity_helper import EntityHelper

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

calc_weapon_price = lambda x: min(99, np.random.randint(10) + int(x) * 25 - 10)
calc_armor_price = lambda x: 3 + np.random.randint(3) + int(x - 1) * 4
calc_tool_price = lambda x: int(x) * 6 + np.random.randint(3)
calc_ammo_price = lambda x: max(1, int(x) - 1)

MAX_SELL_LEVEL = 6

class Inventory:
  def __init__(self, config: nmmo.config.Config,
               team_id: int, entity_helper: EntityHelper, team_helper: TeamHelper) -> None:
    
    self._legal_use = None
    self._legal_sell = None

    self._config = config
    self._team_id = team_id
    self._team_agent_ids = team_helper.teams[team_id]
    self.TEAM_SIZE = len(self._team_agent_ids)
    self.entity_helper = entity_helper

    self.item = Item(config, self.TEAM_SIZE)

  def reset(self):
    pass

  def update(self, obs: Dict):
    # reset
    self.item.best_hats = [None] * self.TEAM_SIZE
    self.item.best_tops = [None] * self.TEAM_SIZE
    self.item.best_bottoms = [None] * self.TEAM_SIZE
    self.item.best_weapons = [None] * self.TEAM_SIZE
    self.item.best_tools = [None] * self.TEAM_SIZE
    self.item.force_use_idx = [None] * self.TEAM_SIZE
    self.item.force_sell_idx = [None] * self.TEAM_SIZE
    self.item.force_buy_idx = [None] * self.TEAM_SIZE
    self.item.force_sell_price = [None] * self.TEAM_SIZE

    for member_id, member_obs in obs.items():
      for inventory_obs in member_obs['Inventory']:
        my_entity = self.entity_helper.entity_by_id(member_obs["Entity"][EntityAttr["id"]])


        self._use_weapons_armors(member_id, inventory_obs, my_entity)
        self._use_tools(member_id, inventory_obs, my_entity)

        # directly sell ammo
        n_items = self._sell_ammos(member_id, inventory_obs)
        if n_items > 0:
          continue

        n_items = self._sell_weapons(member_id, inventory_obs)
        if n_items > 0:
          continue

        n_items = self._sell_tools(member_id, inventory_obs)
        if n_items > 0:
          continue

        self._sell_weapons_armors_profession(member_id, inventory_obs)

  def _use_weapons_armors(self, id, inventory_obs, my_entity) -> None:
    # force use weapons and armors
    usable_types = [  # reflect priority
      ATK_TO_WEAPON[self.entity_helper.member_professions[id]],
      item.Hat.ITEM_TYPE_ID,
      item.Top.ITEM_TYPE_ID,
      item.Bottom.ITEM_TYPE_ID,
    ]
    best_savers = [
      self.item.best_weapons,
      self.item.best_hats,
      self.item.best_tops,
      self.item.best_bottoms,
    ]
    for item_type, saver in zip(usable_types, best_savers):
      if item_type in ARMORS:
        max_equipable_lvl = max(my_entity[[EntityAttr["melee_level"],EntityAttr["range_level"],EntityAttr["mage_level"]]])  # maximum of all levels
      else:
        max_equipable_lvl = my_entity[ITEM_TO_PROF_IDX[item_type]]
      items = inventory_obs[inventory_obs[:, ItemAttr["type_id"]] == item_type]  # those with the target type
      items = items[items[:, ItemAttr["level"]] <= max_equipable_lvl]  # those within the equipable level
      if len(items) > 0:
        idx = self._get_idx_max_type(items, inventory_obs)
        saver[id] = inventory_obs[idx]
        if not inventory_obs[idx][ItemAttr["equipped"]] and self.item.force_use_idx[id] is None:
          self.item.force_use_idx[id] = idx  # save for later translation

  def _use_tools(self, id, inventory_obs, my_entity) -> None:
    # force use tools
    if self.item.best_weapons[id] is None:
      tools = []
      for tool_type in TOOLS:
        tools_ = inventory_obs[inventory_obs[:, ItemAttr["type_id"]] == tool_type]
        max_equipable_lvl = my_entity[ITEM_TO_PROF_IDX[tool_type]]
        tools_ = tools_[tools_[:, ItemAttr["level"]] <= max_equipable_lvl]
        tools.append(tools_)
      tools = np.concatenate(tools)
      if len(tools) > 0:
        idx = self._get_idx_max_type(tools, inventory_obs)
        self.item.best_tools[id] = inventory_obs[idx]
        if not inventory_obs[idx][ItemAttr["equipped"]] and self.item.force_use_idx[id] is None:
          self.item.force_use_idx[id] = idx  # save for later translation

  def _sell_weapons_armors_profession(self, id, inventory_obs) -> None:
    # sell armors and weapons of my profession
    sell_not_best_types = [
      item.Hat.ITEM_TYPE_ID,
      item.Top.ITEM_TYPE_ID,
      item.Bottom.ITEM_TYPE_ID,
      ATK_TO_WEAPON[self.entity_helper.member_professions[id]],
    ]  # reflect priority
    best_savers = [
      self.item.best_hats,
      self.item.best_tops,
      self.item.best_bottoms,
      self.item.best_weapons,
    ]
    for item_type, saver in zip(sell_not_best_types, best_savers):
      items = inventory_obs[inventory_obs[:, ItemAttr["type_id"]] == item_type]  # those with the target type
      if saver[id] is not None:
        items = items[items[:, ItemAttr["id"]] != saver[id][ItemAttr["id"]]]  # filter out the best
        best_lvl = saver[id][ItemAttr["level"]]
        if best_lvl < MAX_SELL_LEVEL:
          # reserve items no more than level 6 for future use
          reserves = items[items[:, ItemAttr["level"]] > best_lvl]
          reserves = reserves[reserves[:, ItemAttr["level"]] <= 6]
          if len(reserves) > 0:
            reserve = sorted(reserves, key=lambda x: x[ItemAttr["level"]])[-1]  # the best one to reserve
            items = items[items[:, ItemAttr["id"]] != reserve[ItemAttr["id"]]]  # filter out the reserved
      if len(items) > 0:
        # sell worst first
        to_sell = sorted(items, key=lambda x: x[ItemAttr["level"]])[0]
        self._mark_sell_idx(id, to_sell, inventory_obs, calc_weapon_price if item_type in WEAPONS else calc_armor_price)

  def _mark_sell_idx(self, id, item, inventory_obs, calc_price):
    item_id = item[ItemAttr["id"]]
    idx = np.argwhere(inventory_obs[:, ItemAttr["id"]] == item_id).item()
    lvl = item[ItemAttr["level"]]
    self.item.force_sell_idx[id] = idx
    self.item.force_sell_price[id] = calc_price(lvl)

  def _sell_type(self, id, inventory_obs, sell_type, calc_price) -> int:
    items = self._concat_types(inventory_obs, sell_type)
    n = len(items)
    if n > 0:
      self._mark_sell_idx(id, items[0], inventory_obs, calc_price)
    return n

  def _sell_weapons(self, id, inventory_obs) -> int:
    other_weapon_types = [w for w in WEAPONS
                          if w != ATK_TO_WEAPON[self.entity_helper.member_professions[id]]]
    n_items = self._sell_type(id, inventory_obs, other_weapon_types, calc_weapon_price)
    return n_items

  def _sell_tools(self, id, inventory_obs) -> int:
    # sell tools
    items = self._concat_types(inventory_obs, TOOLS)
    if self.item.best_weapons[id] is None and self.item.best_tools[id] is not None:
      best_idx = self.item.best_tools[id][ItemAttr["id"]]
      items = items[items[:, ItemAttr["id"]] != best_idx]  # filter out the best
    n = len(items)
    if n > 0:
      self._mark_sell_idx(id, items, inventory_obs, calc_tool_price)

    return n

  def _sell_ammos(self, id, inventory_obs) -> int:
    # sell ammos
    n_items = self._sell_type(id, inventory_obs, AMMOS, calc_ammo_price)
    return n_items

  def _get_idx_max_type(self, items, inventory_obs):
    max_item_lvl = max(items[:, ItemAttr["level"]])
    tools = items[items[:, ItemAttr["level"]] == max_item_lvl]  # those with highest level
    min_id = min(tools[:, ItemAttr["id"]])  # always choose the one with the minimal item id as the best
    idx = np.argwhere(inventory_obs[:, ItemAttr["id"]] == min_id).item()
    return idx

  def _concat_types(self, inventory_obs, types):
    return np.concatenate([
        inventory_obs[inventory_obs[:, ItemAttr["type_id"]] == i]
        for i in types
      ], axis=0)

  def legal_use(self, obs):
    # 'use_t': {i: obs[i]["ActionTargets"][action.Use][action.InventoryItem] for i in range(self.TEAM_SIZE)},
    _legal_use = np.zeros((self.TEAM_SIZE, N_USE + 1))
    _legal_use[:, -1] = 1

    for member_id, member_obs in obs.items():

      inventory_obs = member_obs[member_obs[:, ItemAttr["type_id"]] > 0]
      my_obs = self.entity_helper.entity_by_id(member_obs["Entity"][EntityAttr["id"]])

      if self.item.force_use_idx[member_id] is None:
        poultices = inventory_obs[inventory_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
        if my_obs[EntityAttr["health"]] <= 60 and len(poultices) > 0:
            _legal_use[member_id][USE_POULTICE] = 1
        rations = inventory_obs[inventory_obs[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID]
        if (my_obs[EntityAttr["food"]] < 50 or my_obs[EntityAttr["water"]] < 50) and len(rations) > 0:
            _legal_use[member_id][USE_RATION] = 1

    return self._legal_use

  def legal_sell(self, obs):
    # 'sell_t': {i: obs[i]["ActionTargets"][action.Sell][action.InventoryItem] for i in range(self.TEAM_SIZE)},
    _legal_sell = np.zeros((self.TEAM_SIZE, N_SELL + 1))
    _legal_sell[:, -1] = 1
    for member_id, member_obs in obs.items():

      inventory_obs = member_obs[member_obs[:, ItemAttr["type_id"]] > 0]

      # xcxc only include real items
      n = len(inventory_obs)
      if n > self._config.ITEM_INVENTORY_CAPACITY - 1 and self.item.force_sell_idx[member_id] is None:
        poultices = inventory_obs[inventory_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
        rations = inventory_obs[inventory_obs[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID]
        if len(poultices) > 1:
          _legal_sell[member_id][SELL_POULTICE] = 1
          _legal_sell[member_id][-1] = 0
        if len(rations) > 1:
          _legal_sell[member_id][SELL_RATION] = 1
          _legal_sell[member_id][-1] = 0
    return self._legal_sell

