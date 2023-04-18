from typing import Dict, Any

import numpy as np

import nmmo
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.lib import material
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState

from model.model import ModelArchitecture

from feature_extractor.entity_helper import EntityHelper

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

ITEM_TO_PROF_LEVEL = {
  Item.Sword.ITEM_TYPE_ID: "melee_level",
  Item.Bow.ITEM_TYPE_ID: "range_level",
  Item.Wand.ITEM_TYPE_ID: "mage_level",
  Item.Scrap.ITEM_TYPE_ID: "melee_level",
  Item.Shaving.ITEM_TYPE_ID: "range_level",
  Item.Shard.ITEM_TYPE_ID: "mage_level",
  Item.Rod.ITEM_TYPE_ID: "fishing_level",
  Item.Gloves.ITEM_TYPE_ID: "herbalism_level",
  Item.Pickaxe.ITEM_TYPE_ID: "prospecting_level",
  Item.Chisel.ITEM_TYPE_ID: "carving_level",
  Item.Arcane.ITEM_TYPE_ID: "alchemy_level",
}

ARMORS = {
  Item.Hat.ITEM_TYPE_ID,
  Item.Top.ITEM_TYPE_ID,
  Item.Bottom.ITEM_TYPE_ID,
}

WEAPONS = {
  Item.Sword.ITEM_TYPE_ID,
  Item.Wand.ITEM_TYPE_ID,
  Item.Bow.ITEM_TYPE_ID,
}

TOOLS = {
  Item.Rod.ITEM_TYPE_ID,
  Item.Gloves.ITEM_TYPE_ID,
  Item.Pickaxe.ITEM_TYPE_ID,
  Item.Chisel.ITEM_TYPE_ID,
  Item.Arcane.ITEM_TYPE_ID,
}

AMMOS = {
  Item.Scrap.ITEM_TYPE_ID,
  Item.Shaving.ITEM_TYPE_ID,
  Item.Shard.ITEM_TYPE_ID,
}

CONSUMABLES = {
  Item.Ration.ITEM_TYPE_ID,
  Item.Poultice.ITEM_TYPE_ID,
}

ATK_TO_WEAPON = {
  'Melee': Item.Sword.ITEM_TYPE_ID,
  'Range': Item.Bow.ITEM_TYPE_ID,
  'Mage': Item.Wand.ITEM_TYPE_ID
}

ATK_TO_TOOL = {
  'Melee': Item.Pickaxe.ITEM_TYPE_ID,
  'Range': Item.Chisel.ITEM_TYPE_ID,
  'Mage': Item.Arcane.ITEM_TYPE_ID
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
N_PROF = 8

MAX_SELL_LEVEL = 6

# CHECK ME: revisit the below constants
def calc_weapon_price(level):
  return min(99, np.random.randint(10) + int(level) * 25 - 10)

def calc_armor_price(level):
  return 3 + np.random.randint(3) + int(level - 1) * 4

def calc_tool_price(level):
  return int(level) * 6 + np.random.randint(3)

def calc_ammo_price(level):
  return max(1, int(level) - 1)

# legal use/sell consumables-related
# CHECK ME: how this goes along with _force_use/sell?
N_USE = 2
N_SELL = 2
USE_POULTICE = 0
USE_RATION = 1
SELL_POULTICE = 0
SELL_RATION = 1

# if surpass the limit, automatically sell one
N_ITEM_LIMIT = 11 # inventory capacity (12) - 1


class Inventory:
  def __init__(self, config: nmmo.config.Config,
               entity_helper: EntityHelper) -> None:
    self._config = config
    self._entity_helper = entity_helper
    self._team_size = self._entity_helper.team_size

    self._best_hats = None
    self._best_tops = None
    self._best_bottoms = None
    self._best_weapons = None
    self._best_tools = None
    self._force_use_idx = None
    self._force_sell_idx = None
    self._force_sell_price = None
    self._force_buy_idx = None

  def reset(self):
    pass

  def _reset_best_force(self):
    self._best_hats = [None] * self._team_size
    self._best_tops = [None] * self._team_size
    self._best_bottoms = [None] * self._team_size
    self._best_weapons = [None] * self._team_size
    self._best_tools = [None] * self._team_size
    self._force_use_idx = [None] * self._team_size
    self._force_sell_idx = [None] * self._team_size
    self._force_sell_price = [None] * self._team_size
    self._force_buy_idx = [None] * self._team_size

  def update(self, obs: Dict[int, Any]):
    self._reset_best_force()

    for agent_id, agent_obs in obs.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)
      agent = self._entity_helper.agent(agent_id)
      obs_inv = agent_obs['Inventory']

      # evaluate which items to equip (=use) in the order of priority
      self._equip_weapons_armors(member_pos, agent, obs_inv)
      self._equip_tools(member_pos, agent, obs_inv)

      # evaluate which items to sell in the order of priority:
      #   ammos -> weapons -> tools -> weapons/armors_profession
      for sell_fn in [self._sell_ammos, self._sell_weapons, self._sell_tools,
                      self._sell_weapons_armors_profession]:
        n_items = sell_fn(member_pos, obs_inv)
        if n_items > 0: # there is an item to sell
          break

  def _equip_weapons_armors(self, member_pos, agent, obs_inv) -> None:
    # NOTE: if there are level-3 Hat and level-4 Top, this results in using level-3 Hat
    #   due to the below priority
    usable_types = [  # reflect priority
      ATK_TO_WEAPON[self._entity_helper.member_professions[member_pos]],
      Item.Hat.ITEM_TYPE_ID,
      Item.Top.ITEM_TYPE_ID,
      Item.Bottom.ITEM_TYPE_ID]

    best_savers = [
      self._best_weapons,
      self._best_hats,
      self._best_tops,
      self._best_bottoms]

    for item_type, saver in zip(usable_types, best_savers):
      if item_type in ARMORS:
        max_equipable_lvl = agent.level
      else:
        max_equipable_lvl = getattr(agent, ITEM_TO_PROF_LEVEL[item_type])

      flt_item = (obs_inv[:, ItemAttr["type_id"]] == item_type) & \
                 (obs_inv[:, ItemAttr["level"]] <= max_equipable_lvl)
      items = obs_inv[flt_item]

      if len(items) > 0:
        inv_idx = self._get_idx_max_type(items, obs_inv)
        # update the best item
        saver[member_pos] = obs_inv[inv_idx]
        if not obs_inv[inv_idx][ItemAttr["equipped"]] \
           and self._force_use_idx[member_pos] is None:
          self._force_use_idx[member_pos] = inv_idx
          break # stop here to force use the item

  def _equip_tools(self, member_pos, agent, obs_inv) -> None:
    tools = []
    for tool_type in TOOLS:
      tools_ = obs_inv[obs_inv[:, ItemAttr["type_id"]] == tool_type]
      max_equipable_lvl = getattr(agent, ITEM_TO_PROF_LEVEL[tool_type])
      tools_ = tools_[tools_[:, ItemAttr["level"]] <= max_equipable_lvl]
      tools.append(tools_)
    tools = np.concatenate(tools)

    if len(tools) > 0:
      inv_idx = self._get_idx_max_type(tools, obs_inv)
      self._best_tools[member_pos] = obs_inv[inv_idx]
      if not obs_inv[inv_idx][ItemAttr["equipped"]] \
          and self._force_use_idx[member_pos] is None:
        self._force_use_idx[member_pos] = inv_idx  # save for later translation

  def _sell_weapons_armors_profession(self, member_pos, obs_inv) -> None:
    # sell armors and weapons of my profession
    sell_not_best_types = [ # in the order of priority
      Item.Hat.ITEM_TYPE_ID,
      Item.Top.ITEM_TYPE_ID,
      Item.Bottom.ITEM_TYPE_ID,
      ATK_TO_WEAPON[self._entity_helper.member_professions[member_pos]]]

    best_savers = [
      self._best_hats,
      self._best_tops,
      self._best_bottoms,
      self._best_weapons]

    items = self._concat_types(obs_inv, sell_not_best_types)
    # filter out the best & reserves
    for item_type, saver in zip(sell_not_best_types, best_savers):
      if saver[member_pos] is not None:
        items = items[items[:, ItemAttr["id"]] != saver[member_pos][ItemAttr["id"]]]
        # reserve items no more than level 6 for future use
        reserves = items[items[:, ItemAttr["level"]] <= MAX_SELL_LEVEL]
        if len(reserves) > 0:
          # the best one to reserve
          reserve = sorted(reserves, key=lambda x: x[ItemAttr["level"]])[-1]
          # filter out the reserved
          items = items[items[:, ItemAttr["id"]] != reserve[ItemAttr["id"]]]

    n_items = len(items)
    if n_items > 0:
      # sell worst first
      sorted_items = sorted(items, key=lambda x: x[ItemAttr["level"]])
      self._mark_sell_idx(member_pos, sorted_items[0], obs_inv,
                          calc_weapon_price if item_type in WEAPONS
                          else calc_armor_price)
    return n_items

  def _mark_sell_idx(self, member_pos, item, obs_inv, calc_price):
    # what to sell is already determined
    # CHECK ME: it should be the lowest level-item
    item_id = item[ItemAttr["id"]]
    inv_idx = np.argwhere(obs_inv[:, ItemAttr["id"]] == item_id).item()
    self._force_sell_idx[member_pos] = inv_idx
    self._force_sell_price[member_pos] = calc_price(item[ItemAttr["level"]])

  def _sell_type(self, member_pos, obs_inv, sell_type, calc_price) -> int:
    sorted_items = self._concat_types(obs_inv, sell_type)
    n_items = len(sorted_items)
    if n_items > 0:
      self._mark_sell_idx(member_pos, sorted_items[0], obs_inv, calc_price)
    return n_items

  def _sell_weapons(self, member_pos, obs_inv) -> int:
    agent_weapon = ATK_TO_WEAPON[self._entity_helper.member_professions[member_pos]]
    not_my_weapon = [w for w in WEAPONS if w != agent_weapon]
    n_items = self._sell_type(member_pos, obs_inv, not_my_weapon, calc_weapon_price)
    return n_items

  def _sell_tools(self, member_pos, obs_inv) -> int:
    sorted_items = self._concat_types(obs_inv, TOOLS)
    # filter out the best tool, if the agent has no weapons,
    #   which means if the agent already has a weapon, then it can sell tools
    if self._best_weapons[member_pos] is None \
       and self._best_tools[member_pos] is not None:
      best_tool_id = self._best_tools[member_pos][ItemAttr["id"]]
      sorted_items = sorted_items[sorted_items[:, ItemAttr["id"]] != best_tool_id]

    n_items = len(sorted_items)
    if n_items > 0:
      self._mark_sell_idx(member_pos, sorted_items[0], obs_inv, calc_tool_price)
    return n_items

  def _sell_ammos(self, member_pos, obs_inv) -> int:
    # NOTE: realikun doesn't think ammos seriously
    n_items = self._sell_type(member_pos, obs_inv, AMMOS, calc_ammo_price)
    return n_items

  def _get_inv_idx(self, item_id, obs_inv) -> int:
    return np.argwhere(obs_inv[:, ItemAttr["id"]] == item_id).item()

  def _get_idx_max_type(self, items, inventory_obs):
    max_item_lvl = max(items[:, ItemAttr["level"]])
    tools = items[items[:, ItemAttr["level"]] == max_item_lvl]  # those with highest level
    min_id = min(tools[:, ItemAttr["id"]])  # if level is same, choose the item with min id
    return self._get_inv_idx(min_id, inventory_obs)

  def _concat_types(self, obs_inv, types):
    flt_idx = np.in1d(obs_inv[:, ItemAttr["type_id"]], list(types)) & \
              (obs_inv[:, ItemAttr["equipped"]] == 0) # cannot sell the equipped items
    # sort according to the level
    items = obs_inv[flt_idx]
    if len(items):
      sorted_items = sorted(items, key=lambda x: x[ItemAttr["level"]])
      return np.stack(sorted_items)
    
    return []

  def legal_use_consumables(self, obs):
    # NOTE: this function only considers ration and poultice,
    #   so it's definitely different from the ActionTargets
    # CHECK ME: how the network actually combines this and _force_use_idx???
    _legal_use = np.zeros((self._team_size, N_USE + 1), dtype=np.float32)
    _legal_use[:, -1] = 1

    for agent_id, agent_obs in obs.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)
      agent = self._entity_helper.agent(agent_id)
      obs_inv = agent_obs['Inventory']
      item_type = obs_inv[:, ItemAttr["type_id"]]

      if self._force_use_idx[member_pos] is None:
        poultices = obs_inv[item_type == Item.Poultice.ITEM_TYPE_ID]
        if agent.health <= 60 and len(poultices) > 0:
          _legal_use[member_pos][USE_POULTICE] = 1
          # CHECK ME: added the below line, is it right?
          _legal_use[member_pos][-1] = 0

        rations = obs_inv[item_type == Item.Ration.ITEM_TYPE_ID]
        if (agent.food < 50 or agent.water < 50) and len(rations) > 0:
          _legal_use[member_pos][USE_RATION] = 1
          # CHECK ME: added the below line, is it right?
          _legal_use[member_pos][-1] = 0

    return _legal_use

  def legal_sell_consumables(self, obs):
    # NOTE: this function only considers ration and poultice,
    #   so it's definitely different from the ActionTargets
    # CHECK ME: how the network actually combines this and _force_use_idx???
    _legal_sell = np.zeros((self._team_size, N_SELL + 1), dtype=np.float32)
    _legal_sell[:, -1] = 1

    for agent_id, agent_obs in obs.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)
      obs_inv = agent_obs['Inventory']
      item_type = obs_inv[:, ItemAttr["type_id"]]

      if sum(item_type > 0) > N_ITEM_LIMIT and \
         self._force_sell_idx is None:
        n_keep_consumable = 1
        poultices = obs_inv[item_type == Item.Poultice.ITEM_TYPE_ID]
        if len(poultices) > n_keep_consumable:
          _legal_sell[member_pos][USE_POULTICE] = 1
          _legal_sell[member_pos][-1] = 0

        rations = obs_inv[item_type == Item.Ration.ITEM_TYPE_ID]
        if len(rations) > n_keep_consumable:
          _legal_sell[member_pos][USE_RATION] = 1
          _legal_sell[member_pos][-1] = 0

    return _legal_sell

  def extract_item_feature(self, obs):
    inv_capacity = self._config.ITEM_INVENTORY_CAPACITY
    dummy_item_types = np.zeros(inv_capacity)
    dummy_item_arrs = np.zeros((inv_capacity, ModelArchitecture.ITEM_NUM_FEATURES))

    team_itm_types =[]
    team_itm_arrs = []
    for member_pos in range(self._entity_helper.team_size):
      # replace with dummy feature if dead
      if not self._entity_helper.is_pos_alive(member_pos):
        team_itm_types.append(dummy_item_types)
        team_itm_arrs.append(dummy_item_arrs)
        continue

      agent_id = self._entity_helper.pos_to_agent_id(member_pos)

      item_obs = obs[agent_id]['Inventory']
      agent_itm_types = []
      agent_itm_arrs = []
      for j in range(inv_capacity):
        o = item_obs[j]
        agent_itm_types.append(o[ItemAttr["type_id"]])  # type is 0 if item is empty
        agent_itm_arrs.append(self._extract_per_item_feature(o))
      team_itm_types.append(np.array(agent_itm_types))
      team_itm_arrs.append(np.stack(agent_itm_arrs))

    return np.stack(team_itm_types).astype(np.float32), \
           np.stack(team_itm_arrs).astype(np.float32)

  @staticmethod
  def _extract_per_item_feature(o):
    if np.sum(o) > 0:
      # CHECK ME: revisit item feature scalers
      # CHECK ME: resource_restore is left out in the realikun's model
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
      arr = np.zeros(ModelArchitecture.ITEM_NUM_FEATURES)
    return arr
