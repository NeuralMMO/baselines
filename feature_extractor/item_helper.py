from typing import Dict, Any

import numpy as np

import nmmo
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.lib import material
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState

from model.realikun.model import ModelArchitecture

from feature_extractor.entity_helper import EntityHelper

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

ITEM_TO_PROF_LEVEL = {
  Item.Hat.ITEM_TYPE_ID: "level",
  Item.Top.ITEM_TYPE_ID: "level",
  Item.Bottom.ITEM_TYPE_ID: "level",
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
  Item.Ration.ITEM_TYPE_ID: "level",
  Item.Poultice.ITEM_TYPE_ID: "level",
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

MAX_RESERVE_LEVEL = 6

# CHECK ME: revisit the below constants
def calc_weapon_price(level):
  return min(99, np.random.randint(10) + int(level) * 25 - 10)

def calc_armor_price(level):
  return 3 + np.random.randint(3) + int(level - 1) * 4

def calc_tool_price(level):
  return int(level) * 6 + np.random.randint(3)

def calc_ammo_price(level):
  return max(1, int(level) - 1)

ITEM_TO_PRICE_FN = {
  Item.Hat.ITEM_TYPE_ID: calc_armor_price,
  Item.Top.ITEM_TYPE_ID: calc_armor_price,
  Item.Bottom.ITEM_TYPE_ID: calc_armor_price,
  Item.Sword.ITEM_TYPE_ID: calc_weapon_price,
  Item.Bow.ITEM_TYPE_ID: calc_weapon_price,
  Item.Wand.ITEM_TYPE_ID: calc_weapon_price,
  Item.Scrap.ITEM_TYPE_ID: calc_ammo_price,
  Item.Shaving.ITEM_TYPE_ID: calc_ammo_price,
  Item.Shard.ITEM_TYPE_ID: calc_ammo_price,
  Item.Rod.ITEM_TYPE_ID: calc_tool_price,
  Item.Gloves.ITEM_TYPE_ID: calc_tool_price,
  Item.Pickaxe.ITEM_TYPE_ID: calc_tool_price,
  Item.Chisel.ITEM_TYPE_ID: calc_tool_price,
  Item.Arcane.ITEM_TYPE_ID: calc_tool_price,
  Item.Ration.ITEM_TYPE_ID: calc_armor_price, # arbitrary
  Item.Poultice.ITEM_TYPE_ID: calc_weapon_price, # arbitrary
}

# legal use/sell consumables-related
# CHECK ME: how this goes along with _force_use/sell?
N_USE = 2
N_SELL = 2
LEGAL_POULTICE = 0
LEGAL_RATION = 1

# if surpass the limit, automatically sell one
N_ITEM_LIMIT = 11 # inventory capacity (12) - 1


class ItemHelper:
  def __init__(self, config: nmmo.config.Config,
               entity_helper: EntityHelper) -> None:
    self._config = config
    self._entity_helper = entity_helper
    self._team_size = self._entity_helper.team_size

    self._obs_inv = None

    self.best_hats = None
    self.best_tops = None
    self.best_bottoms = None
    self.best_weapons = None
    self.best_tools = None

    self.best_items = None

    self.force_use_idx = None
    self.force_sell_idx = None
    self.force_sell_price = None
    self.force_buy_idx = None

  def reset(self):
    pass

  def _reset_obs_best_force(self):
    self._obs_inv: Dict = {}

    self.force_use_idx = [None] * self._team_size
    self.force_sell_idx = [None] * self._team_size
    self.force_buy_idx = [None] * self._team_size

    self.best_hats = [None] * self._team_size
    self.best_tops = [None] * self._team_size
    self.best_bottoms = [None] * self._team_size
    self.best_weapons = [None] * self._team_size
    self.best_tools = [None] * self._team_size

    self.best_items = {
      Item.Hat.ITEM_TYPE_ID: self.best_hats,
      Item.Top.ITEM_TYPE_ID: self.best_tops,
      Item.Bottom.ITEM_TYPE_ID: self.best_bottoms,
      Item.Sword.ITEM_TYPE_ID: self.best_weapons,
      Item.Bow.ITEM_TYPE_ID: self.best_weapons,
      Item.Wand.ITEM_TYPE_ID: self.best_weapons,
      Item.Rod.ITEM_TYPE_ID: self.best_tools,
      Item.Gloves.ITEM_TYPE_ID: self.best_tools,
      Item.Pickaxe.ITEM_TYPE_ID: self.best_tools,
      Item.Chisel.ITEM_TYPE_ID: self.best_tools,
      Item.Arcane.ITEM_TYPE_ID: self.best_tools,
    }

  def update(self, obs: Dict[int, Any]):
    if not self._config.ITEM_SYSTEM_ENABLED:
      return

    self._reset_obs_best_force()
    self._evaluate_best_item(obs)

    # CHECK ME: to heuristically generate use actions
    #   DO WE NEED THIS?
    self._equip_best_item(obs)

    for agent_id, agent_obs in obs.items():
      # save for later use, e.g., legal_use(), legal_sell()
      self._obs_inv[agent_id] = agent_obs['Inventory']

      member_pos = self._entity_helper.agent_id_to_pos(agent_id)
      # evaluate which items to sell in the order of priority:
      #   ammos -> weapons -> tools -> weapons/armors_profession
      for sell_fn in [self._sell_ammos, self._sell_weapons, self._sell_tools,
                      self._sell_weapons_armors_profession]:
        if self.force_sell_idx[member_pos] is None:
          sell_fn(member_pos, agent_obs['Inventory'])

  def in_inventory(self, member_pos, inv_idx):
    agent_id = self._entity_helper.pos_to_agent_id(member_pos)
    if agent_id not in self._obs_inv or \
       not self._config.ITEM_SYSTEM_ENABLED or \
       inv_idx is None or \
       inv_idx >= self._config.ITEM_INVENTORY_CAPACITY:
      return False

    item_id = self._obs_inv[agent_id][inv_idx,ItemAttr["id"]]
    return item_id > 0

  # pylint: disable=unused-argument
  def legal_inventory(self, obs, action):
    assert self._config.PROVIDE_ACTION_TARGETS,\
      "config.PROVIDE_ACTION_TARGETS must be set True"
    assert action in [nmmo.action.Use, nmmo.action.Sell, nmmo.action.Destroy],\
      f"action {action} not valid"

    targets = np.zeros((self._team_size,
                        ModelArchitecture.INVENTORY_CAPACITY+1), dtype=np.float32)

    if not self._config.ITEM_SYSTEM_ENABLED:
      return targets

    for member_pos in range(self._team_size):
      ent_id = self._entity_helper.pos_to_agent_id(member_pos)
      if ent_id in obs:
        targets[member_pos][:ModelArchitecture.INVENTORY_CAPACITY] = \
          obs[ent_id]["ActionTargets"][action][nmmo.action.InventoryItem]

        # do not sell/destroy my best items
        if action in [nmmo.action.Sell, nmmo.action.Destroy]:
          for team_best in self.best_items.values():
            my_best = team_best[member_pos]
            if my_best is not None:
              inv_idx = self._get_inv_idx(my_best[ItemAttr['id']], obs[ent_id]['Inventory'])
              targets[member_pos][inv_idx] = 0 # mask the item

    return targets

  def get_price(self, member_pos, inv_idx):
    agent_id = self._entity_helper.pos_to_agent_id(member_pos)
    if agent_id not in self._obs_inv or \
       not self._config.EXCHANGE_SYSTEM_ENABLED or \
       inv_idx >= self._config.ITEM_INVENTORY_CAPACITY:
      return np.nan

    item = ItemState.parse_array(self._obs_inv[agent_id][inv_idx])
    return ITEM_TO_PRICE_FN[item.type_id](item.level)

  #########################################
  # equip/sell helper functions
  #########################################
  def _filter_inventory_obs(self, obs_inv, item_type,
                            max_equipable_lvl=np.inf,
                            to_sell=False) -> np.ndarray:
    flt_opt = (obs_inv[:,ItemAttr["id"]] > 0)
    if to_sell: # cannot sell equipped (and already listed) items
      flt_opt = (obs_inv[:,ItemAttr["equipped"]] == 0)
    if max_equipable_lvl < np.inf: # cannot equip too-high-level (and already listed) items
      flt_opt = (obs_inv[:,ItemAttr["level"]] <= max_equipable_lvl)
    flt_inv = (obs_inv[:,ItemAttr["type_id"]] == item_type) & \
              (obs_inv[:,ItemAttr["listed_price"]] == 0)
    return obs_inv[flt_inv & flt_opt]

  def _get_inv_idx(self, item_id, obs_inv) -> int:
    return np.argwhere(obs_inv[:,ItemAttr["id"]] == item_id).item()

  def _evaluate_best_item(self, obs):
    for agent_id, agent_obs in obs.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)
      agent = self._entity_helper.agent_or_none(agent_id)

      eval_types = [*ARMORS, *TOOLS,
        ATK_TO_WEAPON[self._entity_helper.member_professions[member_pos]]]

      for item_type in eval_types:
        agent_level = getattr(agent, ITEM_TO_PROF_LEVEL[item_type])
        items = self._filter_inventory_obs(agent_obs['Inventory'], item_type,
                                           max_equipable_lvl=agent_level)
        if len(items) > 0:
          max_level = max(items[:,ItemAttr['level']])
          curr_best = self.best_items[item_type][member_pos]
          if curr_best is None or \
             curr_best[ItemAttr['level']] < max_level:
            # update the best item
            sorted_items = sorted(items, key=lambda x: x[ItemAttr["level"]])
            self.best_items[item_type][member_pos] = sorted_items[-1]

  def _equip_best_item(self, obs):
    for agent_id, agent_obs in obs.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)

      # priority: weapon -> hat -> top -> bottom -> then tools
      # NOTE: if there are level-3 Hat and level-4 Top, this results in
      #   using level-3 Hat due to the below priority
      eval_types = [  # reflect priority
        ATK_TO_WEAPON[self._entity_helper.member_professions[member_pos]],
        Item.Hat.ITEM_TYPE_ID,
        Item.Top.ITEM_TYPE_ID,
        Item.Bottom.ITEM_TYPE_ID]

      for item_type in eval_types:
        my_best = self.best_items[item_type][member_pos]
        if self.force_use_idx[member_pos] is None \
          and my_best is not None and (my_best[ItemAttr["equipped"]] == 0):
          inv_idx = self._get_inv_idx(my_best[ItemAttr['id']], agent_obs['Inventory'])
          self.force_use_idx[member_pos] = inv_idx

      # if force_use_idx is still None, then try to equip tool
      if self.force_use_idx[member_pos] is None:
        best_tool = self.best_tools[member_pos]
        if best_tool is not None and best_tool[ItemAttr['equipped']] == 0:
          inv_idx = self._get_inv_idx(best_tool[ItemAttr['id']], agent_obs['Inventory'])
          self.force_use_idx[member_pos] = inv_idx

  #########################################
  # sell helper functions
  #########################################
  def _mark_sell_idx(self, member_pos, item, obs_inv):
    # what to sell is already determined
    # CHECK ME: it should be the lowest level-item
    item_id = item[ItemAttr["id"]]
    inv_idx = np.argwhere(obs_inv[:,ItemAttr["id"]] == item_id).item()
    self.force_sell_idx[member_pos] = inv_idx

  def _concat_types(self, obs_inv, types):
    # cannot use/sell the listed items, cannot sell the equipped items
    flt_idx = np.in1d(obs_inv[:,ItemAttr["type_id"]], list(types)) & \
              (obs_inv[:,ItemAttr["listed_price"]] == 0) & \
              (obs_inv[:,ItemAttr["equipped"]] == 0)
    # sort according to the level, from lowest to highest
    items = obs_inv[flt_idx]
    if len(items):
      sorted_items = sorted(items, key=lambda x: x[ItemAttr["level"]])
      return np.stack(sorted_items)
    return []

  def _sell_type(self, member_pos, obs_inv, sell_type):
    sorted_items = self._concat_types(obs_inv, sell_type)
    if len(sorted_items):
      # items are sorted by level (ascending), so selling the lowest-level items first
      self._mark_sell_idx(member_pos, sorted_items[0], obs_inv)

  def _sell_ammos(self, member_pos, obs_inv) -> int:
    # NOTE: realikun doesn't think ammos seriously
    self._sell_type(member_pos, obs_inv, AMMOS)

  def _sell_weapons(self, member_pos, obs_inv) -> int:
    agent_weapon = ATK_TO_WEAPON[self._entity_helper.member_professions[member_pos]]
    not_my_weapon = [w for w in WEAPONS if w != agent_weapon]
    self._sell_type(member_pos, obs_inv, not_my_weapon)

  def _sell_tools(self, member_pos, obs_inv) -> int:
    sorted_items = self._concat_types(obs_inv, TOOLS)
    # filter out the best tool, if the agent has no weapons,
    #   which means if the agent already has a weapon, then it can sell tools
    if self.best_weapons[member_pos] is None \
       and self.best_tools[member_pos] is not None:
      best_tool_id = self.best_tools[member_pos][ItemAttr["id"]]
      sorted_items = sorted_items[sorted_items[:,ItemAttr["id"]] != best_tool_id]
    if len(sorted_items):
      self._mark_sell_idx(member_pos, sorted_items[0], obs_inv)

  def _sell_weapons_armors_profession(self, member_pos, obs_inv) -> None:
    # sell armors and weapons of my profession
    arms_types = [ # in the order of priority
      Item.Hat.ITEM_TYPE_ID,
      Item.Top.ITEM_TYPE_ID,
      Item.Bottom.ITEM_TYPE_ID,
      ATK_TO_WEAPON[self._entity_helper.member_professions[member_pos]]]

    # process the items in the order of priority
    for item_type in arms_types:
      items = self._filter_inventory_obs(obs_inv, item_type, to_sell=True)

      # filter out the best items
      best_item = self.best_items[item_type][member_pos]
      if best_item is not None:
        items = items[items[:,ItemAttr["id"]] != best_item[ItemAttr["id"]]]

      # also reserve items (that are not equippable now but) for future use
      #   so the level doesn't have to be high
      reserves = items[items[:,ItemAttr["level"]] <= MAX_RESERVE_LEVEL]
      if len(reserves):
        # the best one to reserve
        reserve = sorted(reserves, key=lambda x: x[ItemAttr["level"]])[-1]
        # filter out the reserved
        items = items[items[:,ItemAttr["id"]] != reserve[ItemAttr["id"]]]

      if len(items):
        # sell worst first
        sorted_items = sorted(items, key=lambda x: x[ItemAttr["level"]])
        sell_item = sorted_items[0]
        self._mark_sell_idx(member_pos, sell_item, obs_inv)
        break # stop here

  def legal_use_consumables(self):
    # NOTE: this function only considers ration and poultice,
    #   so it's definitely different from the ActionTargets
    # CHECK ME: how the network actually combines this and _force_use_idx???
    _legal_use = np.zeros((self._team_size, N_USE + 1), dtype=np.float32)
    _legal_use[:, -1] = 1

    for agent_id, obs_inv in self._obs_inv.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)
      agent = self._entity_helper.agent_or_none(agent_id)

      if self.force_use_idx[member_pos] is None:
        flt_poultice = (obs_inv[:,ItemAttr["level"]] <= agent.level) & \
                       (obs_inv[:,ItemAttr["type_id"]] == Item.Poultice.ITEM_TYPE_ID)
        if agent.health <= 60 and len(obs_inv[flt_poultice]) > 0:
          _legal_use[member_pos][LEGAL_POULTICE] = 1
          # CHECK ME: added the below line, is it right?
          _legal_use[member_pos][-1] = 0

        flt_ration = (obs_inv[:,ItemAttr["level"]] <= agent.level) & \
                     (obs_inv[:,ItemAttr["type_id"]] == Item.Ration.ITEM_TYPE_ID)
        if (agent.food < 50 or agent.water < 50) and len(obs_inv[flt_ration]) > 0:
          _legal_use[member_pos][LEGAL_RATION] = 1
          # CHECK ME: added the below line, is it right?
          _legal_use[member_pos][-1] = 0

    return _legal_use

  def legal_sell_consumables(self):
    # NOTE: this function only considers ration and poultice,
    #   so it's definitely different from the ActionTargets
    # CHECK ME: how the network actually combines this and _force_sell_idx???
    #   a similar logic can be used to force destroy/give
    _legal_sell = np.zeros((self._team_size, N_SELL + 1), dtype=np.float32)
    _legal_sell[:, -1] = 1
    n_keep_consumable = 1

    for agent_id, obs_inv in self._obs_inv.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)
      item_type = obs_inv[:,ItemAttr["type_id"]]

      # full inventory, so should get an item out
      if sum(item_type > 0) > N_ITEM_LIMIT and \
         self.force_sell_idx[member_pos] is None:
        poultices = obs_inv[item_type == Item.Poultice.ITEM_TYPE_ID]
        if len(poultices) > n_keep_consumable:
          _legal_sell[member_pos][LEGAL_POULTICE] = 1
          _legal_sell[member_pos][-1] = 0

        rations = obs_inv[item_type == Item.Ration.ITEM_TYPE_ID]
        if len(rations) > n_keep_consumable:
          _legal_sell[member_pos][LEGAL_RATION] = 1
          _legal_sell[member_pos][-1] = 0

    return _legal_sell

  def extract_item_feature(self):
    if not self._config.ITEM_SYSTEM_ENABLED:
      return (
        np.zeros((self._team_size, 1)),
        np.zeros((self._team_size, 1, ModelArchitecture.ITEM_NUM_FEATURES)),
      )

    inv_capacity = self._config.ITEM_INVENTORY_CAPACITY
    dummy_item_types = np.zeros(inv_capacity)
    dummy_item_arrs = np.zeros((inv_capacity, ModelArchitecture.ITEM_NUM_FEATURES))

    team_itm_types =[]
    team_itm_arrs = []
    for member_pos in range(self._entity_helper.team_size):
      agent_id = self._entity_helper.pos_to_agent_id(member_pos)
      # replace with dummy feature if dead
      if not self._entity_helper.is_pos_alive(member_pos) or \
         (agent_id not in self._obs_inv):
        team_itm_types.append(dummy_item_types)
        team_itm_arrs.append(dummy_item_arrs)
        continue

      agent_itm_types = []
      agent_itm_arrs = []
      for j in range(inv_capacity):
        o = self._obs_inv[agent_id][j]
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
