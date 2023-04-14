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
N_PROF = 8
PER_ITEM_FEATURE = 11 # xcxc

class Item:
  def __init__(self, config: nmmo.config.Config, TEAM_SIZE: int) -> None:
    self.config = config

    self.TEAM_SIZE = TEAM_SIZE

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

  def reset(self):
    self.rescue_cooldown = 0
    self.DUMMY_ITEM_TYPES = np.zeros(self.config.ITEM_INVENTORY_CAPACITY)
    self.DUMMY_ITEM_ARRS = np.zeros((self.config.ITEM_INVENTORY_CAPACITY, PER_ITEM_FEATURE))


  def extract_item_feature(self, obs):
    self.rescue_cooldown = max(0, self.rescue_cooldown - 1)

    items_arrs = []
    items_types = []
    for i in range(self.TEAM_SIZE):
      # replace with dummy feature if dead
      if i not in obs:
        items_types.append(self.DUMMY_ITEM_TYPES)
        items_arrs.append(self.DUMMY_ITEM_ARRS)
        continue

      item_obs = obs[i]['Inventory']
      item_arrs = []
      item_types = []
      for j in range(self.config.ITEM_INVENTORY_CAPACITY):
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
