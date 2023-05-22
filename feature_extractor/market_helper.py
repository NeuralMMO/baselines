import numpy as np

import nmmo
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState

from feature_extractor.entity_helper import EntityHelper
from feature_extractor.item_helper import ItemHelper
from feature_extractor.item_helper import ARMORS, ATK_TO_WEAPON, ITEM_TO_PROF_LEVEL

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

LOW_HEALTH = 35
POULTICE_SCORE = 4
RATION_SCORE = 2

WEAPON_SCORE = 10
ARMOR_SCORE = 4


class MarketHelper:
  def __init__(self, config: nmmo.config.AllGameSystems,
               entity_helper: EntityHelper,
               item_helper: ItemHelper) -> None:
    self._config = config
    self._entity_helper = entity_helper
    self._team_size = self._entity_helper.team_size

    self.curr_step = None
    self._item = item_helper

    self._agent_health = None
    self._restore_score = None
    self._combat_score = None

  def reset(self):
    pass

  def update(self, obs, curr_step: int):
    if not self._config.EXCHANGE_SYSTEM_ENABLED:
      return

    self.curr_step = curr_step
    self._item.force_buy_idx = [None] * self._team_size

    # evaluate each agent
    self._calculate_restore_score(obs)
    self._calculate_combat_score()

    # priority: emergency poultice -> weapon/armors -> consumables
    self._emergency_buy_poultice(obs)
    self._buy_weapons_armors(obs)
    self._buy_consumables(obs)

  def _calculate_combat_score(self):
    # to determine which agents to priotize in buying weapons/arms
    self._combat_score = np.zeros(self._team_size)
    for member_pos in range(self._team_size):
      score = 0
      # based on only the agents' best items
      if self._item.best_weapons[member_pos] is not None:
        score += self._item.best_weapons[member_pos][ItemAttr["level"]] * WEAPON_SCORE
      elif self._item.best_tools[member_pos] is not None:
        score += self._item.best_tools[member_pos][ItemAttr["level"]] * ARMOR_SCORE
      if self._item.best_hats[member_pos] is not None:
        score += self._item.best_hats[member_pos][ItemAttr["level"]] * ARMOR_SCORE
      if self._item.best_tops[member_pos] is not None:
        score += self._item.best_tops[member_pos][ItemAttr["level"]] * ARMOR_SCORE
      if self._item.best_bottoms[member_pos] is not None:
        score += self._item.best_bottoms[member_pos][ItemAttr["level"]] * ARMOR_SCORE
      self._combat_score[member_pos] = score

  def _calculate_restore_score(self, obs):
    # to determine which agents to priotize in buying poultice, ration
    self._agent_health = np.zeros(self._team_size)
    self._restore_score = np.zeros(self._team_size)
    for agent_id, agent_obs in obs.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)
      agent = self._entity_helper.agent_or_none(agent_id)
      obs_inv = agent_obs['Inventory']

      self._agent_health[member_pos] = agent.health

      # usable items: within the level & not listed
      flt_level = (obs_inv[:,ItemAttr["level"]] <= agent.level) & \
                  (obs_inv[:,ItemAttr["listed_price"]] == 0)
      flt_poultice = (obs_inv[:,ItemAttr["type_id"]] == Item.Potion.ITEM_TYPE_ID)
      flt_ration = (obs_inv[:,ItemAttr["type_id"]] == Item.Ration.ITEM_TYPE_ID)

      poultice_score = min(2, sum(flt_poultice & flt_level)) * POULTICE_SCORE
      ration_score = min(1, sum(flt_ration & flt_level)) * RATION_SCORE
      self._restore_score[member_pos] = poultice_score + ration_score

  def _filter_market_obs(self, agent, obs_mkt, item_type, price=None) -> np.ndarray:
    # filter item_type with the already marked items, agent level/gold
    if price is None:
      price = agent.gold
    max_equipable_lvl = getattr(agent, ITEM_TO_PROF_LEVEL[item_type])
    flt_mkt = ~np.in1d(obs_mkt[:,ItemAttr["id"]], self._item.force_buy_idx) & \
              (obs_mkt[:,ItemAttr["type_id"]] == item_type) & \
              (obs_mkt[:,ItemAttr["level"]] <= max_equipable_lvl) & \
              (obs_mkt[:,ItemAttr["listed_price"]] <= agent.gold) & \
              (obs_mkt[:,ItemAttr["listed_price"]] <= price)

    return obs_mkt[flt_mkt]

  def _emergency_buy_poultice(self, obs):
    proc_order = np.argsort(self._agent_health)
    for member_pos in proc_order: # start from lowest health
      agent_id = self._entity_helper.pos_to_agent_id(member_pos)
      agent = self._entity_helper.agent_or_none(agent_id)
      if agent_id not in obs: # skip dead
        continue

      if self._agent_health[member_pos] > LOW_HEALTH: # in good health
        continue

      if self._restore_score[member_pos] >= POULTICE_SCORE: # have a poultice
        continue

      # this agent should get one
      obs_mkt = obs[agent_id]['Market']
      listings = self._filter_market_obs(agent, obs_mkt, Item.Potion.ITEM_TYPE_ID)
      if len(listings) > 0:
        # randomly selecting a listing
        #   and NOT going for only the cheapest one, because others will also want it
        item_id = np.random.choice(listings[:,ItemAttr["id"]].flatten())
        self._item.force_buy_idx[member_pos] = \
          np.argwhere(obs_mkt[:,ItemAttr["id"]] == item_id).item()

  def _buy_weapons_armors(self, obs):
    best_savers = [
      self._item.best_weapons,
      self._item.best_hats,
      self._item.best_tops,
      self._item.best_bottoms]

    proc_order = np.argsort(self._combat_score)
    for member_pos in proc_order: # start from lowest restore score
      agent_id = self._entity_helper.pos_to_agent_id(member_pos)
      if agent_id not in obs: # skip dead
        continue

      agent = self._entity_helper.agent_or_none(agent_id)
      obs_mkt = obs[agent_id]['Market']
      arms_types = [  # reflect priority
        ATK_TO_WEAPON[self._entity_helper.member_professions[member_pos]],
        *ARMORS]
      wishlist = []
      enhancements = []

      # process market listings for each weapon/armor type
      for item_type, saver in zip(arms_types, best_savers):
        listings = self._filter_market_obs(agent, obs_mkt, item_type)
        curr_best_lvl = saver[member_pos][ItemAttr["level"]] \
                          if saver[member_pos] is not None else -1
        flt_lvl = listings[:,ItemAttr['level']] > curr_best_lvl
        if len(listings[flt_lvl]) > 0:
          best_item = sorted(listings[flt_lvl], key=lambda x: x[ItemAttr["level"]])[-1]
          score_delta = (best_item[ItemAttr["level"]] - curr_best_lvl) * \
                          ARMOR_SCORE if item_type in ARMORS else WEAPON_SCORE
          wishlist.append(best_item)
          enhancements.append(score_delta)
        else:
          wishlist.append(None)
          enhancements.append(0)

      # get the item with best combat score enhancement
      if max(enhancements) > 0:
        to_buy = wishlist[enhancements.index(max(enhancements))]
        self._item.force_buy_idx[member_pos] = \
          np.argwhere(obs_mkt[:,ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()

  def _buy_consumables(self, obs):
    # trying to have at least two poultices, at least one ration
    buy_goal = [ (Item.Ration.ITEM_TYPE_ID, 1), (Item.Potion.ITEM_TYPE_ID, 2) ]
    proc_order = np.argsort(self._restore_score)
    for member_pos in proc_order: # start from lowest restore score
      agent_id = self._entity_helper.pos_to_agent_id(member_pos)
      if agent_id not in obs: # skip dead
        continue

      agent = self._entity_helper.agent_or_none(agent_id)
      for cons_type, cons_target in buy_goal:
        if self._item.force_buy_idx[member_pos]: # already have something to buy
          continue
        my_items = obs[agent_id]['Inventory'][:,ItemAttr['type_id']]
        if sum(my_items == cons_type) >= cons_target: # already have enough items
          continue

        # this agent should get one
        obs_mkt = obs[agent_id]['Market']
        acceptable_price = 2 + self.curr_step // 300 # CHECK ME: constants
        listings = self._filter_market_obs(agent, obs_mkt, cons_type, acceptable_price)
        if len(listings) > 0:
          min_price = min(listings[:,ItemAttr['listed_price']])
          if min_price < acceptable_price:
            listings = self._filter_market_obs(agent, obs_mkt, cons_type, min_price)

          # this agent can go for the cheapest option because not urgent
          #   if there are multiple min price listings, select one in random
          item_id = np.random.choice(listings[:,ItemAttr["id"]].flatten())
          self._item.force_buy_idx[member_pos] = \
            np.argwhere(obs_mkt[:,ItemAttr["id"]] == item_id).item()
