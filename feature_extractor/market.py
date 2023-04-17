# TODO: remove the below line
# pylint: disable=all

import copy

import nmmo
import numpy as np
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems import item
from nmmo.systems.item import ItemState

from team_helper import TeamHelper
from feature_extractor.item import ARMORS, ATK_TO_WEAPON, N_PROF, ITEM_TO_PROF_IDX, WEAPONS
from feature_extractor.inventory import Inventory

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

LOW_HEALTH = 35

class Market:
  def __init__(self, config: nmmo.config.AllGameSystems,
               team_id: int, team_helper: TeamHelper,
               inventory: Inventory, curr_step: int) -> None:

    self.config = config

    self._team_id = team_id
    self._team_agent_ids = team_helper.teams[team_id]
    self.TEAM_SIZE = len(self._team_agent_ids)
    self.curr_step = curr_step

    self.inventory = inventory
    self.item = inventory.item
    self.raw_market_obs = None
    self.market_obs = None
    self.will_die_id = None
    self.will_die_gold = None
    self.team_n_poultice = [None] * self.TEAM_SIZE

    # CHECK ME: does NMMO has this restriction? or heuristic?
    self.rescue_cooldown = None  # cooldown rounds of sell poultice to teammates

  def reset(self):
    self.rescue_cooldown = 0

  def update(self, obs):
    self.rescue_cooldown = max(0, self.rescue_cooldown - 1)

    self.item.force_buy_idx = [None] * self.TEAM_SIZE
    alive_ids = np.array([i for i in range(self.TEAM_SIZE) if i in obs])
    self.raw_market_obs = obs[alive_ids[0]]['Market']
    self.market_obs = copy.deepcopy(self.raw_market_obs)

    combat_ratings = self._calculate_combat_ratings(obs)
    self._buy_weapons_armors(obs, combat_ratings)
    survival_ratings = self._calculate_survival_ratings(obs)
    self._emergent_buy_poultice(obs)
    self._sell_poultice(obs)
    self._buy_poultice(obs, survival_ratings) 

  def _calculate_combat_ratings(self, obs) -> np.ndarray:
    # combat rating
    ratings = []
    for member_id in obs.keys():
      rating = 0
      if self.item.best_weapons[member_id] is not None:
        rating += self.item.best_weapons[member_id][ItemAttr["level"]] * 10
      elif self.item.best_tools[member_id] is not None:
        rating += self.item.best_tools[member_id][ItemAttr["level"]] * 4
      if self.item.best_hats[member_id] is not None:
        rating += self.item.best_hats[member_id][ItemAttr["level"]] * 4
      if self.item.best_tops[member_id] is not None:
        rating += self.item.best_tops[member_id][ItemAttr["level"]] * 4
      if self.item.best_bottoms[member_id] is not None:
        rating += self.item.best_bottoms[member_id][ItemAttr["level"]] * 4
      ratings.append(rating)
    return np.array(ratings)
  
  def _buy_weapons_armors(self, obs, ratings) -> None:

    agent_order = np.argsort(ratings)
    for agent_id in agent_order:
      entity_obs = obs[agent_id]['Entity']
      my_obs = entity_obs[entity_obs[EntityAttr["id"]] == agent_id]
      my_gold = my_obs[EntityAttr["gold"]]

      care_types = [ATK_TO_WEAPON[self.item.prof[agent_id]], *ARMORS]
      savers = [
          self.item.best_weapons,
          self.item.best_hats,
          self.item.best_tops,
          self.item.best_bottoms,
      ]

      wishlist = []
      enhancements = []
      for typ, saver in zip(care_types, savers):
        if typ in ARMORS:
          max_equipable_lvl = max(my_obs[-N_PROF:])  # maximum of all levels
        else:
          max_equipable_lvl = my_obs[ITEM_TO_PROF_IDX[typ]]
        curr_best_lvl = 0
        if saver[agent_id] is not None:
          curr_best_lvl = saver[agent_id][ItemAttr["level"]]
        mkt_comds = self.market_obs[self.market_obs[:, ItemAttr["type_id"]] == typ]
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
        self.item.force_buy_idx[agent_id] = np.argwhere(self.raw_market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
        idx = np.argwhere(self.market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
        self.market_obs[idx][ItemAttr["quantity"]] -= 1
        if self.market_obs[idx][ItemAttr["quantity"]] == 0:
          # remove from market obs to prevent competition among teammates
          self.market_obs = np.concatenate([self.market_obs[:idx], self.market_obs[idx+1:]])

  def _calculate_survival_ratings(self, obs) -> np.ndarray:
     # survival rating
    ratings = []
    for _, member_obs in obs.items():
      rating = 0
      my_items = member_obs['Inventory']
      n_poultice = len(my_items[my_items[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID])
      n_ration = len(my_items[my_items[:, ItemAttr["type_id"]] == item.Ration.ITEM_TYPE_ID])
      if n_poultice == 1:
        rating += 4
      elif n_poultice >= 2:
        rating += 8
      if n_ration >= 1:
        rating += 2
      ratings.append(rating)
    return np.array(ratings)

  def _sell_poultice(self, obs):
    # sell poultice for emergent rescue
    if self.will_die_id is not None and self.item.rescue_cooldown == 0 and self.will_die_gold > 0:
      for member_id in reversed(obs.keys()):
        if self.team_n_poultice[member_id] > 0 and self.item.force_sell_idx[member_id] is not None:
          my_items = obs[member_id]['Inventory']
          my_poultices = my_items[my_items[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
          entity_ob = next(iter(obs.values()))['Entity']
          team_pop = entity_ob[entity_ob[EntityAttr["id"]] == member_id][EntityAttr["population_id"]]
          to_sell = sorted(my_poultices, key=lambda x: x[ItemAttr["level"]])[-1]  # sell the best
          idx = np.argwhere(my_items[:, ItemAttr["id"]] == to_sell[ItemAttr["id"]]).item()
          self.item.force_sell_idx[member_id] = idx
          self.item.force_sell_price[member_id] = max(int(self.will_die_gold // 2), 1) if team_pop > 0 else 1
          self.item.rescue_cooldown = 3
          break

  def _buy_poultice(self, obs, survival_ratings):
  # normal case to buy at least two poultices, at least one ration
    agent_order = np.argsort(survival_ratings)
    for cons_type in [item.Poultice.ITEM_TYPE_ID, item.Ration.ITEM_TYPE_ID]:
      for member_id in agent_order:
        if self.item.force_buy_idx[member_id] is not None:
          continue
        my_items = obs[member_id]['Inventory']
        if cons_type == item.Ration.ITEM_TYPE_ID and len(my_items[my_items[:, ItemAttr["type_id"]] == cons_type]) >= 1:
          continue
        if cons_type == item.Poultice.ITEM_TYPE_ID and len(my_items[my_items[:, ItemAttr["type_id"]] == cons_type]) >= 2:
          continue
        mkt_cons = self.market_obs[self.market_obs[:, ItemAttr["type_id"]] == cons_type]
        acceptable_price = 2 + self.curr_step // 300
        mkt_cons = mkt_cons[mkt_cons[:, ItemAttr["listed_price"]] <= acceptable_price]
        if len(mkt_cons) > 0:
          to_buy = sorted(mkt_cons, key=lambda x: x[ItemAttr["listed_price"]])[0]  # cheapest
          self.item.force_buy_idx[member_id] = np.argwhere(self.raw_market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
          idx = np.argwhere(self.market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
          self.market_obs[idx][ItemAttr["quantity"]] -= 1
          if self.market_obs[idx][ItemAttr["quantity"]] == 0:
            # remove from market obs to prevent repeatedly buying among teammates
            self.market_obs = np.concatenate([self.market_obs[:idx], self.market_obs[idx + 1:]])

  def _emergent_buy_poultice(self, obs):
    # emergent case to buy poultice
    for member_id, member_obs in obs:
      entity_obs = member_obs['Entity']
      my_obs = entity_obs[entity_obs[EntityAttr["id"]] == member_id]
      my_items = member_obs['Inventory']
      my_poultices = my_items[my_items[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
      self.team_n_poultice[member_id] = len(my_poultices)
      if len(my_poultices) > 0:  # not emergent
        continue
      if my_obs[EntityAttr["health"]] > LOW_HEALTH:  # not emergent
        continue
      my_gold = my_obs[EntityAttr["gold"]]
      mkt_ps = market_obs[market_obs[:, ItemAttr["type_id"]] == item.Poultice.ITEM_TYPE_ID]
      mkt_ps = mkt_ps[mkt_ps[:, ItemAttr["listed_price"]] <= my_gold]
      if len(mkt_ps) > 0:
        to_buy = sorted(mkt_ps, key=lambda x: x[ItemAttr["listed_price"]])[0]  # cheapest
        self.item.force_buy_idx[member_id] = np.argwhere(self.raw_market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
        idx = np.argwhere(market_obs[:, ItemAttr["id"]] == to_buy[ItemAttr["id"]]).item()
        market_obs[idx][ItemAttr["quantity"]] -= 1
        if market_obs[idx][ItemAttr["quantity"]] == 0:
          # remove from market obs to prevent repeatedly buying among teammates
          market_obs = np.concatenate([market_obs[:idx], market_obs[idx+1:]])
      else:
        self.will_die_id = member_id
        self.will_die_gold = my_gold