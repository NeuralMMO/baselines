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

class Market:
  def __init__(self, config: nmmo.config.AllGameSystems) -> None:
    self.config = config

  def reset(self):
    pass

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
