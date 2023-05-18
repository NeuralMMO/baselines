# TODO: remove the below line
# pylint: disable=all

import collections

import nmmo.io.action as nmmo_act
import numpy as np
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.io import action
from nmmo.systems import item
from nmmo.systems.item import ItemState

from feature_extractor.item_helper import PROF_TO_ATK_TYPE


EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

USE_POULTICE = 0
USE_RATION = 1
SELL_POULTICE = 0
SELL_RATION = 1
N_MOVE = 4
# CHECK ME: these are also in other places
N_NPC_CONSIDERED = 9
N_ENEMY_CONSIDERED = 9
N_ATK_TARGET = N_NPC_CONSIDERED + N_ENEMY_CONSIDERED
N_ATK_TYPE = 3
N_USE = 2
N_SELL = 2

class Action:
  def __init__(self) -> None:
    self.prev_actions = None

  def reset(self):
    self.prev_actions = np.array([N_MOVE, N_ATK_TARGET, N_USE, N_SELL])[None, :] \
                          .repeat(self.team_size, axis=0)  # init as idle


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
          entity_obs[:,EntityAttr["id"]] == self.target_entity_id[i]).item()
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

