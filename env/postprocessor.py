
from collections import defaultdict
from os import environ
import os
from re import T
import resource
from typing import Dict, List
from xxlimited import foo
from attr import dataclass

import nmmo
from pettingzoo.utils.env import AgentID
import numpy as np

from nmmo.lib.log import EventCode

import pufferlib
import pufferlib.emulation
from traitlets import default

class Postprocessor(pufferlib.emulation.Postprocessor):
  def __init__(self, env, teams, team_id):
    self._achievements = {}
    super().__init__(env, teams, team_id)

  def reset(self, team_obs):
    super().reset(team_obs)
    self._achievements = {
      id: {} for id in self.env.possible_agents
    }

  def rewards(self, rewards, dones, infos, step):
    agents = list(set(rewards.keys()).union(set(dones.keys())))

    team_reward = sum(rewards.values())
    team_info = {"stats": defaultdict(float)}

    for agent_id in agents:
      agent = self.env.realm.players.dead_this_tick.get(
        agent_id, self.env.realm.players.get(agent_id))

      if agent is None:
        continue

    return team_reward, team_info

# Not currently used

# INFO_KEY_TO_EVENT_CODE = { 'event/'+evt.lower(): val for evt, val in EventCode.__dict__.items()
#                            if isinstance(val, int) }

# def get_player_history(realm, agent_id, agent):
#   log = realm.event_log.get_data(agents = [agent_id])
#   attr_to_col = realm.event_log.attr_to_col

#   history = {}
#   event_cnt = {}
#   for key, code in INFO_KEY_TO_EVENT_CODE.items():
#     # count the freq of each event
#     event_cnt[key] = sum(log[:,attr_to_col['event']] == code)

#   # CHECK ME: passing only interesting info
#   key_event = ['eat_food', 'drink_water', 'score_hit', 'player_kill',
#                'consume_item', 'harvest_item', 'list_item', 'buy_item']
#   for evt in key_event:
#     key = 'event/' + evt
#     history[key] = event_cnt[key] > 0 # interested in whether the agent did this or not

#   history['achieved/unique_events'] = score_unique_events(realm, agent_id, score_diff=False)
#   history['achieved/exploration'] = agent.history.exploration
#   history['achieved/player_kills'] = event_cnt['event/player_kill']

#   check_max = {
#     'level': EventCode.LEVEL_UP,
#     'damage': EventCode.SCORE_HIT, }
#   for attr, code in check_max.items():
#     idx = log[:,attr_to_col['event']] == code
#     history['achieved/max_'+attr] = \
#       max(log[idx,attr_to_col[attr]]) if sum(idx) > 0 else 0

#   # TODO: consume ration/poultice?

#   return history

# def score_unique_events(realm, agent_id, score_diff=True):
#   """Calculate score by counting unique events.

#     score_diff = True gives the difference score for the current tick
#     score_diff = False gives the number of all unique events in the episode

#     EAT_FOOD, DRINK_WATER, GIVE_ITEM, DESTROY_ITEM, GIVE_GOLD are counted only once
#       because the details of these events are not recorded at all

#     Count all PLAYER_KILL, EARN_GOLD (sold item), LEVEL_UP events
#   """
#   log = realm.event_log.get_data(agents = [agent_id])
#   attr_to_col = realm.event_log.attr_to_col

#   if len(log) == 0: # no event logs
#     return 0

#   if score_diff:
#     curr_idx = log[:,attr_to_col['tick']] == realm.tick
#     if sum(curr_idx) == 0: # no new logs
#       return 0

#   # mask some columns to make the event redundant
#   cols_to_ignore = {
#     EventCode.SCORE_HIT: ['combat_style', 'damage'],
#     EventCode.CONSUME_ITEM: ['quantity'], # treat each (item, level) differently
#     EventCode.HARVEST_ITEM: ['quantity'], # but, count each (item, level) only once
#     EventCode.EQUIP_ITEM: ['quantity'],
#     EventCode.LIST_ITEM: ['quantity', 'price'],
#     EventCode.BUY_ITEM: ['quantity', 'price'], }

#   for code, attrs in cols_to_ignore.items():
#     idx = log[:,attr_to_col['event']] == code
#     for attr in attrs:
#       log[idx,attr_to_col[attr]] = 0

#   # make every EARN_GOLD events unique
#   idx = log[:,attr_to_col['event']] == EventCode.EARN_GOLD
#   log[idx,attr_to_col['number']] = log[idx,attr_to_col['id']].copy() # this is a hack

#   # remove redundant events after masking
#   unique_all = np.unique(log[:,attr_to_col['event']:], axis=0)
#   score = len(unique_all)

#   if score_diff:
#     unique_prev = np.unique(log[~curr_idx,attr_to_col['event']:], axis=0)
#     return score - len(unique_prev)

#   return score
