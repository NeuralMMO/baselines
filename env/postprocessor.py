
from os import environ
from re import T
import resource
from typing import Dict, List
from attr import dataclass

import nmmo
from pettingzoo.utils.env import AgentID
import numpy as np

from nmmo.lib.log import EventCode

import pufferlib
import pufferlib.emulation


@dataclass
class RewardsConfig:
  symlog_rewards: bool = False
  hunger: bool = False
  thirst: bool = False
  health: bool = False
  achievements: bool = False
  environment: bool = False

class Postprocessor(pufferlib.emulation.Postprocessor):
  def __init__(self, env, teams, team_id, rewards_config):
    self._rewards_config = rewards_config
    self._achievements = {}
    super().__init__(env, teams, team_id)

  def reset(self, team_obs):
    self._achievements = {
      id: {} for id in self.env.possible_agents
    }

  def rewards(self, rewards, dones, infos, step):
    agents = list(rewards.keys())
    if not self._rewards_config.environment:
      rewards = { id: 0 for id in agents }

    for agent_id in agents:
      if agent_id not in infos:
        infos[agent_id] = {}

      agent = self.env.realm.players.get(agent_id)

      # TODO: Check if this is correct for dead agents
      if agent is None:
        continue

      assert agent is not None, f'Agent {agent_id} not found'

      if self._rewards_config.hunger:
        if agent.food.val / self.env.config.RESOURCE_BASE < 0.4:
          rewards[agent_id] -= 0.1

      if self._rewards_config.thirst:
        if agent.water.val / self.env.config.RESOURCE_BASE < 0.4:
          rewards[agent_id] -= 0.1

      if self._rewards_config.health:
        if agent.health.val / self.env.config.PLAYER_BASE_HEALTH > 0.4:
          rewards[agent_id] -= 0.1

      if self._rewards_config.achievements:
        rewards[agent_id] += 10*score_unique_events(self.env.realm, agent_id)

      # This is probably bad
      if self._rewards_config.symlog_rewards:
        rewards[agent_id] = _symlog(rewards[agent_id])

    for agent_id in dones.keys():
      assert dones[agent_id], f'Agent {agent_id} is not done'
      # TODO: sometimes dead agents haven't been culled yet
      agent = self.env.realm.players.dead.get(
        agent_id, self.env.realm.players.get(agent_id))
      assert agent is not None, f'Agent {agent_id} not found'

      if agent_id not in infos:
        infos[agent_id] = {}

      infos[agent_id]["cod/starved"] = agent.food.val == 0
      infos[agent_id]["cod/dehydrated"] = agent.water.val == 0
      infos[agent_id]["cod/attacked"] = agent.damage.val > 0
      infos[agent_id]["lifespan"] = self.env.realm.tick

      infos[agent_id].update(get_player_history(self.env.realm, agent_id, agent))

    return sum(rewards.values()), infos

def _symlog(value):
    """Applies symmetrical logarithmic transformation to a float value."""
    sign = np.sign(value)
    abs_value = np.abs(value)

    if abs_value >= 1:
        log_value = np.log10(abs_value)
    else:
        log_value = abs_value

    symlog_value = sign * log_value
    return symlog_value


INFO_KEY_TO_EVENT_CODE = { 'event/'+evt.lower(): val for evt, val in EventCode.__dict__.items()
                           if isinstance(val, int) }

def get_player_history(realm, agent_id, agent):
  log = realm.event_log.get_data(agents = [agent_id])
  attr_to_col = realm.event_log.attr_to_col

  history = {}
  event_cnt = {}
  for key, code in INFO_KEY_TO_EVENT_CODE.items():
    # count the freq of each event
    event_cnt[key] = sum(log[:,attr_to_col['event']] == code)

  # CHECK ME: passing only interesting info
  key_event = ['eat_food', 'drink_water', 'score_hit', 'player_kill',
               'consume_item', 'harvest_item', 'list_item', 'buy_item']
  for evt in key_event:
    key = 'event/' + evt
    history[key] = event_cnt[key] > 0 # interested in whether the agent did this or not

  history['achieved/unique_events'] = score_unique_events(realm, agent_id, score_diff=False)
  history['achieved/exploration'] = agent.history.exploration
  history['achieved/player_kills'] = event_cnt['event/player_kill']

  check_max = {
    'level': EventCode.LEVEL_UP,
    'damage': EventCode.SCORE_HIT, }
  for attr, code in check_max.items():
    idx = log[:,attr_to_col['event']] == code
    history['achieved/max_'+attr] = \
      max(log[idx,attr_to_col[attr]]) if sum(idx) > 0 else 0

  # TODO: consume ration/poultice?

  return history

def score_unique_events(realm, agent_id, score_diff=True):
  """Calculate score by counting unique events.

    score_diff = True gives the difference score for the current tick
    score_diff = False gives the number of all unique events in the episode

    EAT_FOOD, DRINK_WATER, GIVE_ITEM, DESTROY_ITEM, GIVE_GOLD are counted only once
      because the details of these events are not recorded at all

    Count all PLAYER_KILL, EARN_GOLD (sold item), LEVEL_UP events
  """
  log = realm.event_log.get_data(agents = [agent_id])
  attr_to_col = realm.event_log.attr_to_col

  if len(log) == 0: # no event logs
    return 0

  if score_diff:
    curr_idx = log[:,attr_to_col['tick']] == realm.tick
    if sum(curr_idx) == 0: # no new logs
      return 0

  # mask some columns to make the event redundant
  cols_to_ignore = {
    EventCode.SCORE_HIT: ['combat_style', 'damage'],
    EventCode.CONSUME_ITEM: ['quantity'], # treat each (item, level) differently
    EventCode.HARVEST_ITEM: ['quantity'], # but, count each (item, level) only once
    EventCode.EQUIP_ITEM: ['quantity'],
    EventCode.LIST_ITEM: ['quantity', 'price'],
    EventCode.BUY_ITEM: ['quantity', 'price'], }

  for code, attrs in cols_to_ignore.items():
    idx = log[:,attr_to_col['event']] == code
    for attr in attrs:
      log[idx,attr_to_col[attr]] = 0

  # make every EARN_GOLD events unique
  idx = log[:,attr_to_col['event']] == EventCode.EARN_GOLD
  log[idx,attr_to_col['number']] = log[idx,attr_to_col['id']].copy() # this is a hack

  # remove redundant events after masking
  unique_all = np.unique(log[:,attr_to_col['event']:], axis=0)
  score = len(unique_all)

  if score_diff:
    unique_prev = np.unique(log[~curr_idx,attr_to_col['event']:], axis=0)
    return score - len(unique_prev)

  return score
