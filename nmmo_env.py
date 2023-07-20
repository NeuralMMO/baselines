from argparse import ArgumentParser, Namespace
from collections import defaultdict

import numpy as np
import nmmo
from nmmo.lib.log import EventCode

import pufferlib
import pufferlib.emulation
from nmmo.render.replay_helper import FileReplayHelper
from typing import Any, Dict

def add_args(parser: ArgumentParser):
  parser.add_argument(
      "--env.num_agents",
      dest="num_agents",
      type=int,
      default=128,
      help="number of agents to use for training (default: 128)",
  )
  parser.add_argument(
      "--env.num_npcs",
      dest="num_npcs",
      type=int,
      default=0,
      help="number of NPCs to use for training (default: 256)",
  )
  parser.add_argument(
      "--env.max_episode_length",
      dest="max_episode_length",
      type=int,
      default=1024,
      help="number of steps per episode (default: 1024)",
  )
  parser.add_argument(
      "--env.death_fog_tick",
      dest="death_fog_tick",
      type=int,
      default=None,
      help="number of ticks before death fog starts (default: None)",
  )
  parser.add_argument(
      "--env.num_maps",
      dest="num_maps",
      type=int,
      default=128,
      help="number of maps to use for training (default: 1)",
  )
  parser.add_argument(
      "--env.maps_path",
      dest="maps_path",
      type=str,
      default="maps/train/",
      help="path to maps to use for training (default: None)",
  )
  parser.add_argument(
      "--env.map_size",
      dest="map_size",
      type=int,
      default=128,
      help="size of maps to use for training (default: 128)",
  )
  parser.add_argument(
      "--env.tasks_path",
      dest="tasks_path",
      type=str,
      default=None,
      help="path to tasks to use for training (default: tasks.pkl)",
  )


class Config(
    nmmo.config.Medium,
    nmmo.config.Terrain,
    nmmo.config.Resource,
    nmmo.config.Progression,
    nmmo.config.Equipment,
    nmmo.config.Item,
    nmmo.config.Exchange,
    nmmo.config.Combat,
    nmmo.config.NPC,
):
  def __init__(self, args: Namespace):
    super().__init__()

    self.PROVIDE_ACTION_TARGETS = True
    self.MAP_FORCE_GENERATION = False
    self.PLAYER_N = args.num_agents
    self.HORIZON = args.max_episode_length
    self.MAP_N = args.num_maps
    self.PLAYER_DEATH_FOG = args.death_fog_tick
    self.PATH_MAPS = f"{args.maps_path}/{args.map_size}/"
    self.MAP_CENTER = args.map_size
    self.NPC_N = args.num_npcs
    self.CURRICULUM_FILE_PATH = args.tasks_path


class Postprocessor(pufferlib.emulation.Postprocessor):
  def __init__(self, env, teams, team_id, replay_save_dir=None):
    super().__init__(env, teams, team_id)
    self._replay_save_dir = replay_save_dir
    if self._replay_save_dir is not None:
      self._replay_helper = FileReplayHelper()
      env.realm.record_replay(self._replay_helper)
    self._reset_episode_stats()


  def _reset_episode_stats(self):
    self._cod_attacked = 0
    self._cod_starved = 0
    self._cod_dehydrated = 0
    self._task_completed = 0

  def rewards(self, team_rewards, team_dones, team_infos, step):
    team_reward = sum(team_rewards.values())
    team_info = {"stats": defaultdict(float)}

    for agent_id in team_dones:
      if team_dones[agent_id] is True:
        agent = self.env.realm.players.dead_this_tick.get(
            agent_id, self.env.realm.players.get(agent_id)
        )
        if agent is None:
          continue

        # check if the agent has completed the task
        task = self.env.agent_task_map[agent_id][0]
        if task.completed:
          # NOTE: The default StayAlive task returns True after the first tick
          self._task_completed += 1. / self.team_size

        # log the cause of death for each dead agent
        if agent.damage.val > 0:
          self._cod_attacked += 1. / self.team_size
        elif agent.food.val == 0:
          self._cod_starved += 1. / self.team_size
        elif agent.water.val == 0:
          self._cod_dehydrated += 1. / self.team_size

    return team_reward, team_info

  def infos(self, team_reward, env_done, team_done, team_infos, step):
    team_infos = super().infos(team_reward, env_done, team_done, team_infos, step)

    # record the stats when the episode ends
    if env_done:
      team_infos["stats"]["cod/attacked"] = self._cod_attacked
      team_infos["stats"]["cod/starved"] = self._cod_starved
      team_infos["stats"]["cod/dehydrated"] = self._cod_dehydrated
      team_infos["stats"]["task/completed"] = self._task_completed

      achieved, performed, _ = \
        process_event_log(self.env.realm, self.teams[self.team_id])
      for key, val in list(achieved.items()) + list(performed.items()):
        team_infos["stats"][key] = float(val)

    return team_infos

  # def features(self, obs, step):
  #   # for ob in obs.values():
  #   #   ob["featurized"] = self._feature_extractor(obs)
  #   return obs

  # def actions(self, actions, step):
  #   return self._feature_extractor.translate_actions(actions)

def create_binding(args: Namespace):
  return pufferlib.emulation.Binding(
      env_cls=nmmo.Env,
      default_args=[Config(args)],
      env_name="Neural MMO",
      suppress_env_prints=False,
      emulate_const_horizon=args.max_episode_length,
      postprocessor_cls=Postprocessor,
      postprocessor_args=[],
  )


#####################################################################

INFO_KEY_TO_EVENT_CODE = { 'event/'+evt.lower(): val for evt, val in EventCode.__dict__.items()
                           if isinstance(val, int) }

def process_event_log(realm, agent_list):
  log = realm.event_log.get_data(agents=agent_list)
  attr_to_col = realm.event_log.attr_to_col

  # count the number of events
  event_cnt = {}
  for key, code in INFO_KEY_TO_EVENT_CODE.items():
    # count the freq of each event
    event_cnt[key] = sum(log[:,attr_to_col["event"]] == code)

  # convert the numbers into binary (performed or not) for the key events
  key_event = ["eat_food", "drink_water", "score_hit", "player_kill",
               "equip_item", "consume_item", "harvest_item", "list_item", "buy_item"]
  performed = {}
  for evt in key_event:
    key = "event/" + evt
    performed[key] = event_cnt[key] > 0

  # record important achievements
  achieved = {}
  check_max = {
    "level": EventCode.LEVEL_UP,
    "damage": EventCode.SCORE_HIT,
    "distance": EventCode.GO_FARTHEST
  }
  for attr, code in check_max.items():
    idx = log[:,attr_to_col["event"]] == code
    achieved["achieved/max_"+attr] = \
      max(log[idx,attr_to_col[attr]]) if sum(idx) > 0 else 0
  # correct the initial level
  if achieved["achieved/max_level"] == 0:
    achieved["achieved/max_level"] = 1
  achieved["achieved/player_kill"] = event_cnt["event/player_kill"]
  achieved["achieved/unique_events"] = score_unique_events(realm, log, score_diff=False)

  # TODO: log consume ration/poultice?

  return achieved, performed, event_cnt

def score_unique_events(realm, log, score_diff=True):
  """Calculate score by counting unique events.

    score_diff = True gives the difference score for the current tick
    score_diff = False gives the number of all unique events in the episode

    EAT_FOOD, DRINK_WATER, GIVE_ITEM, DESTROY_ITEM, GIVE_GOLD are counted only once
      because the details of these events are not recorded at all

    Count all PLAYER_KILL, EARN_GOLD (sold item), LEVEL_UP events
  """
  attr_to_col = realm.event_log.attr_to_col

  if len(log) == 0: # no event logs
    return 0

  if score_diff:
    curr_idx = log[:,attr_to_col["tick"]] == realm.tick
    if sum(curr_idx) == 0: # no new logs
      return 0

  # mask some columns to make the event redundant
  cols_to_ignore = {
    EventCode.SCORE_HIT: ["combat_style", "damage"],
    EventCode.CONSUME_ITEM: ["quantity"], # treat each (item, level) differently
    EventCode.HARVEST_ITEM: ["quantity"], # but, count each (item, level) only once
    EventCode.EQUIP_ITEM: ["quantity"],
    EventCode.LIST_ITEM: ["quantity", "price"],
    EventCode.BUY_ITEM: ["quantity", "price"], }

  for code, attrs in cols_to_ignore.items():
    idx = log[:,attr_to_col["event"]] == code
    for attr in attrs:
      log[idx,attr_to_col[attr]] = 0

  # make every EARN_GOLD events unique
  idx = log[:,attr_to_col["event"]] == EventCode.EARN_GOLD
  log[idx,attr_to_col["number"]] = log[idx,attr_to_col["tick"]].copy() # this is a hack

  # remove redundant events after masking
  unique_all = np.unique(log[:,attr_to_col["event"]:], axis=0)
  score = len(unique_all)

  if score_diff:
    unique_prev = np.unique(log[~curr_idx,attr_to_col["event"]:], axis=0)
    score -= len(unique_prev)

    # reward hack to make agents learn to eat and drink
    basic_idx = np.in1d(log[curr_idx,attr_to_col["event"]],
                        [EventCode.EAT_FOOD, EventCode.DRINK_WATER])
    if sum(basic_idx) > 0:
      score += 1 if realm.tick < 200 else \
        np.random.choice([0, 1], p=[2/3, 1/3]) # use prob. reward after 200 ticks

    return min(2, score) # clip max score to 2

  return score
