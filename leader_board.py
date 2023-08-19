import os
import time
import logging
from typing import Optional, List
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

import pufferlib
import pufferlib.emulation

from nmmo.core.realm import Realm
from nmmo.lib.log import EventCode
from nmmo.render.replay_helper import FileReplayHelper


@dataclass
class TeamResult:
    policy_id: str = None

    # event-log based, coming from process_event_log
    total_score: int = 0
    player_kill: int = 0,
    max_level: int = 0,
    max_damage: int = 0,
    max_distance: int = 0,
    eat_food_count: int = 0,
    drink_water_count: int = 0,
    attack_count: int = 0,
    item_equip_count: int = 0,
    item_harvest_count: int = 0,
    item_list_count: int = 0,
    item_buy_count: int = 0,

    # agent object based (fill these in the environment)
    # CHECK ME: perhaps create a stat wrapper for putting all stats in one place?
    time_alive: int = 0,
    gold_owned: int = 0,
    completed_task_count: int = 0,
    damage_received: int = 0,
    damage_inflicted: int = 0,
    ration_consumed: int = 0,
    potion_consumed: int = 0,
    melee_level: int = 0,
    range_level: int = 0,
    mage_level: int = 0,
    fishing_level: int = 0,
    herbalism_level: int = 0,
    prospecting_level: int = 0,
    carving_level: int = 0,
    alchemy_level: int = 0,

    # system-level
    n_timeout: Optional[int] = 0

    @classmethod
    def names(cls) -> List[str]:
        return [
            "total_score",
            "player_kill",
            "max_level",
            "max_damage",
            "max_distance",
            "eat_food_count",
            "drink_water_count",
            "attack_count",
            "item_equip_count",
            "item_harvest_count",
            "item_list_count",
            "item_buy_count",
            "time_alive",
            "gold_owned",
            "completed_task_count",
            "damage_received",
            "damage_inflicted",
            "ration_consumed",
            "potion_consumed",
            "melee_level",
            "range_level",
            "mage_level",
            "fishing_level",
            "herbalism_level",
            "prospecting_level",
            "carving_level",
            "alchemy_level",
        ]

def get_team_result(realm: Realm, teams, team_id):
    achieved, performed, event_cnt = process_event_log(realm, teams[team_id])
    team_result = TeamResult(
        policy_id = str(team_id), # TODO: put actual team name here
        total_score = achieved["achieved/unique_events"],
        player_kill = achieved["achieved/player_kill"],
        max_level = achieved["achieved/max_level"],
        max_damage = achieved["achieved/max_damage"],
        max_distance = achieved["achieved/max_distance"],
        eat_food_count = event_cnt["event/eat_food"],
        drink_water_count = event_cnt["event/drink_water"],
        attack_count = event_cnt["event/score_hit"],
        item_equip_count = event_cnt["event/equip_item"],
        item_harvest_count = event_cnt["event/harvest_item"],
        item_list_count = event_cnt["event/list_item"],
        item_buy_count = event_cnt["event/buy_item"],
    )

    return team_result, achieved, performed, event_cnt


class StatPostprocessor(pufferlib.emulation.Postprocessor):
    """Postprocessing actions and metrics of Neural MMO.
       Process wandb/leader board stats, and save replays.
    """
    def __init__(self, env, teams, team_id, replay_save_dir=None):
        super().__init__(env, teams, team_id)
        self._num_replays_saved = 0
        self._replay_save_dir = None
        if replay_save_dir is not None and self.team_id == 1:
            self._replay_save_dir = replay_save_dir
        self._replay_helper = None
        self._reset_episode_stats()

    def reset(self, team_obs, dummy=False):
        super().reset(team_obs)
        if not dummy:
            if self._replay_helper is None and self._replay_save_dir is not None:
                self._replay_helper = FileReplayHelper()
                self.env.realm.record_replay(self._replay_helper)
            if self._replay_helper is not None:
                self._replay_helper.reset()

        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self._cod_attacked = 0
        self._cod_starved = 0
        self._cod_dehydrated = 0
        self._task_completed = 0
        self._curriculum = defaultdict(list)

        # for team results
        self._time_alive = 0
        self._gold_owned = 0
        self._damage_received = 0
        self._damage_inflicted = 0
        self._ration_consumed = 0
        self._potion_consumed = 0
        self._melee_level = 0
        self._range_level = 0
        self._mage_level = 0
        self._fishing_level = 0
        self._herbalism_level = 0
        self._prospecting_level = 0
        self._carving_level = 0
        self._alchemy_level = 0

    def _update_stats(self, agent):
        task = self.env.agent_task_map[agent.ent_id][0]
        # For each task spec, record whether its max progress and reward count
        self._curriculum[task.spec_name].append((task._max_progress, task.reward_signal_count))
        if task.completed:
            self._task_completed += 1.0 / self.team_size

        if agent.damage.val > 0:
            self._cod_attacked += 1.0 / self.team_size
        elif agent.food.val == 0:
            self._cod_starved += 1.0 / self.team_size
        elif agent.water.val == 0:
            self._cod_dehydrated += 1.0 / self.team_size

        # For TeamResult
        self._time_alive += agent.history.time_alive.val
        self._gold_owned += agent.gold.val
        self._damage_received += agent.history.damage_received
        self._damage_inflicted += agent.history.damage_inflicted
        self._ration_consumed += agent.ration_consumed
        self._potion_consumed += agent.poultice_consumed
        self._melee_level += agent.melee_level.val
        self._range_level += agent.range_level.val
        self._mage_level += agent.mage_level.val
        self._fishing_level += agent.fishing_level.val
        self._herbalism_level += agent.herbalism_level.val
        self._prospecting_level += agent.prospecting_level.val
        self._carving_level += agent.carving_level.val
        self._alchemy_level += agent.alchemy_level.val

    def rewards(self, team_rewards, team_dones, team_infos, step):
        for agent_id in team_dones:
            if team_dones[agent_id] is True:
                agent = self.env.realm.players.dead_this_tick.get(
                    agent_id, self.env.realm.players.get(agent_id)
                )
                if agent is None:
                    continue
                self._update_stats(agent)

    def infos(self, team_reward, env_done, team_done, team_infos, step):
        """Update team infos and save replays."""
        team_infos = super().infos(team_reward, env_done, team_done, team_infos, step)
        team_infos["stats"] = defaultdict(float)

        if env_done:
            team_infos["stats"]["cod/attacked"] = self._cod_attacked
            team_infos["stats"]["cod/starved"] = self._cod_starved
            team_infos["stats"]["cod/dehydrated"] = self._cod_dehydrated
            team_infos["stats"]["task/completed"] = self._task_completed
            team_infos["curriculum"] = self._curriculum

            team_result, achieved, performed, _ = get_team_result(
                self.env.realm, self.teams, self.team_id
            )
            for key, val in list(achieved.items()) + list(performed.items()):
                team_infos["stats"][key] = float(val)

            # Fill in the TeamResult
            team_result.time_alive = self._time_alive
            team_result.gold_owned = self._gold_owned
            team_result.completed_task_count = round(self._task_completed * self.team_size)
            team_result.damage_received = self._damage_received
            team_result.damage_inflicted = self._damage_inflicted
            team_result.ration_consumed = self._ration_consumed
            team_result.potion_consumed = self._potion_consumed
            team_result.melee_level = self._melee_level
            team_result.range_level = self._range_level
            team_result.mage_level = self._mage_level
            team_result.fishing_level = self._fishing_level
            team_result.herbalism_level = self._herbalism_level
            team_result.prospecting_level = self._prospecting_level
            team_result.carving_level = self._carving_level
            team_result.alchemy_level = self._alchemy_level

            team_infos["team_results"] = (self.team_id, team_result)

            if self._replay_helper is not None:
                replay_file = os.path.join(
                    self._replay_save_dir, f"replay_{time.strftime('%Y%m%d_%H%M%S')}")
                logging.info("Saving replay to %s", replay_file)
                self._replay_helper.save(replay_file, compress=False)
                self._num_replays_saved += 1

        return team_infos

# Event processing utilities for Neural MMO.

INFO_KEY_TO_EVENT_CODE = {
    "event/" + evt.lower(): val
    for evt, val in EventCode.__dict__.items()
    if isinstance(val, int)
}

def process_event_log(realm, agent_list):
    """Process the event log and extract performed actions and achievements."""
    log = realm.event_log.get_data(agents=agent_list)
    attr_to_col = realm.event_log.attr_to_col

    # count the number of events
    event_cnt = {}
    for key, code in INFO_KEY_TO_EVENT_CODE.items():
        # count the freq of each event
        event_cnt[key] = int(sum(log[:, attr_to_col["event"]] == code))

    # convert the numbers into binary (performed or not) for the key events
    key_event = [
        "eat_food",
        "drink_water",
        "score_hit",
        "player_kill",
        "equip_item",
        "consume_item",
        "harvest_item",
        "list_item",
        "buy_item",
    ]
    performed = {}
    for evt in key_event:
        key = "event/" + evt
        performed[key] = event_cnt[key] > 0

    # record important achievements
    achieved = {}
    check_max = {
        "level": EventCode.LEVEL_UP,
        "damage": EventCode.SCORE_HIT,
        "distance": EventCode.GO_FARTHEST,
    }
    for attr, code in check_max.items():
        idx = log[:, attr_to_col["event"]] == code
        achieved["achieved/max_" + attr] = (
            int(max(log[idx, attr_to_col[attr]])) if sum(idx) > 0 else 0
        )
    # correct the initial level
    if achieved["achieved/max_level"] == 0:
        achieved["achieved/max_level"] = 1
    achieved["achieved/player_kill"] = event_cnt["event/player_kill"]
    achieved["achieved/unique_events"] = score_unique_events(
        realm, log, score_diff=False
    )

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

    if len(log) == 0:  # no event logs
        return 0

    if score_diff:
        curr_idx = log[:, attr_to_col["tick"]] == realm.tick
        if sum(curr_idx) == 0:  # no new logs
            return 0

    # mask some columns to make the event redundant
    cols_to_ignore = {
        EventCode.SCORE_HIT: ["combat_style", "damage"],
        # treat each (item, level) differently
        EventCode.CONSUME_ITEM: ["quantity"],
        # but, count each (item, level) only once
        EventCode.HARVEST_ITEM: ["quantity"],
        EventCode.EQUIP_ITEM: ["quantity"],
        EventCode.LIST_ITEM: ["quantity", "price"],
        EventCode.BUY_ITEM: ["quantity", "price"],
    }

    for code, attrs in cols_to_ignore.items():
        idx = log[:, attr_to_col["event"]] == code
        for attr in attrs:
            log[idx, attr_to_col[attr]] = 0

    # make every EARN_GOLD events unique
    idx = log[:, attr_to_col["event"]] == EventCode.EARN_GOLD
    log[idx, attr_to_col["number"]] = log[
        idx, attr_to_col["tick"]
    ].copy()  # this is a hack

    # remove redundant events after masking
    unique_all = np.unique(log[:, attr_to_col["event"]:], axis=0)
    score = len(unique_all)

    if score_diff:
        unique_prev = np.unique(log[~curr_idx, attr_to_col["event"]:], axis=0)
        score -= len(unique_prev)

        # reward hack to make agents learn to eat and drink
        basic_idx = np.in1d(
            log[curr_idx, attr_to_col["event"]],
            [EventCode.EAT_FOOD, EventCode.DRINK_WATER],
        )
        if sum(basic_idx) > 0:
            score += (
                1 if realm.tick < 200 else np.random.choice([0, 1], p=[2 / 3, 1 / 3])
            )  # use prob. reward after 200 ticks

        return min(2, score)  # clip max score to 2

    return score
