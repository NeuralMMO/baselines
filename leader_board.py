from typing import Optional, List
from dataclasses import dataclass
from collections import defaultdict, Counter

import math
import numpy as np

import pufferlib
import pufferlib.emulation

from nmmo.core.realm import Realm
from nmmo.lib.log import EventCode
import nmmo.systems.item as Item

@dataclass
class TeamResult:
    policy_id: str = None

    # event-log based, coming from process_event_log
    total_score: int = 0
    agent_kill_count: int = 0,
    npc_kill_count: int = 0,
    max_combat_level: int = 0,
    max_harvest_level: int = 0,
    max_damage: int = 0,
    max_progress_to_center: int = 0,
    eat_food_count: int = 0,
    drink_water_count: int = 0,
    attack_count: int = 0,
    item_harvest_count: int = 0,
    item_list_count: int = 0,
    item_buy_count: int = 0,

    # agent object based (fill these in the environment)
    # CHECK ME: perhaps create a stat wrapper for putting all stats in one place?
    time_alive: int = 0,
    earned_gold: int = 0,
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
            "agent_kill_count",
            "npc_kill_count",
            "max_combat_level",
            "max_harvest_level",
            "max_damage",
            "max_progress_to_center",
            "eat_food_count",
            "drink_water_count",
            "attack_count",
            "item_equip_count",
            "item_harvest_count",
            "item_list_count",
            "item_buy_count",
            "time_alive",
            "earned_gold",
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

def get_episode_result(realm: Realm, agent_id):
    achieved, performed, event_cnt = process_event_log(realm, [agent_id])
    # NOTE: Not actually a "team" result. Just a "team" of one agent
    result = TeamResult(
        policy_id = str(agent_id),  # TODO: put actual team/policy name here
        agent_kill_count = achieved["achieved/agent_kill_count"],
        npc_kill_count = achieved["achieved/npc_kill_count"],
        max_damage = achieved["achieved/max_damage"],
        max_progress_to_center = achieved["achieved/max_progress_to_center"],
        eat_food_count = event_cnt["event/eat_food"],
        drink_water_count = event_cnt["event/drink_water"],
        attack_count = event_cnt["event/score_hit"],
        item_harvest_count = event_cnt["event/harvest_item"],
        item_list_count = event_cnt["event/list_item"],
        item_buy_count = event_cnt["event/buy_item"],
    )

    return result, achieved, performed, event_cnt


class StatPostprocessor(pufferlib.emulation.Postprocessor):
    """Postprocessing actions and metrics of Neural MMO.
       Process wandb/leader board stats, and save replays.
    """
    def __init__(self, env, agent_id):
        super().__init__(env, is_multiagent=True, agent_id=agent_id)
        self._reset_episode_stats()

    def reset(self, observation):
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.epoch_return = 0
        self.epoch_length = 0

        self._cod_attacked = 0
        self._cod_starved = 0
        self._cod_dehydrated = 0
        self._task_completed = 0
        self._task_with_2_reward_signal = 0
        self._task_with_0p2_max_progress = 0
        self._curriculum = defaultdict(list)
        self._combat_level = []
        self._harvest_level = []
        self._prev_unique_count = 0
        self._curr_unique_count = 0

        # for agent results
        self._time_alive = 0
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

        # saving actions for masking/scoring
        self._last_moves = []
        self._last_price = 0

    def _update_stats(self, agent):
        task = self.env.agent_task_map[agent.ent_id][0]
        # For each task spec, record whether its max progress and reward count
        self._curriculum[task.spec_name].append((task._max_progress, task.reward_signal_count))
        if task.reward_signal_count >= 2:
            self._task_with_2_reward_signal += 1.0
        if task._max_progress >= 0.2:
            self._task_with_0p2_max_progress += 1.0
        if task.completed:
            self._task_completed += 1.0

        if agent.damage.val > 0:
            self._cod_attacked += 1.0
        elif agent.food.val == 0:
            self._cod_starved += 1.0
        elif agent.water.val == 0:
            self._cod_dehydrated += 1.0

        self._combat_level.append(agent.attack_level)
        self._harvest_level.append(max(
            agent.fishing_level.val,
            agent.herbalism_level.val,
            agent.prospecting_level.val,
            agent.carving_level.val,
            agent.alchemy_level.val,
        ))

        # For TeamResult
        self._time_alive += agent.history.time_alive.val
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

    def observation(self, observation):
        # Mask out the last selected price
        observation["ActionTargets"]["Sell"]["Price"][self._last_price] = 0
        return observation

    def action(self, action):
        self._last_moves.append(action[8])  # 8 is the index for move direction
        self._last_price = action[10]  # 10 is the index for selling price
        return action

    def reward_done_info(self, reward, done, info):
        """Update stats + info and save replays."""

        # Remove the task from info. Curriculum info is processed in _update_stats()
        info.pop('task', None)

        # Count and store unique event counts for easier use
        log = self.env.realm.event_log.get_data(agents=[self.agent_id])
        self._prev_unique_count = self._curr_unique_count
        self._curr_unique_count = len(extract_unique_event(log, self.env.realm.event_log.attr_to_col))

        if not done:
            self.epoch_length += 1
            self.epoch_return += reward
            return reward, done, info

        if 'stats' not in info:
            info['stats'] = {}

        agent = self.env.realm.players.dead_this_tick.get(
            self.agent_id, self.env.realm.players.get(self.agent_id)
        )
        assert agent is not None
        self._update_stats(agent)

        info['return'] = self.epoch_return
        info['length'] = self.epoch_length

        info["stats"]["cod/attacked"] = self._cod_attacked
        info["stats"]["cod/starved"] = self._cod_starved
        info["stats"]["cod/dehydrated"] = self._cod_dehydrated
        info["stats"]["task/completed"] = self._task_completed
        info["stats"]["task/pcnt_2_reward_signal"] = self._task_with_2_reward_signal
        info["stats"]["task/pcnt_0p2_max_progress"] = self._task_with_0p2_max_progress
        info["stats"]["achieved/max_combat_level"] = max(self._combat_level)
        info["stats"]["achieved/max_harvest_level"] = max(self._harvest_level)
        info["stats"]["achieved/team_time_alive"] = self._time_alive
        info["stats"]["achieved/unique_events"] = self._curr_unique_count
        info["curriculum"] = self._curriculum

        result, achieved, performed, _ = get_episode_result(self.env.realm, self.agent_id)
        for key, val in list(achieved.items()) + list(performed.items()):
            info["stats"][key] = float(val)

        # Fill in the "TeamResult"
        result.total_score = self._curr_unique_count
        result.time_alive = self._time_alive
        result.earned_gold = achieved["achieved/earned_gold"]
        result.completed_task_count = self._task_completed
        result.damage_received = self._damage_received
        result.damage_inflicted = self._damage_inflicted
        result.ration_consumed = self._ration_consumed
        result.potion_consumed = self._potion_consumed
        result.melee_level = self._melee_level
        result.range_level = self._range_level
        result.mage_level = self._mage_level
        result.fishing_level = self._fishing_level
        result.herbalism_level = self._herbalism_level
        result.prospecting_level = self._prospecting_level
        result.carving_level = self._carving_level
        result.alchemy_level = self._alchemy_level

        info["team_results"] = (self.agent_id, result)

        return reward, done, info

# Event processing utilities for Neural MMO.

INFO_KEY_TO_EVENT_CODE = {
    "event/" + evt.lower(): val
    for evt, val in EventCode.__dict__.items()
    if isinstance(val, int)
}

# convert the numbers into binary (performed or not) for the key events
KEY_EVENT = [
    "eat_food",
    "drink_water",
    "score_hit",
    "player_kill",
    "consume_item",
    "harvest_item",
    "list_item",
    "buy_item",
]

ITEM_TYPE = {
    "armor": [item.ITEM_TYPE_ID for item in [Item.Hat, Item.Top, Item.Bottom]],
    "weapon": [item.ITEM_TYPE_ID for item in [Item.Spear, Item.Bow, Item.Wand]],
    "tool": [item.ITEM_TYPE_ID for item in \
             [Item.Axe, Item.Gloves, Item.Rod, Item.Pickaxe, Item.Chisel]],
    "ammo": [item.ITEM_TYPE_ID for item in [Item.Runes, Item.Arrow, Item.Whetstone]],
    "consumable": [item.ITEM_TYPE_ID for item in [Item.Potion, Item.Ration]],
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

    # record true or false for each event
    performed = {}
    for evt in KEY_EVENT:
        key = "event/" + evt
        performed[key] = event_cnt[key] > 0

    # check if tools, weapons, ammos, ammos were equipped
    for item_type, item_ids in ITEM_TYPE.items():
        if item_type == "consumable":
            continue
        key = "event/equip_" + item_type
        idx = (log[:, attr_to_col["event"]] == EventCode.EQUIP_ITEM) & \
              np.in1d(log[:, attr_to_col["item_type"]], item_ids)
        performed[key] = sum(idx) > 0

    # check if weapon was harvested
    key = "event/harvest_weapon"
    idx = (log[:, attr_to_col["event"]] == EventCode.HARVEST_ITEM) & \
          np.in1d(log[:, attr_to_col["item_type"]], ITEM_TYPE["weapon"])
    performed[key] = sum(idx) > 0

    # record important achievements
    achieved = {}

    # get progress to center
    idx = log[:, attr_to_col["event"]] == EventCode.GO_FARTHEST
    achieved["achieved/max_progress_to_center"] = \
        int(max(log[idx, attr_to_col["distance"]])) if sum(idx) > 0 else 0

    # get earned gold
    idx = log[:, attr_to_col["event"]] == EventCode.EARN_GOLD
    achieved["achieved/earned_gold"] = int(sum(log[idx, attr_to_col["gold"]]))

    # get max damage
    idx = log[:, attr_to_col["event"]] == EventCode.SCORE_HIT
    achieved["achieved/max_damage"] = int(max(log[idx, attr_to_col["damage"]])) if sum(idx) > 0 else 0

    # get max possessed item levels: from harvesting, looting, buying
    idx = np.in1d(log[:, attr_to_col["event"]],
                  [EventCode.HARVEST_ITEM, EventCode.LOOT_ITEM, EventCode.BUY_ITEM])
    if sum(idx) > 0:
      for item_type, item_ids in ITEM_TYPE.items():
          idx_item = np.in1d(log[idx, attr_to_col["item_type"]], item_ids)
          achieved["achieved/max_" + item_type + "_level"] = \
            int(max(log[idx][idx_item, attr_to_col["level"]])) if sum(idx_item) > 0 else 1  # min level = 1

    # other notable achievements
    idx = (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL)
    achieved["achieved/agent_kill_count"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] > 0)))
    achieved["achieved/npc_kill_count"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] < 0)))

    return achieved, performed, event_cnt

def extract_unique_event(log, attr_to_col):
    if len(log) == 0:  # no event logs
        return set()

    # mask some columns to make the event redundant
    cols_to_ignore = {
        EventCode.GO_FARTHEST: ["distance"],
        EventCode.SCORE_HIT: ["damage"],
        # treat each (item, level) differently
        EventCode.CONSUME_ITEM: ["quantity"],
        # but, count each (item, level) only once
        EventCode.HARVEST_ITEM: ["quantity"],
        EventCode.EQUIP_ITEM: ["quantity"],
        EventCode.LOOT_ITEM: ["quantity"],
        EventCode.LIST_ITEM: ["quantity", "price"],
        EventCode.BUY_ITEM: ["quantity", "price"],
    }

    for code, attrs in cols_to_ignore.items():
        idx = log[:, attr_to_col["event"]] == code
        for attr in attrs:
            log[idx, attr_to_col[attr]] = 0

    # make every EARN_GOLD events unique, from looting and selling
    idx = log[:, attr_to_col["event"]] == EventCode.EARN_GOLD
    log[idx, attr_to_col["number"]] = log[
        idx, attr_to_col["tick"]
    ].copy()  # this is a hack

    # return unique events after masking
    return set(tuple(row) for row in log[:, attr_to_col["event"]:])

def calculate_entropy(sequence):
    frequencies = Counter(sequence)
    total_elements = len(sequence)
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_elements
        entropy -= probability * math.log2(probability)
    return entropy
