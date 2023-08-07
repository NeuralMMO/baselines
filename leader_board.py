from typing import Optional
from dataclasses import dataclass

import numpy as np

from nmmo.core.realm import Realm
from nmmo.lib.log import EventCode


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
