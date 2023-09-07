"""Manual test for creating learning curriculum manually"""
# pylint: disable=invalid-name,redefined-outer-name,bad-builtin
# pylint: disable=wildcard-import,unused-wildcard-import
from typing import List

from nmmo.task import constraint as c
from nmmo.task.base_predicates import AttainSkill, CountEvent, EarnGold, TickGE
from nmmo.task.task_spec import TaskSpec

curriculum: List[TaskSpec] = []

# Stay alive as long as possible
curriculum.append(
    TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": 1024})
)

# Perform these 10 times
essential_skills = [
    "EAT_FOOD",
    "DRINK_WATER",
    "SCORE_HIT",
    "PLAYER_KILL",
    "HARVEST_ITEM",
    "EQUIP_ITEM",
    "CONSUME_ITEM",
    "LEVEL_UP",
    "EARN_GOLD",
    "LIST_ITEM",
    "BUY_ITEM",
    "GIVE_ITEM",
    "DESTROY_ITEM",
    "GIVE_GOLD",
]
for event_code in essential_skills:
    curriculum.append(
        TaskSpec(
            eval_fn=CountEvent,
            eval_fn_kwargs={"event": event_code, "N": 10},
        )
    )

# Reach skill level 10
for skill in c.combat_skills + c.harvest_skills:
    curriculum.append(
        TaskSpec(
            eval_fn=AttainSkill,
            eval_fn_kwargs={"skill": skill, "level": 10, "num_agent": 1},
        )
    )

# Earn gold 50
curriculum.append(
    TaskSpec(eval_fn=EarnGold, eval_fn_kwargs={"amount": 50})
)
