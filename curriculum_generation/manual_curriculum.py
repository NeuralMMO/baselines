"""Manual test for creating learning curriculum manually"""
# pylint: disable=invalid-name,redefined-outer-name,bad-builtin
# pylint: disable=wildcard-import,unused-wildcard-import
from typing import List

import nmmo
import nmmo.lib.material as m
from nmmo.task import constraint as c
from nmmo.task.base_predicates import (
    AttainSkill,
    BuyItem,
    CanSeeAgent,
    CanSeeGroup,
    CanSeeTile,
    ConsumeItem,
    CountEvent,
    DistanceTraveled,
    EarnGold,
    EquipItem,
    HarvestItem,
    HoardGold,
    InventorySpaceGE,
    ListItem,
    MakeProfit,
    OccupyTile,
    OwnItem,
    ScoreHit,
    SpendGold,
    TickGE,
)
from nmmo.task.task_spec import TaskSpec, make_task_from_spec

EVENT_NUMBER_GOAL = [3, 4, 5, 7, 9, 12, 15, 20, 30, 50]
INFREQUENT_GOAL = list(range(1, 10))
STAY_ALIVE_GOAL = [50, 100, 150, 200, 300, 500]
LEVEL_GOAL = list(range(1, 10))  # TODO: get config
AGENT_NUM_GOAL = [1, 2, 3, 4, 5]  # competition team size: 8
ITEM_NUM_GOAL = AGENT_NUM_GOAL
SKILLS = c.combat_skills + c.harvest_skills
COMBAT_STYLE = c.combat_skills
ALL_ITEM = c.armour + c.weapons + c.tools + c.ammunition + c.consumables
EQUIP_ITEM = c.armour + c.weapons + c.tools + c.ammunition
HARVEST_ITEM = c.weapons + c.ammunition + c.consumables

task_spec: List[TaskSpec] = []

# explore, eat, drink, attack any agent, harvest any item, level up any skill
#   which can happen frequently
essential_skills = [
    "GO_FARTHEST",
    "EAT_FOOD",
    "DRINK_WATER",
    "SCORE_HIT",
    "HARVEST_ITEM",
    "LEVEL_UP",
]
for event_code in essential_skills:
  for cnt in EVENT_NUMBER_GOAL:
    task_spec.append(
        TaskSpec(
            eval_fn=CountEvent,
            eval_fn_kwargs={"event": event_code, "N": cnt},
            sampling_weight=30,
        )
    )

# item/market skills, which happen less frequently or should not do too much
item_skills = [
    "CONSUME_ITEM",
    "GIVE_ITEM",
    "DESTROY_ITEM",
    "EQUIP_ITEM",
    "GIVE_GOLD",
    "LIST_ITEM",
    "EARN_GOLD",
    "BUY_ITEM",
]
for event_code in item_skills:
  task_spec += [
      TaskSpec(eval_fn=CountEvent, eval_fn_kwargs={
               "event": event_code, "N": cnt})
      for cnt in INFREQUENT_GOAL
  ]  # less than 10

# find resource tiles
for resource in m.Harvestable:
  task_spec.append(
      TaskSpec(
          eval_fn=CanSeeTile,
          eval_fn_kwargs={"tile_type": resource},
          sampling_weight=10,
      )
  )

# stay alive ... like ... for 300 ticks
# i.e., getting incremental reward for each tick alive as an individual or a team
for num_tick in STAY_ALIVE_GOAL:
  task_spec.append(
      TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": num_tick}))

# occupy the center tile, assuming the Medium map size
# TODO: it'd be better to have some intermediate targets toward the center
task_spec.append(TaskSpec(eval_fn=OccupyTile,
                 eval_fn_kwargs={"row": 80, "col": 80}))

# find the other team leader
for target in ["left_team_leader", "right_team_leader"]:
  task_spec.append(TaskSpec(eval_fn=CanSeeAgent,
                   eval_fn_kwargs={"target": target}))

# find the other team (any agent)
for target in ["left_team", "right_team"]:
  task_spec.append(TaskSpec(eval_fn=CanSeeGroup,
                   eval_fn_kwargs={"target": target}))

# explore the map -- sum the l-inf distance traveled by all subjects
for dist in [10, 20, 30, 50, 100]:  # each agent
  task_spec.append(TaskSpec(eval_fn=DistanceTraveled,
                   eval_fn_kwargs={"dist": dist}))

# level up a skill
for skill in SKILLS:
  for level in LEVEL_GOAL[1:]:
    # since this is an agent task, num_agent must be 1
    task_spec.append(
        TaskSpec(
            eval_fn=AttainSkill,
            eval_fn_kwargs={"skill": skill, "level": level, "num_agent": 1},
            sampling_weight=10 * (5 - level) if level < 5 else 1,
        )
    )

# practice specific combat style
for style in COMBAT_STYLE:
  for cnt in EVENT_NUMBER_GOAL:
    task_spec.append(
        TaskSpec(
            eval_fn=ScoreHit,
            eval_fn_kwargs={"combat_style": style, "N": cnt},
            sampling_weight=5,
        )
    )

# hoarding gold -- evaluated on the current gold
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(
      TaskSpec(
          eval_fn=HoardGold, eval_fn_kwargs={"amount": amount}, sampling_weight=3
      )
  )

# earning gold -- evaluated on the total gold earned by selling items
# does NOT include looted gold
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(
      TaskSpec(eval_fn=EarnGold, eval_fn_kwargs={
               "amount": amount}, sampling_weight=3)
  )

# spending gold, by buying items
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(
      TaskSpec(
          eval_fn=SpendGold, eval_fn_kwargs={"amount": amount}, sampling_weight=3
      )
  )

# making profits by trading -- only buying and selling are counted
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(
      TaskSpec(
          eval_fn=MakeProfit, eval_fn_kwargs={"amount": amount}, sampling_weight=3
      )
  )


# managing inventory space
def PracticeInventoryManagement(gs, subject, space, num_tick):
  return InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick)


for space in [2, 4, 8]:
  task_spec += [
      TaskSpec(
          eval_fn=PracticeInventoryManagement,
          eval_fn_kwargs={"space": space, "num_tick": num_tick},
      )
      for num_tick in STAY_ALIVE_GOAL
  ]

# own item, evaluated on the current inventory
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1:  # heuristic prune
        task_spec.append(
            TaskSpec(
                eval_fn=OwnItem,
                eval_fn_kwargs={
                    "item": item,
                    "level": level,
                    "quantity": quantity,
                },
                sampling_weight=4 - level if level < 4 else 1,
            )
        )

# equip item, evaluated on the current inventory and equipment status
for item in EQUIP_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    task_spec.append(
        TaskSpec(
            eval_fn=EquipItem,
            eval_fn_kwargs={"item": item, "level": level, "num_agent": 1},
            sampling_weight=4 - level if level < 4 else 1,
        )
    )

# consume items (ration, potion), evaluated based on the event log
for item in c.consumables:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1:  # heuristic prune
        task_spec.append(
            TaskSpec(
                eval_fn=ConsumeItem,
                eval_fn_kwargs={
                    "item": item,
                    "level": level,
                    "quantity": quantity,
                },
                sampling_weight=4 - level if level < 4 else 1,
            )
        )

# harvest items, evaluated based on the event log
for item in HARVEST_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1:  # heuristic prune
        task_spec.append(
            TaskSpec(
                eval_fn=HarvestItem,
                eval_fn_kwargs={
                    "item": item,
                    "level": level,
                    "quantity": quantity,
                },
                sampling_weight=4 - level if level < 4 else 1,
            )
        )

# list items, evaluated based on the event log
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1:  # heuristic prune
        task_spec.append(
            TaskSpec(
                eval_fn=ListItem,
                eval_fn_kwargs={
                    "item": item,
                    "level": level,
                    "quantity": quantity,
                },
                sampling_weight=4 - level if level < 4 else 1,
            )
        )

# buy items, evaluated based on the event log
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1:  # heuristic prune
        task_spec.append(
            TaskSpec(
                eval_fn=BuyItem,
                eval_fn_kwargs={
                    "item": item,
                    "level": level,
                    "quantity": quantity,
                },
                sampling_weight=4 - level if level < 4 else 1,
            )
        )


if __name__ == "__main__":
  # pylint: disable=bare-except
  import multiprocessing as mp
  from contextlib import contextmanager

  import dill
  import numpy as np
  import psutil

  @contextmanager
  def create_pool(num_proc):
    pool = mp.Pool(processes=num_proc)
    yield pool
    pool.close()
    pool.join()

  def check_task_spec(spec_list):
    teams = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9], 4: [10, 11]}
    config = nmmo.config.Default()
    env = nmmo.Env(config)
    for idx, single_spec in enumerate(spec_list):
      # pylint: disable=cell-var-from-loop
      test_task = make_task_from_spec(teams, [single_spec])
      try:
        env.reset(make_task_fn=lambda: test_task)
        for _ in range(3):
          env.step({})
      except:
        print("invalid task spec:", single_spec)

      if idx > 0 and idx % 50 == 0:
        print(idx, "task specs checked.")

  # 1535 task specs: divide the specs into chunks
  num_cores = psutil.cpu_count(logical=False)
  spec_chunks = np.array_split(task_spec, num_cores)
  with create_pool(num_cores) as pool:
    pool.map(check_task_spec, spec_chunks)

  # test if the task spec is pickalable
  with open("manual_curriculum.pkl", "wb") as f:
    dill.dump(task_spec, f)