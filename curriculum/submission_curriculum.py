from nmmo.task.base_predicates import *

from curriculum.submission_eval_fn import *

# Participant can make their own task generator to generate task spec
#  task_spec = CustomTaskGenerator.generate_curriculum()

STAY_ALIVE_GOAL = [50, 100, 150, 200, 300, 500]
EVENT_NUMBER_GOAL = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20, 30, 50]

task_spec = []

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
  task_spec += [
      ("agent", CountEvent, {"event": event_code, "N": cnt})
      for cnt in EVENT_NUMBER_GOAL
  ]

for dist in [1, 3, 5, 10]:
  task_spec += [
      ("team", PracticeFormation, {"dist": dist, "num_tick": num_tick})
      for num_tick in STAY_ALIVE_GOAL
  ]

for space in [2, 4, 8]:
  task_spec += [
      ("agent", PracticeInventoryManagement, {
       "space": space, "num_tick": num_tick})
      for num_tick in STAY_ALIVE_GOAL
  ]

task_spec.append(("agent", PracticeEating, {}))

# print(task_spec)
