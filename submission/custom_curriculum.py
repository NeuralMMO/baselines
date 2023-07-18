# pylint: disable=wildcard-import,invalid-name,unused-wildcard-import,unused-argument

# allow custom functions to use pre-built eval functions without prefix
from nmmo.task.base_predicates import CountEvent, InventorySpaceGE, TickGE, norm
from nmmo.task.task_spec import TaskSpec

##############################################################################
# define custom evaluation functions
# pylint: disable=redefined-outer-name


# NOTE: norm is a helper function to normalize the value to [0, 1]
#    imported from nmmo.task.base_predicates
def PracticeInventoryManagement(gs, subject, space, num_tick):
  return norm(InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick))


def PracticeEating(gs, subject):
  """The progress, the max of which is 1, should
  * increase small for each eating
  * increase big for the 1st and 3rd eating
  * reach 1 with 10 eatings
  """
  num_eat = len(subject.event.EAT_FOOD)
  progress = num_eat * 0.06
  if num_eat >= 1:
    progress += 0.1
  if num_eat >= 3:
    progress += 0.3
  return norm(progress)


##############################################################################
# Use TaskSpec class to define each training task
# See curriculum/manual_curriculum.py for detailed examples based on pre-built eval fns

STAY_ALIVE_GOAL = [50, 100, 150, 200, 300, 500]
EVENT_NUMBER_GOAL = [3, 4, 5, 7, 9, 12, 15, 20, 30, 50]

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
      TaskSpec(
          eval_fn=CountEvent,
          eval_fn_kwargs={"event": event_code, "N": cnt},
          sampling_weight=3,
      )
      for cnt in EVENT_NUMBER_GOAL
  ]

for space in [2, 4, 8]:
  task_spec += [
      TaskSpec(
          eval_fn=PracticeInventoryManagement,
          eval_fn_kwargs={"space": space, "num_tick": num_tick},
      )
      for num_tick in STAY_ALIVE_GOAL
  ]

task_spec.append(TaskSpec(eval_fn=PracticeEating,
                 eval_fn_kwargs={}, sampling_weight=5))
