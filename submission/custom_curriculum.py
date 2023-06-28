# pylint: disable=wildcard-import,invalid-name,unused-wildcard-import,unused-argument

# allow custom functions to use pre-built eval functions without prefix
from nmmo.task.base_predicates import *
from nmmo.task.base_predicates import norm as norm_progress

##############################################################################
# define custom evaluation functions
# pylint: disable=redefined-outer-name

def PracticeFormation(gs, subject, dist, num_tick):
  return norm_progress(AllMembersWithinRange(gs, subject, dist) * TickGE(gs, subject, num_tick))

def PracticeInventoryManagement(gs, subject, space, num_tick):
  return norm_progress(InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick))

def PracticeEating(gs, subject):
  """The progress, the max of which is 1, should
        * increase small for each eating
        * increase big for the 1st and 3rd eating
        * reach 1 with 10 eatings
  """
  num_eat = len(subject.event.EAT_FOOD)
  progress = num_eat * 0.06
  if num_eat >= 1:
    progress += .1
  if num_eat >= 3:
    progress += .3
  return norm_progress(progress)

##############################################################################
# define learning task spec, a list of tuple: (reward_to, eval_fn, eval_fn_kwargs)
#   * reward_to: 'agent' or 'team'
#   * eval_fn: one eval function defined above or in nmmo.task.base_predicates
#   * eval_fn_kwargs: parameters to pass into eval_fn

STAY_ALIVE_GOAL = [50, 100, 150, 200, 300, 500]
EVENT_NUMBER_GOAL = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20, 30, 50]

task_spec = []

# explore, eat, drink, attack any agent, harvest any item, level up any skill
#   which can happen frequently
essential_skills = ['GO_FARTHEST', 'EAT_FOOD', 'DRINK_WATER',
                    'SCORE_HIT', 'HARVEST_ITEM', 'LEVEL_UP']
for event_code in essential_skills:
  task_spec += [('agent', CountEvent, {'event': event_code, 'N': cnt})
                for cnt in EVENT_NUMBER_GOAL]

for dist in [1, 3, 5, 10]:
  task_spec += [('team', PracticeFormation, {'dist': dist, 'num_tick': num_tick})
                for num_tick in STAY_ALIVE_GOAL]

for space in [2, 4, 8]:
  task_spec += [('agent', PracticeInventoryManagement, {'space': space, 'num_tick': num_tick})
                for num_tick in STAY_ALIVE_GOAL]

task_spec.append(('agent', PracticeEating, {}))
