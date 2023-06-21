import re
import math
from collections import Counter

import nmmo
from nmmo.task.group import Group
from nmmo.task.task_api import make_team_tasks
from nmmo.task.predicate_api import make_predicate

# TODO: automatically get all pre-built functions from nmmo.task.base_predicates
UNIQUE_PREDICATES = [
  "TickGE","StayAlive","AllDead","EatFood","DrinkWater","CanSeeTile","CanSeeAgent",
  "OccupyTile","DistanceTraveled","AllMembersWithinRange","ScoreHit","ScoreKill",
  "AttainSkill","InventorySpaceGE","OwnItem","EquipItem","FullyArmed","ConsumeItem",
  "GiveItem","DestroyItem","HarvestItem","HoardGold","GiveGold","ListItem","EarnGold",
  "BuyItem","SpendGold","MakeProfit"]


######################################################################
# some sample phenotypes:

def entropy(task):
  """A sample metric for the behaviour space, computing entropy to count repeated strings"""
  words = re.split(r'[ _\(\):]+', task)
  words = [word for word in words if word]
  word_freq = Counter(words)
  # Calculate the probability of each word
  total_words = len(words)
  word_prob = [count / total_words for count in word_freq.values()]
  # Calculate the Shannon entropy
  ent = -sum(prob * math.log2(prob) for prob in word_prob)

  # rescale to behaviour space
  return min(math.ceil(ent),10)

def calculate_length(task):
  """Scaling metrics between two values. It is very important for the selected phenotypes
  to be able to have values and easily move across the defined behaviour space.
  in this case 0-10  scale # of characters in task (100-9000) to behaviour space 0-10"""
  min_val = 100
  max_val = 9000
  new_min = 0
  new_max = 10

  # Scale the value
  scaled_value = ((len(task) - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

  return math.ceil(scaled_value)

######################################################################
# nmmo task-related helper functions

def extract_kwargs(function_definition):
  pattern = r'def\s+\w+\((.*?)\)'
  parameter_pattern = r'(\w+)\s*:\s*([^\s,]+)'

  match = re.search(pattern, function_definition)
  if match:
    parameters_string = match.group(1)
    parameters = re.findall(parameter_pattern, parameters_string)
    parameter_dict = dict(parameters) #{name: data_type for name, data_type in parameters}
    return parameter_dict

  return {}

def check_task_spec(spec_list):
  # pylint: disable=bad-builtin,bare-except,inconsistent-return-statements
  teams = {0:[1,2,3], 1:[4,5], 2:[6,7], 3:[8,9], 4:[10,11]}
  config = nmmo.config.Default()
  env = nmmo.Env(config)
  for idx, single_spec in enumerate(spec_list):
    # pylint: disable=cell-var-from-loop
    test_task = make_team_tasks(teams, [single_spec])
    try:
      env.reset(make_task_fn=lambda: test_task)
      for _ in range(3):
        env.step({})
    except:
      print('invalid task spec:', single_spec)
      return False

    if idx > 0 and idx % 50 == 0:
      print(idx, 'task specs checked.')

def str_to_task_spec(task_list):
  # pylint: disable=exec-used
  task_specs = []
  for task in task_list:
    func = {}
    try:
      exec(task, globals(), func)
      # get kwargs
      kwargs = extract_kwargs(task)
      task_specs.append(("agent", task, kwargs))
    except: # pylint: disable=bare-except
      pass
  return task_specs

def task_spec_to_str(task_specs):
  # convert task spec to str code
  str_tasks = []
  for task_spec in task_specs:
    predicate = make_predicate(task_spec[1])
    inst_predicate = predicate(Group(0))
    str_tasks.append(inst_predicate.get_source_code())
  return "\n".join(str_tasks)
