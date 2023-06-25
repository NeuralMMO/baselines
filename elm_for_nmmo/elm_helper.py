import re
import sys
import math
import time
import inspect
import multiprocessing as mp
from collections import Counter

import numpy as np

import nmmo
from nmmo.lib.material import Harvestable
from nmmo.task import constraint as c
from nmmo.task.task_api import make_team_tasks

# required to run check_task_spec
# pylint: disable=wildcard-import,unused-import,unused-wildcard-import
from nmmo.task.group import Group
from nmmo.task.game_state import GameState
from nmmo.task.base_predicates import *


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

def is_camel_case(input_string):
  if input_string == input_string.lower(): # all lower case
    return False
  # Check if the string matches the camel case pattern
  pattern = r"^(?:[A-Z]{2}|[A-Z][a-z0-9]+|[a-z][a-z0-9]*)+$"
  return re.match(pattern, input_string) is not None

PREBUILT_TASK_FN = [fn for fn in dir(nmmo.task.base_predicates) if is_camel_case(fn)]

# extract training task function from the ELM result
def extract_task_fn(result_str, fn_name):
  split = result_str.split("\n")
  fn_str = []
  for line in split[::-1]:
    if line.startswith(f"def {fn_name}("):
      fn_str.append(line)
      break
    fn_str.append(line)
  return "\n".join(fn_str[::-1])

def sample_parameter(key, type_hint):
  # pylint: disable=invalid-name,unnecessary-lambda
  # try to return helpful values
  TARGET = ['left_team', 'right_team',
            'left_team_leader', 'right_team_leader', 'my_team_leader']
  EVENT_NAME = c.event_names
  SKILLS = c.combat_skills + c.harvest_skills
  COMBAT_STYLE = c.combat_skills
  ALL_ITEM = c.armour + c.weapons + c.tools + c.ammunition + c.consumables
  sample_dict = {
    'event': lambda: np.random.choice(EVENT_NAME),
    'N': lambda: round(1+np.random.gamma(1,3)),
    'tile_type': lambda: np.random.choice(list(Harvestable)),
    'num_tick': lambda: round(np.random.gamma(10,20)),
    'target': lambda: np.random.choice(TARGET),
    'row': lambda: round(80+np.random.randn()*15),
    'col': lambda: round(80+np.random.randn()*15),
    'dist': lambda: round(np.random.rand()*10),
    'num_agent': lambda: 1,
    'level': lambda: min(round(1+np.random.gamma(1,3)),10),
    'skill': lambda: np.random.choice(SKILLS),
    'combat_style': lambda: np.random.choice(COMBAT_STYLE),
    'agent_type': lambda: np.random.choice(['npc','player']),
    'amount': lambda: round(1+np.random.gamma(3,3)),
    'space': lambda: round(2+np.random.rand()*6),
    'item': lambda: np.random.choice(ALL_ITEM),
    'quantity': lambda: round(1+np.random.gamma(1,1)),
  }
  if key in sample_dict:
    return sample_dict[key]()

  # TODO: prompting would be helpful below
  hint_dict = {
    'int': lambda: round(1+np.random.gamma(1,3)),
    'float': lambda: np.random.rand(),
  }
  if type_hint in hint_dict:
    return hint_dict[type_hint]()

  return 1

TIME_OUT = 15 # sec
def is_task_spec_valid(spec_list):
  # pylint: disable=bad-builtin,bare-except,inconsistent-return-statements
  teams = {0:[1,2,3], 1:[4,5], 2:[6,7], 3:[8,9], 4:[10,11]}
  config = nmmo.config.Default()
  env = nmmo.Env(config)
  num_success = 0
  for single_spec in spec_list:
    # pylint: disable=cell-var-from-loop
    test_task = make_team_tasks(teams, [single_spec])
    env.reset(make_task_fn=lambda: test_task)
    def run_env():
      for _ in range(3):
        env.step({})
      sys.exit(0) # success

    # sometimes the task fn has a long loop, so we need to time out
    proc = mp.Process(target=run_env)
    proc.start()
    start_time = time.time()
    while proc.is_alive():
      elapsed_time = time.time() - start_time
      if elapsed_time > TIME_OUT:
        print("NMMO task timed out")
        proc.terminate()
        break
      time.sleep(0.1)

    if proc.exitcode == 0:
      num_success += 1

  return num_success > 0 # at least 1 spec runs

def generate_task_spec(result_str, fn_name, num_sample=3):
  # pylint: disable=bare-except,exec-used,bad-builtin
  task_spec = []
  task_fn_str = extract_task_fn(result_str, fn_name)
  import_str = "from nmmo.task.game_state import GameState\n" +\
                "from nmmo.task.group import Group\n" +\
                "from nmmo.task.base_predicates import *\n\n"

  locals_dict = {}
  try:
    # NOTE: this is a security vulenerability
    # TODO: make this secure
    exec(import_str + task_fn_str, globals(), locals_dict)
  except:
    # return empty task spec for invalid function
    print('Invalid python function generated ...')
    return task_spec
  task_fn = locals_dict[fn_name]
  fn_params = inspect.signature(task_fn).parameters

  included_kwargs = set()
  for _ in range(num_sample):
    eval_kwargs = {}
    for key, param in fn_params.items():
      if key in ['gs', 'subject']:
        continue
      type_hint = param.annotation.__name__
      eval_kwargs[key] = sample_parameter(key, type_hint)
    args_vals = tuple(eval_kwargs.values())
    if args_vals not in included_kwargs:
      task_spec.append(('agent', task_fn, eval_kwargs))
      included_kwargs.add(args_vals)

  return task_spec

def task_spec_to_str(task_specs):
  # extract task_fn source code from task_spec
  extracted_fn_list = set()
  task_fn_src = []
  for task_spec in task_specs:
    fn_name = task_spec[1].__name__
    if fn_name not in extracted_fn_list:
      task_fn_src.append(inspect.getsource(task_spec[1]))
      extracted_fn_list.add(fn_name)
  return "\n".join(task_fn_src)
