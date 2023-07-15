from typing import List
from types import ModuleType
import re
import sys
import math
import time
import inspect
import multiprocessing as mp
from collections import Counter

import nmmo
import numpy as np
from nmmo.lib.material import Harvestable
import nmmo.task
from nmmo.task import constraint as c
from nmmo.task import task_spec as ts

# required to run check_task_spec
# pylint: disable=wildcard-import,unused-import,unused-wildcard-import
from nmmo.task.group import Group
from nmmo.task.game_state import GameState
from nmmo.task.base_predicates import *


######################################################################
# some sample phenotypes:


def entropy(task):
  """A sample metric for the behaviour space, computing entropy to count repeated strings"""
  words = re.split(r"[ _\(\):]+", task)
  words = [word for word in words if word]
  word_freq = Counter(words)
  # Calculate the probability of each word
  total_words = len(words)
  word_prob = [count / total_words for count in word_freq.values()]
  # Calculate the Shannon entropy
  ent = -sum(prob * math.log2(prob) for prob in word_prob)

  # rescale to behaviour space
  return min(math.ceil(ent), 10)


def calculate_length(task):
  """Scaling metrics between two values. It is very important for the selected phenotypes
  to be able to have values and easily move across the defined behaviour space.
  in this case 0-10  scale # of characters in task (100-9000) to behaviour space 0-10
"""
  min_val = 100
  max_val = 9000
  new_min = 0
  new_max = 10

  # Scale the value
  scaled_value = ((len(task) - min_val) / (max_val - min_val)) * (
      new_max - new_min
  ) + new_min

  return math.ceil(scaled_value)


######################################################################
# nmmo task-related helper functions

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
  TARGET = [
      "left_team",
      "right_team",
      "left_team_leader",
      "right_team_leader",
      "my_team_leader",
  ]
  EVENT_NAME = c.event_names
  SKILLS = c.combat_skills + c.harvest_skills
  COMBAT_STYLE = c.combat_skills
  ALL_ITEM = c.armour + c.weapons + c.tools + c.ammunition + c.consumables
  sample_dict = {
      "event": lambda: np.random.choice(EVENT_NAME),
      "N": lambda: round(1 + np.random.gamma(1, 3)),
      "tile_type": lambda: np.random.choice(list(Harvestable)),
      "num_tick": lambda: round(np.random.gamma(10, 20)),
      "target": lambda: np.random.choice(TARGET),
      "row": lambda: round(80 + np.random.randn() * 15),
      "col": lambda: round(80 + np.random.randn() * 15),
      "dist": lambda: round(np.random.rand() * 10),
      "num_agent": lambda: 1,
      "level": lambda: min(round(1 + np.random.gamma(1, 3)), 10),
      "skill": lambda: np.random.choice(SKILLS),
      "combat_style": lambda: np.random.choice(COMBAT_STYLE),
      "agent_type": lambda: np.random.choice(["npc", "player"]),
      "amount": lambda: round(1 + np.random.gamma(3, 3)),
      "space": lambda: round(2 + np.random.rand() * 6),
      "item": lambda: np.random.choice(ALL_ITEM),
      "quantity": lambda: round(1 + np.random.gamma(1, 1)),
  }
  if key in sample_dict:
    return sample_dict[key]()

  # TODO: prompting would be helpful below
  hint_dict = {
      "int": lambda: round(1 + np.random.gamma(1, 3)),
      "float": lambda: np.random.rand(),
  }
  if type_hint in hint_dict:
    return hint_dict[type_hint]()

  return 1


TIME_OUT = 15  # sec

def is_task_spec_valid(spec_list: List[ts.TaskSpec]):
  # pylint: disable=bad-builtin,bare-except,inconsistent-return-statements
  teams = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9], 4: [10, 11]}
  config = nmmo.config.Default()
  env = nmmo.Env(config)
  num_success = 0
  for single_spec in spec_list:
    # pylint: disable=cell-var-from-loop
    test_task = ts.make_task_from_spec(teams, [single_spec])
    env.reset(make_task_fn=lambda: test_task)

    def run_env():
      for _ in range(3):
        env.step({})
      sys.exit(0)  # success

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

  return num_success > 0  # at least 1 spec runs


def generate_task_spec(result_str, fn_name, num_sample=3):
  # pylint: disable=bare-except,exec-used,bad-builtin
  task_spec = []
  task_fn_str = extract_task_fn(result_str, fn_name)
  import_str = (
      "from nmmo.task.game_state import GameState\n"
      + "from nmmo.task.group import Group\n"
      + "from nmmo.task.base_predicates import *\n\n"
  )

  locals_dict = {}
  try:
    # NOTE: this is a security vulenerability
    # TODO: make this secure
    exec(import_str + task_fn_str, globals(), locals_dict)
  except:
    # return empty task spec for invalid function
    print("Invalid python function generated ...")
    return task_spec
  task_fn = locals_dict[fn_name]
  fn_params = inspect.signature(task_fn).parameters

  included_kwargs = set()
  for _ in range(num_sample):
    task_fn_kwargs = {}
    for key, param in fn_params.items():
      if key in ["gs", "subject"]:
        continue
      type_hint = param.annotation.__name__
      task_fn_kwargs[key] = sample_parameter(key, type_hint)
    args_vals = tuple(task_fn_kwargs.values())
    if args_vals not in included_kwargs:
      task_spec.append(ts.TaskSpec(eval_fn=task_fn,
                                   eval_fn_kwargs=task_fn_kwargs))
      included_kwargs.add(args_vals)

  return task_spec

def task_spec_to_str(task_spec: List[ts.TaskSpec]):
  # extract task_fn source code from task_spec
  extracted_fn_list = set()
  task_fn_src = []
  for single_spec in task_spec:
    fn_name = single_spec.eval_fn.__name__
    if fn_name not in extracted_fn_list:
      task_fn_src.append(inspect.getsource(single_spec.eval_fn))
      extracted_fn_list.add(fn_name)
  return "\n".join(task_fn_src)

# get the list of pre-built task functions
def is_function_type(obj):
  return inspect.isfunction(obj) and not inspect.isbuiltin(obj)

def extract_module_fn(module: ModuleType):
  fn_dict = {}
  for name, fn in module.__dict__.items():
    if is_function_type(fn) and not name.startswith('_'):
      fn_dict[name] = fn
  return fn_dict

PREBUILT_TASK_FN = extract_module_fn(nmmo.task.base_predicates)

# used in OpenELMTaskGenerator: see self.config.env.impr = import_str["short_import"]
import_str = {"short_import": """from predicates import TickGE,StayAlive,AllDead,EatFood,DrinkWater,CanSeeTile,CanSeeAgent,OccupyTile
from predicates import DistanceTraveled,AllMembersWithinRange,ScoreHit,ScoreKill,AttainSkill,InventorySpaceGE,OwnItem
from predicates import EquipItem,FullyArmed,ConsumeItem,GiveItem,DestroyItem,HarvestItem,HoardGold,GiveGold,ListItem,EarnGold,BuyItem,SpendGold,MakeProfit
# Armour
item.Hat, item.Top, item.Bottom, 
# Weapon
item.Sword, item.Bow, item.Wand
# Tool
item.Rod, item.Gloves, item.Pickaxe, item.Chisel, item.Arcane
# Ammunition
item.Scrap, item.Shaving, item.Shard, 
# Consumable
item.Ration, item.Poultice
# Materials
tile_type.Lava, tile_type.Water,tile_type.Grass, tile_type.Scrub, 
tile_type.Forest, tile_type.Stone, tile_type.Slag,  tile_type.Ore
tile_type.Stump, tile_type.Tree, tile_type.Fragment, tile_type.Crystal,
tile_type.Weeds, tile_type.Ocean, tile_type.Fish""",

"long_import":"""
Base Predicates to use in tasks:
TickGE(gs, num_tick):True if the current tick is greater than or equal to the specified num_tick.Is progress counter.
CanSeeTile(gs, subject,tile_type):True if any agent in subject can see a tile of tile_type
StayAlive(gs,subject): True if all subjects are alive.
AllDead(gs, subject):True if all subjects are dead.
OccupyTile(gs,subject,row, col):True if any subject agent is on the desginated tile.
AllMembersWithinRange(gs,subject,dist):True if the max l-inf distance of teammates is less than or equal to dist
CanSeeAgent(gs,subject,target): True if obj_agent is present in the subjects' entities obs.
CanSeeGroup(gs,subject,target): Returns True if subject can see any of target
DistanceTraveled(gs, subject, dist): True if the summed l-inf distance between each agent's current pos and spawn pos is greater than or equal to the specified _dist.
AttainSkill(gs,subject,skill,level,num_agent):True if the number of agents having skill level GE level is greather than or equal to num_agent
CountEvent(gs,subject,event,N): True if the number of events occured in subject corresponding to event >= N
ScoreHit(gs, subject, combat_style, N):True if the number of hits scored in style combat_style >= count
HoardGold(gs, subject, amount): True iff the summed gold of all teammate is greater than or equal to amount.
EarnGold(gs, subject, amount): True if the total amount of gold earned is greater than or equal to amount.
SpendGold(gs, subject, amount): True if the total amount of gold spent is greater than or equal to amount.
MakeProfit(gs, subject, amount) True if the total amount of gold earned-spent is greater than or equal to amount.
InventorySpaceGE(gs, subject, space): True if the inventory space of every subjects is greater than or equal to the space. Otherwise false.
OwnItem(gs, subject, item, level, quantity): True if the number of items owned (_item, >= level) is greater than or equal to quantity.
EquipItem(gs, subject, item, level, num_agent): True if the number of agents that equip the item (_item_type, >=_level) is greater than or equal to _num_agent.
FullyArmed(GameState, subject, combat_style, level, num_agent): True if the number of fully equipped agents is greater than or equal to _num_agent Otherwise false. To determine fully equipped, we look at hat, top, bottom, weapon, ammo, respectively, these are equipped and has level greater than or equal to _level.
ConsumeItem(GameState, subject, item, level, quantity): True if total quantity consumed of item type above level is >= quantity
HarvestItem(GameState, Group, Item, int, int): True if total quantity harvested of item type above level is >= quantity
ListItem(GameState,subject,item,level, quantity): True if total quantity listed of item type above level is >= quantity
BuyItem(GameState, subject, item, level, quantity) :  True if total quantity purchased of item type above level is >= quantity
# Armour
item.Hat, item.Top, item.Bottom, 
# Weapon
item.Sword, item.Bow, item.Wand
# Tool
item.Rod, item.Gloves, item.Pickaxe, item.Chisel, item.Arcane
# Ammunition
item.Scrap, item.Shaving, item.Shard, 
# Consumable
item.Ration, item.Poultice
# Materials
tile_type.Lava, tile_type.Water,tile_type.Grass, tile_type.Scrub, 
tile_type.Forest, tile_type.Stone, tile_type.Slag,  tile_type.Ore
tile_type.Stump, tile_type.Tree, tile_type.Fragment, tile_type.Crystal,
tile_type.Weeds, tile_type.Ocean, tile_type.Fish
"""}
