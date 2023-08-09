import inspect
import math
import multiprocessing as mp
import re
import sys
import ast
import time
import random
from collections import Counter
from types import ModuleType
from typing import List, Optional, Dict, Union

from dataclasses import dataclass, field

import numpy as np

import nmmo
from nmmo.lib.material import Harvestable
from nmmo.task import constraint as c, task_spec as ts
import nmmo.task.base_predicates
from nmmo.task.base_predicates import *

from openelm import ELM
from openelm.environments.base import Genotype, Phenotype
from openelm.configs import EnvConfig
from openelm.environments import BaseEnvironment, Genotype
from openelm.mutation_model import MutationModel
from openelm.configs import ELMConfig, MAPElitesConfig, PromptModelConfig

from curriculum_generation.task_sampler import LearnableTaskSampler


# used in OpenELMTaskGenerator: see self.config.env.impr = import_str["short_import"]
import_str = {
    "short_import": """from predicates import TickGE,StayAlive,AllDead,EatFood,DrinkWater,CanSeeTile,CanSeeAgent,OccupyTile
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
    "long_import": """
Base Predicates to use in tasks:
TickGE(gs, num_tick):True if the current tick is greater than or equal to the specified num_tick.Is progress counter.
CanSeeTile(gs, subject,tile_type):True if any agent in subject can see a tile of tile_typegg
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
""",
}


def extract_task_fn(result_str, fn_name):
    """
    Extracts the source code of a function from a given string.

    Args:
        result_str: The string from which the function is to be extracted.
        fn_name: The name of the function to be extracted.

    Returns:
        The source code of the function as a string.
    """
    split = result_str.split("\n")
    fn_str = []
    for line in split[::-1]:
      if line.startswith(f"def {fn_name}("):
        fn_str.append(line)
        break
      fn_str.append(line)
    return "\n".join(fn_str[::-1])

def sample_parameter(key, type_hint):
    """
    Generates sample parameter values based on the key and type_hint provided.

    Args:
        key: The key for which a sample value is to be generated.
        type_hint: The type of the value to be generated.

    Returns:
        A sample value for the given key and type_hint.
    """
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

    # Define a dictionary to generate sample values for each key
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

    # Define a dictionary to generate sample values for each type_hint
    hint_dict = {
        "int": lambda: round(1 + np.random.gamma(1, 3)),
        "float": lambda: np.random.rand(),
    }

    # Use the sample_dict if the key is in it, otherwise use the hint_dict. If the type_hint is not in hint_dict, return 1
    return sample_dict.get(key, hint_dict.get(type_hint, lambda: 1))()


def is_task_spec_valid(spec_list: List[ts.TaskSpec], timeout=15) -> bool:
    """
    This function validates a list of task specifications by running each task in a
    separate process for a set number of steps or until a timeout.

    If at least one task from the list runs successfully, the function will return True.
    If all tasks fail or time out, the function will return False.

    Args:
        spec_list (List[ts.TaskSpec]): List of task specifications to validate.

    Returns:
        bool: True if at least one task ran successfully, False otherwise.
    """
    # Predefined teams of agents for task execution.
    teams = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9], 4: [10, 11]}

    # Create an instance of the environment using default configuration.
    config = nmmo.config.Default()
    env = nmmo.Env(config)

    num_success = 0  # Counter for successful task runs.

    for single_spec in spec_list:
        # Generate a task from the task specification.
        test_task = ts.make_task_from_spec(teams, [single_spec])
        # Reset the environment with the new task.
        env.reset(make_task_fn=lambda: test_task)

        # Function to run environment steps in a separate process.
        def run_env():
            for _ in range(3):
                env.step({})
            sys.exit(0)  # Exit to signify a successful run.

        # Start the process to run the task steps.
        proc = mp.Process(target=run_env)
        proc.start()

        # Monitor the process, terminate it if it runs too long.
        start_time = time.time()
        while proc.is_alive():
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("NMMO task timed out")
                proc.terminate()
                break
            time.sleep(0.1)  # Brief pause to reduce CPU usage.

        # Increment the counter if the task ran successfully.
        if proc.exitcode == 0:
            num_success += 1

    # Return True if at least one task ran successfully.
    return num_success > 0

def generate_task_spec(result_str, fn_name, num_sample=3):
  """
  Generates a list of TaskSpec objects from the task function string provided during the class instantiation.
  Each TaskSpec is an instantiation of the task function with sampled parameters.

  Args:
      program_str: The string representation of the task function.
      fn_name: The name of the task function.
      num_sample: The number of TaskSpecs to generate. Defaults to None, which will generate a TaskSpec for each valid
      function parameter set.

  Returns:
      A list of valid TaskSpec objects. If the task function string is invalid or no valid TaskSpecs can be generated,
      an empty list is returned.
  """
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
      task_spec.append(
          ts.TaskSpec(eval_fn=task_fn, eval_fn_kwargs=task_fn_kwargs)
      )
      included_kwargs.add(args_vals)

  return task_spec


class NMMOTaskFn(Genotype):
    """A task in the NMMO environment."""

    def __init__(self, program_str: str, fn_name: str, module: ModuleType):
        """
        Initialize the NMMO task function.

        Args:
            program_str: The string representation of the task function.
            fn_name: The name of the task function.
            module: The module where task function predicates are defined.
        """
        self._fitness = -np.inf
        self._fn_name = fn_name
        self.program_str = extract_task_fn(program_str, self._fn_name)
        self.valid = is_task_spec_valid(
            generate_task_spec(program_str, self._fn_name)
        )

        self.PREBUILT_TASK_FN = {
            name: fn
            for name, fn in module.__dict__.items()
            if inspect.isfunction(fn)
            and not inspect.isbuiltin(fn)
            and not name.startswith("_")
        }

        self.morphology = {}
        if self.valid:
            code_only = re.sub(r" +#.*\n", "", self.program_str)  # Removes comments.
            self.morphology["predicates"] = self._count_predicates(code_only)
            self.morphology["length"] = calculate_length(code_only)
            self.morphology["lines"] = code_only.count("\n")

    def _count_predicates(self, task_str: str) -> int:
        """
        Counts the number of used predicates in the task.

        Args:
            task_str: The string representation of the task.

        Returns:
            The number of used predicates in the task.
        """
        called_fns = [
            node.func.id
            for node in ast.walk(ast.parse(task_str))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        used_pred_fn = {
            fn_name for fn_name in called_fns if fn_name in self.PREBUILT_TASK_FN
        }
        return len(used_pred_fn)

    def evaluate(self) -> float:
        """
        Computes a fitness score for the task based on the task length.

        Returns:
            The fitness score for the task.
        """
        self._fitness = len(self.program_str) / 10
        return self._fitness

    def generate_task_spec(self, num_sample=None) -> List:
        """
        Generates a list of TaskSpec objects from the task function string provided during the class instantiation.
        Each TaskSpec is an instantiation of the task function with sampled parameters.

        Args:
            num_sample: The number of TaskSpecs to generate. Defaults to None, which will generate a TaskSpec for each valid
            function parameter set.

        Returns:
            A list of valid TaskSpec objects. If the task function string is invalid or no valid TaskSpecs can be generated,
            an empty list is returned.
        """
        if not self.valid:
            return []
        task_spec = []
        for single_spec in generate_task_spec(
            self.program_str, self._fn_name, num_sample
        ):
            if is_task_spec_valid([single_spec]):
                task_spec.append(single_spec)
        return task_spec

    @property
    def fitness(self) -> Optional[float]:
        """
        The fitness of the task.

        Returns:
            The fitness score of the task.
        """
        return self._fitness

    def to_phenotype(self) -> Optional[Phenotype]:
        if self.valid:
            return np.array(
                [
                    self.morphology["predicates"],
                    self.morphology["length"],
                    self.morphology["lines"],
                ]
            ).astype(int)
        else:
            return None


class OpenELMTaskGenerator(LearnableTaskSampler):
    """Container class to include all the configs and generate tasks"""

    def __init__(
        self,
        task_spec: List[ts.TaskSpec],
        checkpoint,
        temperature=1.1,
        batch_size=1,
        gen_fn_name="training_task",
    ):
        assert 0.9 <= temperature <= 1.4, "temperature should be between 0.9 and 1.4"
        super().__init__(task_spec)

        self.config = ELMConfig()
        self.config.batch_size = batch_size

        self.config.env = NMMOConfig()
        self.config.env.impr = import_str["short_import"]
        self.config.env.init_prompt = self.task_spec_to_str(task_spec)
        self.config.env.mutate = True
        self.config.env.batch_size = batch_size
        self.gen_fn_name = gen_fn_name
        self.config.env.gen_fn_name = gen_fn_name

        self.config.qd = MAPElitesConfig()

        self.config.model = PromptModelConfig()
        self.config.model.temp = temperature
        self.config.model.batch_size = batch_size
        self.config.model.model_path = checkpoint
        self.config.model.load_in_8bit = True

    @staticmethod
    def task_spec_to_str(task_spec: List[ts.TaskSpec]):
        """
        Converts a list of TaskSpec objects to a string of function source code.

        Args:
            task_spec: A list of TaskSpec objects.

        Returns:
            A string of function source code.
        """
        return "\n".join(
            set(inspect.getsource(single_spec.eval_fn) for single_spec in task_spec)
        )

    def evolve_tasks(
        self, task_spec: List[ts.TaskSpec], num_tasks, steps=10, debug=False
    ) -> List[ts.TaskSpec]:
        """Evolve the given task specs for the given number of steps
        and return the num_tasks task specs
        """
        if debug:  # just to check if the end-to-end works
            return self.sample_tasks(num_tasks)

        # NOTE: evolve task to generate a function, then generate parameters to deliver num_tasks
        self.config.env.init_prompt = self.task_spec_to_str(task_spec)
        elm = ELM(self.config, env=NMMOEnvironment)

        best_task = None
        while best_task is None:
            elm.run(init_steps=2, total_steps=steps)
            # for now, just using the maximum fitness genome
            # TODO: we may want to sample best ones across the MAP (see MAP-Elites)
            # TODO: vary the name of generated functions. Now, it's all training_task (self.gen_fn_name)
            best_task = elm.qd_algorithm.current_max_genome

        return best_task.generate_task_spec(num_tasks)


@dataclass
class NMMOConfig(EnvConfig):
    """Configuration class for the NMMO environment."""

    env_name: str = "NMMO"
    prebuilt_task_module = nmmo.task.base_predicates

    # Determines the behavior space to improve diversity.
    behavior_space: List[List[float]] = field(
        default_factory=lambda: [[0, 10], [0, 10], [0, 10]]
    )

    init_prompt: str = ""
    impr: str = ""
    starting_seeds: List[str] = field(default_factory=lambda: ["square"])
    instruction: int = 1
    crossover: bool = False
    batch_size: int = 1
    mutate: bool = False
    num_sample_spec = 5  # Number of parameter sets to sample for an eval function.
    gen_fn_name = "training_task"  # Default function name to generate.


class NMMOEnvironment(BaseEnvironment[NMMOTaskFn]):
    """The NMMO environment."""

    def __init__(self, config: NMMOConfig, mutation_model: MutationModel) -> None:
        self.config: NMMOConfig = config
        self.batch_size = self.config.batch_size
        self.mutation_model: MutationModel = mutation_model
        self.genotype_space = np.array(self.config.behavior_space).T
        self.gen_fn_name = config.gen_fn_name
        self.num_sample_spec = config.num_sample_spec
        self.impr = config.impr
        self.prebuilt_task_module = config.prebuilt_task_module

    def construct_prompt(
        self, code_batch: Optional[Union[List[str], str]] = None
    ) -> Dict[str, str]:
        """Constructs the prompt for the environment."""
        prompt_str = self.impr
        task_idea = "explore the map, eat food, and drink water"

        inst = (
            "\n# use the predicates listed in the imports and "
            + "complete the task using diverse predicates than before\n\n"
            + "# normalized float progress to a value between 0 and 1\n"
            + "def norm(progress):\n"
            + "  return max(min(progress, 1.0), 0.0)\n\n"
            + "# training_task evaluates progress towards "
            + task_idea
            + "\n"
            + "# and must return a float between 0-1 using norm()\n"
            + "# the lines of code should be less than 30\n"
            + f"def {self.gen_fn_name}(gs: GameState, subject: Group, "
        )
        prompt_str += inst
        return {"prompt": prompt_str, "template": self.impr + inst}

    def _generate_task_fn(self, code_batch: List[Dict[str, str]]) -> str:
        """Generates task functions."""
        local_scope_exec: bool = False
        return self.mutation_model.generate_programs(code_batch, local_scope_exec)

    def generate_programs(self, code_batch: List[Dict[str, str]]) -> List[NMMOTaskFn]:
        """Generates task programs."""
        eval_fn_batch = self._generate_task_fn(code_batch)
        task_list = []
        for gen_str in eval_fn_batch:
            gene = NMMOTaskFn(gen_str, self.gen_fn_name, self.prebuilt_task_module)
            if gene.valid:
                task_list.append(gene)
        return task_list

    def random(self) -> list[NMMOTaskFn]:
        """Generates a random task."""
        program_list = [
            self.construct_prompt() for _ in range(2)
        ]  # NOTE: where the 2 comes from?
        new_tasks = self.generate_programs(program_list)
        return new_tasks

    def mutate(self, x: list[NMMOTaskFn]) -> list[NMMOTaskFn]:
        """Mutate the text of the given tasks with ELM"""
        task_list = [sr.program_str for sr in x]
        program_list = list(map(self.construct_prompt, task_list))
        new_tasks = self.generate_programs(program_list)
        return new_tasks

    def fitness(self, x: NMMOTaskFn) -> float:
        """Evaluate fitness if x is valis, -inf otherwise"""
        if x.valid:
            return x.evaluate()
        return -np.inf

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        #warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        #warnings.warn("WARNING: rng state not used in this environment")
        pass

def entropy(task):
    """
    Computes the entropy of the task string by counting the frequency of each word.
    This can be used as a measure of complexity or variability in the task string.

    Args:
        task: A task string

    Returns:
        The entropy of the task string, rescaled to be between 0 and 10.
    """
    words = [word for word in re.split(r"[ _\(\):]+", task) if word]
    word_prob = [count / len(words) for count in Counter(words).values()]
    ent = -sum(prob * math.log2(prob) for prob in word_prob)
    return min(math.ceil(ent), 10)


def calculate_length(task):
    """
    Scales the length of the task string to a value between 0 and 10.

    Args:
        task: A task string

    Returns:
        The scaled length of the task string.
    """
    scale = (
        lambda val, vmin, vmax, dmin, dmax: ((val - vmin) / (vmax - vmin))
        * (dmax - dmin)
        + dmin
    )
    return math.ceil(scale(len(task), 100, 9000, 0, 10))