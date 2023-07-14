from typing import Optional, Union
from dataclasses import dataclass, field

import ast
import re
import numpy as np

from openelm.configs import EnvConfig
from openelm.environments import BaseEnvironment, Genotype
from openelm.mutation_model import DiffModel

from .elm_helper import calculate_length, PREBUILT_TASK_FN
from .elm_helper import generate_task_spec, extract_task_fn, is_task_spec_valid


@dataclass
class NMMOConfig(EnvConfig):
  """Config for the NMMO environment."""
  env_name:str = "NMMO"

  # Important for specifying the diversity of a task.
  # Well defined the behaviour space, better quality/diversity from OpenELM
  behavior_space: list[list[float]] = field(
    default_factory=lambda: [
      # Baseline config considers the number of unique predicates,
      # length of the task/normalized
      # number of lines in the task / normalized
      [0, 10],
      [0, 10],
      [0, 10],
    ]
  )

  init_prompt: str = ""
  impr: str = ""
  starting_seeds: list[str] = field(default_factory=lambda: ["square"])
  instruction: int = 1
  crossover: bool = False
  batch_size: int = 1
  mutate: bool = False

  # how many sets of kwargs parameters (task_spec) to sample for a valid eval fn
  num_sample_spec = 5
  # the default name to generate
  gen_fn_name = "training_task"


Phenotype = Optional[np.ndarray]
class NMMOTaskFn(Genotype):
  """A task in the NMMO environment."""
  def __init__(self, program_str: str, fn_name: str):
    self._fitness = -np.inf
    self._fn_name = fn_name
    self.program_str = extract_task_fn(program_str, self._fn_name)
    self.valid = self.is_valid_nmmo_fn(self.program_str)
    self.morphology = {}

    if self.valid:
      code_only = re.sub(r" +#.*\n", "", self.program_str) # remove comments
      # TODO: add more metrics here
      self.morphology["predicates"] = self._count_predicates(code_only)
      self.morphology["length"] = calculate_length(code_only)
      self.morphology["lines"] = code_only.count("\n")

  def evaluate(self) -> float:
    # how to evaluate the fitness of a task? (time taken for the baseline RL algo to solve?)
    # for now, just using the length of the task. The longer the task, the better the fitness.
    # NOTE: This is a very BAD fitness function.

    # TODO: could include like penalizing for hallucinated predicates.
    # low values if the task is either to easy or too hard
    self._fitness = len(self.program_str)/10
    return self._fitness

  def is_valid_nmmo_fn(self, eval_fn_str: str):
    # if program_str has more than 2048 tokens the tasks is not valid
    # Important function to quickly check if the generated task is valid.
    tokens = len(eval_fn_str)/5
    if tokens >= 2048:
      return False

    # generate_task_spec returns task_spec if the function is valid python
    task_spec = generate_task_spec(eval_fn_str, self._fn_name)

    # is_task_spec_valid tests if these task_spec runs in the nmmo
    return is_task_spec_valid(task_spec)

  def generate_task_spec(self, num_sample=None):
    if not self.valid:
      return []
    task_spec = []
    for single_spec in generate_task_spec(self.program_str, self._fn_name, num_sample):
      if is_task_spec_valid([single_spec]):
        task_spec.append(single_spec)
    return task_spec

  def __str__(self) -> str:
    return self.program_str

  def _count_predicates(self, task_str):
    # NOTE: PREBULIT_TASK_FN contains norm and other trivial fns
    called_fns = [node.func.id for node in ast.walk(ast.parse(task_str))
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    used_pred_fn = {fn_name for fn_name in called_fns
                    if fn_name in PREBUILT_TASK_FN}
    return len(used_pred_fn)

  def to_phenotype(self) -> Optional[Phenotype]:
    # phenotypes of the task?
    # this is a dummy version based on string length of the task, unique predicates,
    # if self.valid:
    return np.array(
      [
        self.morphology["predicates"],
        self.morphology["length"],
        self.morphology["lines"]
      ]
    )
    # else:
    #   return None

  @property
  def fitness(self) -> Optional[float]:
    return self._fitness


# TODO: make the prompts, etc easy to edit
class NMMOEnvironment(BaseEnvironment[NMMOTaskFn]):
  """The NMMO environment."""
  def __init__(self,
               config: NMMOConfig,
               mutation_model: DiffModel,) -> None:
    # pylint: disable=super-init-not-called
    self.config: NMMOConfig = config
    self.batch_size = self.config.batch_size
    self.mutation_model: DiffModel = mutation_model
    self.genotype_space = np.array(self.config.behavior_space).T
    self.genotype_ndim = self.genotype_space.shape[1]
    self.impr = config.impr
    self.init_prompt = config.init_prompt
    self.gen_fn_name = config.gen_fn_name
    self.num_sample_spec = config.num_sample_spec

  def construct_prompt(self,
                       code_batch: Optional[Union[list[str], str]] = None
                       ) -> dict[str, str]:
    import_s = self.impr
    prompt_str = import_s

    if self.config.mutate and code_batch is not None:
      # add prev task to the prompt if mutate is true
      if isinstance(code_batch, list):
        prompt_str += "\n"+code_batch[0]
      elif isinstance(code_batch, str):
        prompt_str += "\n"+code_batch
    else:
      prompt_str += "\n"+self.init_prompt

    # this inst seems critical in determining what function the elm writes
    # TODO: use heuristics or LLM to generate a diverse goal statement here
    task_idea = "explore the map, eat food, and drink water"

    # instruction added to the prompt
    inst = "\n# use the predicates listed in the imports and " +\
           "complete the task using diverse predicates than before\n\n" +\
           "# normalized float progress to a value between 0 and 1\n" +\
           "def norm(progress):\n" + \
           "  return max(min(progress, 1.0), 0.0)\n\n" + \
           "# training_task evaluates progress towards " + task_idea + "\n" +\
           "# and must return a float between 0-1 using norm()\n" +\
           "# the lines of code should be less than 30\n" +\
           f"def {self.gen_fn_name}(gs: GameState, subject: Group, "
    import_s += inst
    prompt_str += inst
    return {"prompt": prompt_str, "template": import_s}

  # returns a string that contains the generated eval fn
  def _generate_task_fn(self, code_batch: list[dict[str, str]]) -> str:
    local_scope_exec: bool = False
    return self.mutation_model.generate_programs(
      code_batch, local_scope_exec
    )

  def generate_programs(self, code_batch: list[dict[str, str]]) -> list[NMMOTaskFn]:
    eval_fn_batch = self._generate_task_fn(code_batch)
    task_list = []
    for gen_str in eval_fn_batch:
      gene = NMMOTaskFn(gen_str, self.gen_fn_name)
      if gene.valid:
        task_list.append(gene)
    return task_list

  # Executed First
  def random(self) -> list[NMMOTaskFn]:
    # NOTE: where the 2 comes from?
    program_list = [self.construct_prompt() for _ in range(2)]
    new_tasks = self.generate_programs(program_list)
    return new_tasks

  def mutate(self, x: list[NMMOTaskFn]) -> list[NMMOTaskFn]:
    task_list = [sr.program_str for sr in x]
    program_list = list(map(self.construct_prompt, task_list))
    new_tasks = self.generate_programs(program_list)
    return new_tasks

  def fitness(self, x: NMMOTaskFn) -> float:
    if x.valid:
      return x.evaluate()
    return -np.inf
