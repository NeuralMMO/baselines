from typing import Optional, Union
from dataclasses import dataclass, field

import re
import numpy as np

from openelm.configs import EnvConfig
from openelm.environments import BaseEnvironment, Genotype
from openelm.mutation_model import DiffModel

from .elm_helper import calculate_length, check_task_spec, str_to_task_spec, UNIQUE_PREDICATES


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


Phenotype = Optional[np.ndarray]
class NMMOTask(Genotype):
  """A task in the NMMO environment."""
  def __init__(self, program_str: str, impr: str = None):
    self._fitness = -np.inf
    # remove the prompt from the program_str
    program_str = program_str.replace(impr, "")
    # extract def task_ from the program_str
    split = program_str.split("\n")
    task = []
    # Consider only the last task generated
    for lines in split[::-1]:
      if lines.startswith("def task_"):
        task.append(lines)
        break
      task.append(lines)

    program_str = "\n".join(task[::-1])
    # print("Program_str at init")
    # print(program_str)

    # to check if the task is valid
    if self.check_valid(program_str):
      self.valid = True
      self.program_str: str = re.sub(r" +#.*\n", "", program_str)
      # need to ignore comments
      self.morphology = {}
      self.morphology["predicates"] = self._count_predicates(program_str)
      self.morphology["length"] = calculate_length(program_str)
      self.morphology["lines"] = program_str.count("\n")
    else:
      self.valid = False

  def evaluate(self) -> float:
    # how to evaluate the fitness of a task? (time taken for the baseline RL algo to solve?)
    # for now, just using the length of the task. The longer the task, the better the fitness.
    # NOTE: This is a very BAD fitness function.

    # TODO: could include like penalizing for hallucinated predicates.
    # low values if the task is either to easy or too hard
    self._fitness = len(self.program_str)/10
    return self._fitness

  def check_valid(self, program_str: str):
    # if program_str has more than 2048 tokens the tasks is not valid

    # Important function to quickly check if the generated task is valid.
    tokens = len(program_str)/5
    if tokens >= 2048:
      return False

    # CHECK ME: convert to task spec, is the task spec correct?
    try: 
      task_spec = str_to_task_spec([program_str])[0]
      if check_task_spec(task_spec):
        return True
    except: # pylint: disable=bare-except
      pass

    return False

  def __str__(self) -> str:
    return self.program_str

  def _count_predicates(self, task_str):
    predicates = set()
    for i in UNIQUE_PREDICATES:
      if i in task_str:
        predicates.add(i)
    return len(predicates)

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


class NMMOEnvironment(BaseEnvironment[NMMOTask]):
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

    # instruction added to the prompt
    inst = "\n# use the predicates listed in the imports and " +\
           "complete the task using diverse predicates than before\ndef task_"
    import_s += inst
    prompt_str += inst
    return {"prompt": prompt_str, "template": import_s}

  # NMMOTask should only contain the task, after the prompt
  def generate_programs(self, code_batch: list[dict[str, str]]) -> list[NMMOTask]:
    local_scope_exec: bool = False
    generated_tasks = self.mutation_model.generate_programs(
      code_batch, local_scope_exec
    )
    # check if each task is valid then append to the list
    task_list = []
    for t in generated_tasks:
      task = NMMOTask(t, self.impr)
      if task.valid:
        task_list.append(task)
    return task_list

  # Executed First
  def random(self) -> list[NMMOTask]:
    program_list = [self.construct_prompt() for _ in range(2)]
    new_tasks = self.generate_programs(program_list)
    return new_tasks

  def mutate(self, x: list[NMMOTask]) -> list[NMMOTask]:
    task_list = [sr.program_str for sr in x]
    program_list = list(map(self.construct_prompt, task_list))
    new_tasks = self.generate_programs(program_list)
    return new_tasks

  def fitness(self, x: NMMOTask) -> float:
    if x.valid:
      return x.evaluate()
    return -np.inf
