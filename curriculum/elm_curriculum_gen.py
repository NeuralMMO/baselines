from typing import Optional, Union
from dataclasses import dataclass, field

import re
import math
import Counter

import wandb
import random
import argparse
import numpy as np


from openelm import ELM
from openelm.configs import ELMConfig, PromptModelConfig, EnvConfig, MAPElitesConfig
from openelm.environments import BaseEnvironment, Genotype, ENVS_DICT
from openelm.mutation_model import DiffModel

import nmmo
from train_helper import SimpleTaskGenerator
from nmmo.task.group import Group
from nmmo.systems.item import ItemState
from nmmo.entity.entity import Entity, EntityState
from nmmo.task.task_api import OngoingTask, make_team_tasks
from nmmo.task.predicate.core import Predicate, AND, OR
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.task.predicate_api import make_predicate, Predicate

from transformers import AutoTokenizer

from sample_tasks import uniq_predicates, tasks, import_str

"""
Script to use the OpenELM package to generate/evolve tasks for 2023 NeurIPS competition.
The default config/args in this script can be used to generate the competition baseline tasks
"""

# # Mock Classes to test the generated tasks

# # optional code to improve task quality
# class MockRealm:
#   def __init__(self):
#     self.config = nmmo.config.Default()
#     self.config.PLAYERS = range(100)
#     self.datastore = NumpyDatastore()
#     self.items={}
#     self.datastore.register_object_type("Entity", EntityState.State.num_attributes)
#     self.datastore.register_object_type("Item", ItemState.State.num_attributes)

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
    entropy = -sum(prob * math.log2(prob) for prob in word_prob)

    # rescale to behaviour space
    return min(math.ceil(entropy),10)

def calculate_length(task):
    """Scaling metrics between two values. It is very important for the selected phenotypes 
    to be able to have values and easily move across the defined behaviour space. in this case 0-10 """
    # scale # of characters in task (100-9000) to behaviour space 0-10
    min_val = 100
    max_val = 9000
    new_min = 0
    new_max = 10

    # Scale the value
    scaled_value = ((len(task) - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

    return math.ceil(scaled_value)

def extract_kwargs(function_definition):
    pattern = r'def\s+\w+\((.*?)\)'
    parameter_pattern = r'(\w+)\s*:\s*([^\s,]+)'

    match = re.search(pattern, function_definition)
    if match:
        parameters_string = match.group(1)
        parameters = re.findall(parameter_pattern, parameters_string)
        parameter_dict = {name: data_type for name, data_type in parameters}
        return parameter_dict
    else:
        return {}
    
def check_task_spec(spec_list):
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
    task_specs = []
    for task in task_list:
        func = {}
        exec(task, globals(), func)
        # get kwargs
        kwargs = extract_kwargs(task)
        task_specs.append(("agent", task, kwargs))
    return task_specs

def task_spec_to_str(task_specs):
    # convert task spec to str code
    str_tasks = []
    for task_spec in task_specs:
        predicate = make_predicate(task_spec[1])
        inst_predicate = predicate(Group(0))
        str_tasks.append(inst_predicate.get_source_code())
    return "\n".join(str_tasks)

@dataclass
class NMMOConfig(EnvConfig):
    """Config for the NMMO environment."""
    env_name:str = "NMMO"
    # Important for specifying the diversity of a task. Well defined the behaviour space, better quality/diversity from OpenELM
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
        # for now, just the length of the task. The longer the task, the better the fitness (a very bad fitness function). 

        # Very important metric, could include like penalizing for hallucinated predicates. 
        # low values if the task is either to easy or too hard
        self._fitness = len(self.program_str)/10

        return self._fitness
    
    def check_valid(self, program_str: str):
        # if program_str has more than 2048 tokens the tasks is not valid

        # Important function to quickly check if the generated task is valid. 
        
        tokens = len(program_str)/5
        if tokens >= 2048:
            return False
        
        # convert to task spec
        
        task_spec = str_to_task_spec([program_str])[0]
        if check_task_spec(task_spec):
            return True
        else:
            return False

    def __str__(self) -> str:
        return self.program_str
    
    def _count_predicates(self, task_str):
        predicates = set()
        for i in uniq_predicates:
            if i in task_str:
                predicates.add(i)
        return len(predicates)
        
    def to_phenotype(self) -> Optional[Phenotype]:
        # phenotypes of the task?
        # creating a dummy version, string length of the task, unique predicates, difficulty of the task,
        # if self.valid:
        return np.array(
            [
                self.morphology["predicates"],
                self.morphology["length"],
                self.morphology["lines"]
            ]
        )
        # else: 
        #     return None

    @property
    def fitness(self) -> Optional[float]:
        return self._fitness


class NMMOEnvironment(BaseEnvironment[NMMOTask]):
    """The NMMO environment."""

    def __init__(
        self,
        config: NMMOConfig,
        mutation_model: DiffModel,
    ) -> None:

        self.config: NMMOConfig = config
        self.batch_size = self.config.batch_size
        self.mutation_model: DiffModel = mutation_model
        self.genotype_space = np.array(self.config.behavior_space).T
        self.genotype_ndim = self.genotype_space.shape[1]
        self.impr = config.impr
        self.init_prompt = config.init_prompt

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
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
        inst = "\n# use the predicates listed in the imports and complete the task using diverse predicates than before\ndef task_"
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
        else:
            return -np.inf


class OpenELMCurriculumGenerator():
    """Container class to include all the configs and generate tasks"""

    def __init__(self, temperature, model, batch_size, task_specs):

        super().__init__(task_specs)
        
        assert model in ["2", "6"], "model should be either 2B or 6B"
        assert 0.9<=temperature<=1.4, "temperature should be between 0.9 and 1.4"
        
        self.config = ELMConfig()
        self.config.batch_size = batch_size
        
        self.config.env = NMMOConfig()
        self.config.env.impr = import_str["short_import"]
        self.config.env.init_prompt = task_spec_to_str(task_specs)
        self.config.env.mutate = True
        self.config.env.batch_size = batch_size

        self.config.qd = MAPElitesConfig()

        self.config.model = PromptModelConfig()
        self.config.model.temp = temperature
        self.config.model.batch_size = batch_size
        self.config.model.model_path = f"Salesforce/codegen-{model}-mono"

        ENVS_DICT["NMMO"] = NMMOEnvironment

    def evolve_tasks(self, steps, task_spec):
        """Evolve the given task specs for the given number of steps and return the evolved task specs"""
        self.config.env.init_prompt = task_spec_to_str(task_spec)
        elm = ELM(self.config)
        elm.run(init_steps = 2, total_steps = steps)
        # flatten genomes to a list of tasks
        return str_to_task_spec(list(elm.qd_algorithm.genomes.array.flatten()))
        



def main(temperature, imports, n_tasks, model, mutate, batch_size):

    return elm.run(init_steps = 2, total_steps = 100)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--temperature", type=float, default=0.9, help="temperature for the LLM for sampling")
    args.add_argument("--mutate", action='store_true', default=False, help="include the task output from the LLM in the next gens of prompt or not")
    args.add_argument("--imports", type=str, default="short_import", help="Use a smaller import statement or a larger well-defined one")
    args.add_argument("--n_tasks", type=int, default=2, help="number of sample tasks to use as prompt")
    args.add_argument("--model", type=int, default=2, help="model size to use, 2B/6B")
    args.add_argument("--batch_size", type=int, default=4, help="batch size")
    # Deep Speed
    # args.add_argument("--local_rank", type=int, default=0)
    args = args.parse_args()

    wandb.init(
        project="NMMO-ELM",
        config=vars(args)
    )

    max_fitness, niches, qd, fitnesses = main(args.temperature, args.imports, args.n_tasks, args.model, args.mutate, args.batch_size)

    # write max fitness niches and qd to file with n_tasks, temperature, model, imports as the file name
    with open(f"tasks_{args.n_tasks}_{args.temperature}_{args.imports}.txt", "w", encoding="utf-8") as f:
        f.write(f"max fitness: {max_fitness}\n")
        f.write(f"niches: {niches}\n")
        f.write(f"qd: {qd}\n")
    print(f"Niches filled: {niches}")
    print(f"QD: {qd}")


    # TODO:

    # [] - Generate only Valid tasks
    # [] - Measure fitness with respect to predicates only

