from typing import Optional, Union
from dataclasses import dataclass, field

import re
import wandb
import random
import argparse
import numpy as np


from openelm import ELM
from openelm.configs import ELMConfig, PromptModelConfig, EnvConfig, MAPElitesConfig
from openelm.environments import BaseEnvironment, Genotype, ENVS_DICT
from openelm.mutation_model import DiffModel

from transformers import AutoTokenizer

from sample_tasks import uniq_predicates, tasks, import_str

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")

@dataclass
class NMMOConfig(EnvConfig):
    """Config for the NMMO environment."""
    env_name:str = "NMMO"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            # Unique_predicates, length of the task, difficulty
            [0, 20],
            [0, 20],
            [0, 20],
        ]
    )
    init_prompt: str = ""
    impr: str = ""
    starting_seeds: list[str] = field(default_factory=lambda: ["square"])
    instruction: int = 1
    crossover: bool = False
    batch_size: int = 1
    mutate: bool = False

# optional code to improve task quality
# class MockRealm:
#   def __init__(self):
#     self.config = nmmo.config.Default()
#     self.config.PLAYERS = range(100)
#     self.datastore = NumpyDatastore()
#     self.items={}
#     self.datastore.register_object_type("Entity", EntityState.State.num_attributes)
#     self.datastore.register_object_type("Item", ItemState.State.num_attributes)

Phenotype = Optional[np.ndarray]
class NMMOTask(Genotype):
    """A task in the NMMO environment."""
    def __init__(self, program_str: str, impr: str = None):

        # remove the prompt from the program_str
        program_str = program_str.replace(impr, "")
        # extract def task_ from the program_str
        split = program_str.split("\n")
        task = []
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
            self.morphology["length"] = len(program_str)/100
            self.morphology["lines"] = program_str.count(r"\n")
        else:
            self.valid = False

    def evaluate(self) -> float:
        # how to evaluate the fitness of a task? (time taken for the baseline RL algo to solve?)
        # for now, just the length of the task
        self._fitness = len(self.program_str)/10

        return self._fitness
    
    def check_valid(self, program_str: str):
        # additional checks if tasks are correct
        # if program_str has more than 2048 tokens the tasks is not valid
        
        tokens = len(tokenizer(program_str)["input_ids"])
        if tokens >= 2048:
            return False
        
        return True

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


class NMMO(BaseEnvironment[NMMOTask]):
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
            
        # instruction postpended to the prompt
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
        # code to check the validity of the task, commented out to speed up the process
        # for task in generated_tasks:
        #     with open("generated_progam.py","a") as f:
        #         f.write(task)
        # realm = MockRealm()
        # entity_id = 123
        # population_id = 11
        # entity = Entity(realm, (10,20), entity_id, "name", "color", population_id)


        # results = pool_exec_processes(
        #     generated_tasks,
        #     timeout=5.0,
        #     args={"entity":entity},
        #     debug=False
        # )
        # result_list: list = []
        # for i, result in enumerate(results):
        #     try:
        #         if isinstance(result, AND) or isinstance(result, OR) or isinstance(result, Predicate): 
        #             print(generated_tasks[i])
        #             result_list.append(generated_tasks[i])
        #     except Exception as e:
        #         print(type(e))

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

def main(temperature, imports, n_tasks, model, mutate, batch_size):

    impr = import_str[imports]
    prompt_tasks = "\n" + "\n".join(random.sample(tasks, n_tasks))

    config = ELMConfig()
    config.env = NMMOConfig()
    config.env.impr = impr
    config.env.init_prompt = prompt_tasks
    config.env.mutate = mutate
    config.env.batch_size = batch_size
    config.qd = MAPElitesConfig()
    # config.qd.map_grid_size = (20)
    config.model = PromptModelConfig()
    config.model.temp = temperature
    config.model.batch_size = batch_size
    config.batch_size = batch_size
    config.model.model_path = f"Salesforce/codegen-{model}B-mono"

    ENVS_DICT["NMMO"] = NMMO

    elm = ELM(config)

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