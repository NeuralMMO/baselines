import random
import argparse
from typing import Tuple

import nmmo
import nmmo.task.task_api as t
import nmmo.task.base_predicates as p
from nmmo.task import constraint
from nmmo.core.config import Config
from nmmo.core.env import Env as TaskEnv
from nmmo.task.scenario import Scenario

from scripted import baselines

"""
Script to heuristically generate and save baseline tasks for 2023 NeurIPS competition.
The default args in this script are the ones used to generate the competition baseline tasks.
"""
# TODO: type hints

class RandomTaskInfoGenerator:
  def __init__(self, config: Config) -> None:
    """
    Generates info for clauses of Predicates which can be combined to create a task.

    Args:
        config: game instance config
    """
    self.config = config
    self._pred_specs = []
    self._pred_spec_weights = []

  def add_pred_spec(self, pred_class: t.Predicate,
                    param_space: list[constraint.Constraint],
                    weight: float = 1):
    """
    Builds the list of Predicates to choose from when sampling.

    Args:
      pred_class: (base) Predicate class
      param_space: list containing Constraint type for each param of the pred_class;
                   SHOULD NOT include a Constraint for the 'subject' param as that is defined at runtime
      weight: weighting for this pred_spec in random choice (compared to other pred_specs)
    """
    self._pred_specs.append((pred_class, param_space or []))
    self._pred_spec_weights.append(weight)

  def sample(self,
             min_clauses: int = 1,
             max_clauses: int = 3,
             min_clause_size: int = 1,
             max_clause_size: int = 3,
             not_p: float = 0.05,
             min_reward: float = 0.1,
             max_reward: float = 1.0) -> Tuple[t.Predicate, float]:
    """
    Randomly generates parameters that can be used to instantiate clauses of Predicates.

    Args:
        min_clauses: min clauses in the task
        max_clauses: max clauses in the task
        min_clause_size: min Predicates in each clause
        max_clause_size: max Predicates in each clause
        not_p: probability that a Predicate will be NOT'd
    """

    task_info = {}

    # A list of lists
    # Outer list: each index corresponds to one clause of the task
    # Inner list: each index corresponds to one Predicate of the clause
    clauses = []

    num_clauses = random.randint(min_clauses, max_clauses)
    for _ in range(num_clauses):
      pred_specs = random.choices(
        self._pred_specs,
        weights = self._pred_spec_weights,
        k = random.randint(min_clause_size, max_clause_size)
      )

      pred_list = [] # Inner list
      for pred_class, pred_param_space in pred_specs:
        pred_info = {
           'class' : pred_class,
           'not'   : random.random() < not_p,
           'params': [pp.sample(self.config) for pp in pred_param_space]
        }
        pred_list.append(pred_info) 

      clauses.append(pred_list)
    
    task_info['clauses'] = clauses
    task_info['reward'] = random.uniform(min_reward, max_reward)
    
    return task_info

class HeuristicTaskGenerator:
    def __init__(self, config: Config):
        """
        Generates Tasks by heuristic input/output shaping using RandomTaskInfoGenerator.
        """
        self.config = config

    def generate_task_definitions(self,
                                  n: int=1,
                                  max_clauses: int=3,
                                  max_clause_size: int=3,
                                  not_p: float=0.05):
        """
        Uses RandomTaskInfoGenerator 

        Args:
            config: game instance config
            max_clauses: max clauses in each Task
            max_clause_size: max Predicates in each clause
            not_p: probability that a Predicate will be NOT'd
        """

        ### Predicate specific values for rng ###
        N = constraint.ScalarConstraint()
        coord = constraint.COORDINATE_CONSTRAINT
        tile_type = constraint.MATERIAL_CONSTRAINT
        agent = constraint.AGENT_NUMBER_CONSTRAINT
        group = constraint.TEAM_GROUPS
        skill = constraint.SKILL_CONSTRAINT
        level = constraint.PROGRESSION_CONSTRAINT
        event = constraint.EVENTCODE_CONSTRAINT
        combat_style = constraint.COMBAT_SKILL_CONSTRAINT
        item = constraint.ITEM_CONSTRAINT
        inventory_cap = constraint.INVENTORY_CONSTRAINT
        consumable = constraint.CONSUMABLE_CONSTRAINT

        gen = RandomTaskInfoGenerator(self.config)
        gen.add_pred_spec(p.TickGE, [N])
        gen.add_pred_spec(p.CanSeeTile, [tile_type])
        gen.add_pred_spec(p.StayAlive, [])
        gen.add_pred_spec(p.AllDead, []) # TODO: should this predicate have a "target" param?
        gen.add_pred_spec(p.OccupyTile, [coord, coord])
        gen.add_pred_spec(p.AllMembersWithinRange, [coord])
        gen.add_pred_spec(p.CanSeeAgent, [agent])
        gen.add_pred_spec(p.CanSeeGroup, [group])
        gen.add_pred_spec(p.DistanceTraveled, [N]) # TODO: should this param be COORDINATE_CONSTRAINT?
        gen.add_pred_spec(p.AttainSkill, [skill, level, agent])
        gen.add_pred_spec(p.CountEvent, [event, N])
        gen.add_pred_spec(p.ScoreHit, [combat_style, N])
        gen.add_pred_spec(p.HoardGold, [N])
        gen.add_pred_spec(p.EarnGold, [N])
        gen.add_pred_spec(p.SpendGold, [N])
        gen.add_pred_spec(p.MakeProfit, [N])
        gen.add_pred_spec(p.InventorySpaceGE, [N]) # TODO: should this param be INVENTORY_CONSTRAINT?
        gen.add_pred_spec(p.OwnItem, [item, level, inventory_cap])
        gen.add_pred_spec(p.EquipItem, [item, level, agent])
        gen.add_pred_spec(p.FullyArmed, [combat_style, level, agent])
        gen.add_pred_spec(p.ConsumeItem, [consumable, level, N])
        gen.add_pred_spec(p.HarvestItem, [item, level, N])
        gen.add_pred_spec(p.ListItem, [item, level, N])
        gen.add_pred_spec(p.BuyItem, [item, level, N])

        task_infos = []
        for _ in range(n):
            task_info = gen.sample(max_clauses=max_clauses,
                                   max_clause_size=max_clause_size,
                                   not_p=not_p)
            task_infos.append(task_info)

        return task_infos

    def generate_task(self, task_info: list):
        """
        Takes the generated clause info, instantiates the Predicates, and combines them using 
        Conjunctive Normal Form (CNF).

        Args:
            task_info: list of infos for each clause in the task
        """
        @t.define_predicate
        def new_task(gs, subject):
            clauses = []
            for clause_info in task_info:
                predicates = []
                for pred_info in clause_info:
                    # Instantiate Predicate
                    pred_class = pred_info['class']
                    predicate = pred_class(subject, *pred_info['params']) 
                    if pred_info['not'] == True: predicate = t.NOT(predicate) 
                    predicates.append(predicate)

                clause = t.OR(*predicates)
                clauses.append(clause)
        
            task = t.AND(*clauses)
            print("TASK: ", task)
            return task

        return new_task

    def generate_tasks(self, n: int):
        """
        Generates n tasks.
        """
        task_infos = self.generate_task_definitions(n)
        tasks = []
        for task_info in task_infos:
            task = self.generate_task(task_info['clauses'])
            tasks.append(task()*task_info['reward'])

        return tasks

# tmp test config
class ScriptedAgentTestConfig(nmmo.config.Small, nmmo.config.AllGameSystems):
  __test__ = False

  LOG_ENV = True

  LOG_MILESTONES = True
  LOG_EVENTS = False
  LOG_VERBOSE = False

  SPECIALIZE = True
  PLAYERS = [
    baselines.Fisher, baselines.Herbalist,
    baselines.Prospector,baselines.Carver, baselines.Alchemist,
    baselines.Melee, baselines.Range, baselines.Mage]

def test_rollout(tasks):
    # Test rollout with each task
    # Each task_info corresponds to one task
    for task in tasks:
        env = TaskEnv(config)
        scenario = Scenario(config)
        scenario.add_tasks(task)
        env.change_task(scenario.tasks)
        for _ in range(30):
            env.step({})

if __name__ == '__main__':
    # TODO: put necessary things as args
    #   seed, generate_task_definitions params, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n', default=5, help='number of tasks to generate')
    args = parser.parse_args()

    config = ScriptedAgentTestConfig()
    gen = HeuristicTaskGenerator(config)
    tasks = gen.generate_tasks(n=int(args.n))

    # TODO: implement save and load generated tasks (or task_infos)
    #   and add command line arg to evaluate tasks from load for demo

    # Test run tasks to see if they "compile" in the environment
    test_rollout(tasks)
