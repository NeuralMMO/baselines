'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

from collections import defaultdict
from tqdm import tqdm
import numpy as np

import ray
import openskill

import nmmo

from scripted import baselines
import tasks

from demos import minimal


def parallel_simulations(cores, horizon):
    '''Simulate environments in parallel and compute skill ratings

    Runs one environment per core for horizon steps, then uses
    the nmmo.OpenSkillRating wrapper to estimate per-policy skill'''
    nmmo.MapGenerator(Config()).generate_all_maps()

    # Parallel sim using base ray
    all_stats = ray.get([
            ray.remote(minimal.simulate).remote(
                    nmmo.Env, Config, horizon=horizon)
            for worker in range(cores)])

    # NMMO OpenSkill wrapper for computing SR
    sr = nmmo.OpenSkillRating(Config.AGENTS, anchor=baselines.Combat)
    for stats in all_stats:
        stats = stats['Stats']

        # SR updates take a list of policy IDs and scores for all agents
        ratings = sr.update(
                policy_ids=stats['PolicyID'],
                scores=stats['Task_Reward'])

        # Prettify results
        results = []
        for agent, rating in ratings.items():
            results.append(f'{agent.__name__}: {str(rating.mu)[:6]:.6s}')
        print('   '.join(results))


class Config(nmmo.config.Default):
    '''Default baseline config with only scripted agents'''
    AGENTS = [baselines.Meander, baselines.Forage, baselines.Combat]
    TASKS  = tasks.All

    # Share maps with baseline evaluations
    PATH_MAPS = 'maps/medium/evaluation/'


if __name__ == '__main__':
    CORES   = 100
    HORIZON = 128

    ray.init()
    parallel_simulations(CORES, HORIZON)
