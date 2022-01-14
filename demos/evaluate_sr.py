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


def rank(policy_ids, scores):
    '''Compute policy rankings from per-agent scores'''
    agents = defaultdict(list)
    for policy_id, score in zip(policy_ids, scores):
        agents[policy_id].append(score + 1e-8*np.random.normal())

    # Double argsort returns ranks
    return np.argsort(
            np.argsort(
                    [-np.mean(vals) for policy, vals in 
                    sorted(agents.items())])).tolist()


class OpenSkillRating:
    '''Rating wrapper'''
    def __init__(self, mu=1e-9, sigma=100/3):
        # 95% win chance against 2*sigma lower SR
        self.ratings = {agent.__name__: openskill.Rating(mu=mu, sigma=sigma)
                for agent in Config.AGENTS}

        self.mu    = mu
        self.sigma = sigma

        self.anchor()

    def update(self, ranks):
        teams = [[e] for e in list(self.ratings.values())]
        ratings = openskill.rate(teams, rank=ranks)
        ratings = [openskill.create_rating(team[0]) for team in ratings]
        for agent_name, rating in zip(self.ratings, ratings):
            self.ratings[agent_name] = rating

        self.anchor()

    def anchor(self, anchor='Meander'):
        for agent_name, rating in self.ratings.items():
            rating.sigma = self.sigma
            if agent_name == anchor:
                rating.mu = self.mu
                rating.sigma = self.sigma


def parallel_simulations(cores, horizon):
    '''Simulate environments in parallel and compute skill ratings'''
    nmmo.MapGenerator(Config()).generate_all_maps()

    # Parallel sim using base ray
    all_stats = ray.get([
            ray.remote(minimal.simulate).remote(
                    nmmo.Env, Config, horizon=horizon)
            for worker in range(cores)])

    for stats in all_stats:
        stats      = stats['Stats']
        ranks      = rank(stats['PolicyID'], stats['Task_Reward'])
        sr.update(ranks)

        print(sr.ratings)


class Config(nmmo.config.Default):
    AGENTS = [baselines.Meander, baselines.Forage, baselines.Combat]
    TASKS  = tasks.All


if __name__ == '__main__':
    CORES   = 10
    HORIZON = 32

    ray.init()
    parallel_simulations(CORES, HORIZON)
