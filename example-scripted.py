'''Scripted-only example not dependent upon RLlib and WanDB'''

from pdb import set_trace as T

from collections import defaultdict
from tqdm import tqdm
import numpy as np

import ray
import openskill
import trueskill

from nmmo import Env, config

from scripted import baselines
import tasks


class Config(config.Default):
    AGENTS  = [baselines.Meander, baselines.Forage, baselines.Combat]
    TASKS   = tasks.All

@ray.remote
def simulate(horizon, print_progress=True):
    '''Simulate an environment for a fixed horizon'''
    conf = Config()
    env  = Env(conf)
    env.reset()

    iterable = range(HORIZON)
    if print_progress:
        iterable = tqdm(iterable)

    for t in iterable:
        actions = {} #Scripted API computes actions
        obs, rewards, dones, infos = env.step(actions=actions)

    return env.terminal()['Stats']

def rank(policy_ids, scores):
    '''Compute policy rankings from per-agent scores'''
    agents = defaultdict(list)
    for policy_id, score in zip(policy_ids, scores):
        agents[policy_id].append(score)

    #Double argsort returns ranks
    return np.argsort(np.argsort([-np.mean(vals) for policy, vals in sorted(agents.items())]))

class SkillRating:
    '''Rating wrapper'''
    def __init__(self):
        #1/sqrt(2)=76% win chance within beta, 95% win chance vs 3*beta=100 SR
        trueskill.setup(mu=0, sigma=2*100/3, beta=100/3, tau=2/3, draw_probability=0)
        self.ratings = [{agent.__name__: trueskill.Rating(mu=0, sigma=2*100/3)} for agent in Config.AGENTS]
        #self.ratings = {agent.__name__: openskill.Rating(mu=1e-9, sigma=2*100/3) for agent in Config.AGENTS}
        self.anchor()

    def anchor(self, anchor='Meander', mu=0, sigma=1):
        for rating_dict in self.ratings:
            for agent_name, rating in rating_dict.items():
                if agent_name == anchor:
                    rating_dict[agent_name] = openskill.Rating(mu=mu, sigma=sigma)

    def update(self, ranks):
        self.ratings = trueskill.rate(self.ratings, ranks)
        self.anchor()

    '''
    def update(self, ranks):
        ratings = list(self.ratings.values())
        teams = [[ratings[rank]] for rank in ranks]
        ratings = openskill.rate(teams)
        ratings = [openskill.create_rating(team[0]) for team in ratings]
        for agent_name, rating in zip(self.ratings, ratings):
            self.ratings[agent_name] = rating

        self.anchor()

    def anchor(self, anchor='Meander', mu=1e-9, sigma=1):
        for agent_name, rating in self.ratings.items():
            if agent_name == anchor:
                self.ratings[agent_name] = openskill.Rating(mu=mu, sigma=sigma)
    '''


def parallel_simulations(cores, horizon):
    '''Simulate environments in parallel and compute skill ratings'''
    sr = SkillRating()

    # Parallel sim using base ray
    all_stats = ray.get([simulate.remote(horizon, print_progress=worker==0) for worker in range(cores)])

    for stats in all_stats:
        ranks      = rank(stats['PolicyID'], stats['Achievement_Reward'])
        sr.update(ranks)

        print(sr.ratings)

if __name__ == '__main__':
    CORES   = 10
    HORIZON = 32

    ray.init()
    parallel_simulations(CORES, HORIZON)
