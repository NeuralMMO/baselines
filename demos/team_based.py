'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T
from collections import defaultdict
import numpy as np

import nmmo
from nmmo import Task

import tasks
from scripted import baselines

from demos import minimal


class TeamRewardEnv(nmmo.Env):
    '''Some reward functions require access to other players
    This demo computes the union of tasks across players in
    the same population (team)
    
    Override the entire step function in these cases'''
    def step(self, decisions):
        obs, rewards, dones, infos = super().step(decisions)
        team_rewards = defaultdict(lambda: defaultdict(int))
        ts           = self.config.TEAM_SPRIT

        # Aggregate rewards across population
        populations = {}
        for entID, info in infos.items():
            pop = info.pop('population')
            populations[entID] = pop
            team = team_rewards[pop]
            for task, reward in info.items():
                team[task] = max(team[task], reward)

        # Team spirit interpolated between agent and team summed task rewards
        for entID, reward in rewards.items():
            pop = populations[entID]
            rewards[entID] = ts*sum(team_rewards[pop].values()) + (1-ts)*reward

        return obs, rewards, dones, infos


class TeamConfig(minimal.Config):
    '''Assign reward 1 for the first and third kills'''
    AGENTS = [baselines.Combat]

    # Interpolate between individual and team rewards
    TEAM_SPRIT = 0.5

    # Enable all tasks
    TASKS  = tasks.All


if __name__ == '__main__':
    stats = minimal.simulate(
            TeamRewardEnv,
            TeamConfig,
            horizon=128)['Stats']

    reward   = np.mean(stats['Task_Reward'])
    complete = np.mean(stats['Tasks_Completed'])

    print(f'Task Reward: {reward:.3f}, Tasks Complete: {complete:.3f}')


