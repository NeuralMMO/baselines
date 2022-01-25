'''WIP Demo to be documented at neuralmmo.github.io'''

from pdb import set_trace as T
import numpy as np

import nmmo

import rllib_wrapper
from main import run_tune_experiment
from config import baselines


class PopulationTaskEnv(rllib_wrapper.RLlibEnv):
    def reward(self, player):
        _, info = super().reward(player)

        if player.entID not in self.realm.players:
            return -1, info

        pop = player.pop

        if __debug__:
            err = f'This is a 4 pop demo; got {pop}'
            assert pop in range(4), err

        if pop == 0:
            task = 'player_kills'
        elif pop == 1:
            task = 'equipment'
        elif pop == 2:
            task = 'exploration'
        elif pop == 3:
            task = 'foraging'

        reward = 0
        for k, v in info.items():
            if k.startswith(task):
               reward += v

        return reward, info


class PopulationTaskConfig(baselines.Medium):
    RESTORE   = False
    NPOLICIES = 4


if __name__ == '__main__':
    run_tune_experiment(
            PopulationTaskConfig(),
            rllib_wrapper.PPO,
            rllib_env=PopulationTaskEnv)
