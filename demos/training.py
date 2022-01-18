from pdb import set_trace as T
import numpy as np

import nmmo

from scripted import baselines

import tasks
import rllib_wrapper
from demos import minimal
from config import bases, scale
from main import run_tune_experiment


class SmallExploreEnv(nmmo.Env):
    '''Simple environment with task exploration'''
    def reward(self, player):
        reward, info = super().reward(player)

        if not hasattr(player, 'exploration'):
            player.exploration = 0

        exploration = player.history.exploration
        if exploration > player.exploration:
            reward += 0.05 * (exploration - player.exploration)

        return reward, info


class SmallExploreConfig(scale.Debug, bases.Small, nmmo.config.Resource):
    '''Config for NMMO default environment with concurrent spawns'''
    RESTORE  = None
    NUM_GPUS = 1

    GENERATE_MAP_PREVIEWS = True


    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT
    pass


class ForageBaseline(SmallExploreConfig):
    AGENTS = [baselines.Forage]

class MeanderBaseline(SmallExploreConfig):
    AGENTS = [baselines.Meander]


if __name__ == '__main__':
    #lifetime = minimal.simulate(SmallExploreEnv, ForageBaseline, horizon=128)['Stats']['Lifetime']
    #print(f'Average Scripted Forage Lifetime: {np.mean(lifetime)}')

    #lifetime = minimal.simulate(SmallExploreEnv, MeanderBaseline, horizon=128)['Stats']['Lifetime']
    #print(f'Average Scripted Meander Lifetime: {np.mean(lifetime)}')

    run_tune_experiment(SmallExploreConfig, rllib_wrapper.PPO)
