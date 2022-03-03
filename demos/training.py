from pdb import set_trace as T
import numpy as np

from fire import Fire

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
        # Default survival reward
        reward, info = super().reward(player)

        # Inject exploration attribute into player
        if not hasattr(player, 'exploration'):
            player.exploration = 0

        # Historical exploration already part of player state
        exploration = player.history.exploration

        # Only reward agents for distance increment
        # over previous farthest exploration
        if exploration > player.exploration:
            reward += 0.05 * (exploration - player.exploration)

        return reward, info

class SmallExploreRLlibEnv(SmallExploreEnv, rllib_wrapper.RLlibEnv):
    pass

class SmallExploreConfig(scale.Debug, bases.Small, nmmo.config.Resource):
    '''Config for NMMO default environment with concurrent spawns'''

    # Always trains from scratch, enable 1 GPU
    RESTORE  = False
    NUM_GPUS = 1

    # Render maps during generation for user to preview
    GENERATE_MAP_PREVIEWS = True

    # Spawn all agents at t=0
    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT


class ForageBaseline(SmallExploreConfig):
    AGENTS = [baselines.Forage]

class MeanderBaseline(SmallExploreConfig):
    AGENTS = [baselines.Meander]

class CLI:
    '''Google Fire CLI for this simple training demo. Commands: baselines, neural'''
    def baselines(self):
        '''Scripted baselines for demo'''
        lifetime = minimal.simulate(SmallExploreEnv, ForageBaseline,
                horizon=128)['Stats']['Lifetime']
        print(f'Average Scripted Forage Lifetime: {np.mean(lifetime)}')

        lifetime = minimal.simulate(SmallExploreEnv, MeanderBaseline,
                horizon=128)['Stats']['Lifetime']
        print(f'Average Scripted Meander Lifetime: {np.mean(lifetime)}')

        print('Note that these are noisy estimates on one env')

    def neural(self):
        '''Train neural model from scratch'''
        run_tune_experiment(
                SmallExploreConfig(),
                rllib_wrapper.PPO,
                rllib_env=SmallExploreRLlibEnv)

if __name__ == '__main__':
    Fire(CLI)
