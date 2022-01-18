'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T
import numpy as np

import nmmo
from nmmo import Task

from scripted import baselines

from demos import minimal


class PlayerKillRewardEnv(nmmo.Env):
    '''Assigns a reward of 0.1 per player defeated and the default -1 reward for dying'''
    def reward(self, player):
        reward, info = super().reward(player)

        if not hasattr(player, 'kills'):
            player.kills = 0

        kills = player.history.playerKills
        if kills > player.kills:
            reward += 0.1 * (kills - player.kills)

        return reward, info


def player_kills(realm, player):
    '''Total number of players defeated'''
    return player.history.playerKills


class PlayerKillTaskConfig(minimal.Config):
    '''Assign reward 1 for the first and third kills'''
    TASKS  = [Task(player_kills, target=1, reward=2), Task(player_kills, target=3, reward=2)]
    AGENTS = [baselines.Combat]


if __name__ == '__main__':
    stats    = minimal.simulate(PlayerKillRewardEnv, PlayerKillTaskConfig, horizon=128)['Stats']
    reward   = np.mean(stats['Task_Reward'])
    complete = np.mean(stats['Task_Completed'])

    print(f'Task Reward: {reward:.3f}, Tasks Complete: {complete:.3f}')


