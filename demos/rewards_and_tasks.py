'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T
import numpy as np

import nmmo
from nmmo import Task

from scripted import baselines

from demos import minimal


class PlayerKillRewardEnv(nmmo.Env):
    '''Reward 0.1 per player defeated, -1 for death'''
    def reward(self, player):
        # Default -1 reward for death
        # Infos returns per-task rewards
        reward, info = super().reward(player)

        # Inject new attribute
        if not hasattr(player, 'kills'):
            player.kills = 0

        # Historical kills already in player state
        kills = player.history.playerKills

        # Only reward for new kills
        if kills > player.kills:
            reward += 0.1 * (kills - player.kills)

        return reward, info


def player_kills(realm, player):
    '''Total number of players defeated'''
    return player.history.playerKills


class PlayerKillTaskConfig(minimal.Config):
    '''Assign reward 1 for the first and third kills'''
    AGENTS = [baselines.Combat]

    # Task params: reward fn, score to complete, completion reward
    TASKS  = [Task(player_kills, target=1, reward=2),
              Task(player_kills, target=3, reward=2)]


if __name__ == '__main__':
    stats = minimal.simulate(
            PlayerKillRewardEnv,
            PlayerKillTaskConfig,
            horizon=128)['Stats']

    reward   = np.mean(stats['Task_Reward'])
    complete = np.mean(stats['Tasks_Completed'])

    print(f'Task Reward: {reward:.3f}, Tasks Complete: {complete:.3f}')


