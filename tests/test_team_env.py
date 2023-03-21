import unittest
from typing import Any, Dict

import gym
import numpy as np
from pettingzoo.utils.env import ParallelEnv

from team_env import TeamEnv

class TestEnv(ParallelEnv):
  def __init__(self):
    self.action_space_map = {
      0: gym.spaces.Discrete(2),
      1: gym.spaces.Discrete(3),
      2: gym.spaces.Discrete(2),
      3: gym.spaces.Discrete(3)
    }
    self.observation_space_map = {
      0: gym.spaces.Box(low=0, high=1, shape=(2,)),
      1: gym.spaces.Box(low=0, high=1, shape=(3,)),
      2: gym.spaces.Box(low=0, high=1, shape=(2,)),
      3: gym.spaces.Box(low=0, high=1, shape=(3,))
    }

  def reset(self, **kwargs) -> Dict[int, Any]:
    return {
      0: np.array([0.1, 0.2]),
      1: np.array([0.3, 0.4, 0.5]),
      2: np.array([0.6, 0.7]),
      3: np.array([0.8, 0.9, 1.0])
    }

  def step(self, actions: Dict[int, Any]):
    obs = self.reset()
    rewards = {i: action * 0.5 for i, action in actions.items()}
    dones = {i: action == self.action_space_map[i].n - 1 for i, action in actions.items()}
    infos = {i: {} for i in actions.keys()}
    return obs, rewards, dones, infos

  def action_space(self, agent: int) -> gym.Space:
    return self.action_space_map[agent]

  def observation_space(self, agent: int) -> gym.Space:
    return self.observation_space_map[agent]


class TestTeamEnv(unittest.TestCase):
  def test_team_env(self):
    simple_env = TestEnv()
    teams = [[0, 1], [2, 3]]
    team_env = TeamEnv(simple_env, teams)

    # Test reset
    obs = team_env.reset()
    np.testing.assert_equal(obs, {0: {0: np.array([0.1, 0.2]), 1: np.array([0.3, 0.4, 0.5])},
                            1: {0: np.array([0.6, 0.7]), 1: np.array([0.8, 0.9, 1.0])}})

    # Test step
    team_actions = {0: {0: 1, 1: 2}, 1: {0: 1, 1: 1}}
    obs, rewards, dones, infos = team_env.step(team_actions)

    expected_obs = {0: {0: np.array([0.1, 0.2]), 1: np.array([0.3, 0.4, 0.5])},
                    1: {0: np.array([0.6, 0.7]), 1: np.array([0.8, 0.9, 1.0])}}
    expected_rewards = {0: 1.5, 1: 1.0}
    expected_dones = {0: True, 1: False}
    expected_infos = {0: {0: {}, 1: {}}, 1: {0: {}, 1: {}}}

    np.testing.assert_equal(obs, expected_obs)
    np.testing.assert_equal(rewards, expected_rewards)
    np.testing.assert_equal(dones, expected_dones)
    np.testing.assert_equal(infos, expected_infos)


if __name__ == '__main__':
  unittest.main()
