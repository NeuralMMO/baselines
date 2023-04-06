from typing import Any, Dict
import gym

import nmmo
import numpy as np

from feature_extractor.feature_extractor import FeatureExtractor
from team_env import TeamEnv


class BaselineEnv(TeamEnv):
  def __init__(self, env, team_helper):
    assert isinstance(env, nmmo.Env)
    super().__init__(env, team_helper)
    self._feature_extractors = {
        team_id: FeatureExtractor(env.config, team_helper, team_id) for team_id in range(team_helper.num_teams)
    }

  def action_space(self, team):
    return self._env.action_space(team)

  def observation_space(self, team):
    def box(rows, cols):
      return gym.spaces.Box(
          low=-2**20, high=2**20,
          shape=(rows, cols),
          dtype=np.float32)

    return gym.spaces.Dict({
      "tile": box(1, 41425),
    })

    # return self._env.observation_space(team)

  def reset(self, **kwargs) -> Dict[int, Any]:
    obs = super().reset(**kwargs)
    for k, v in obs.items():
      self._feature_extractors[k].reset(v)
      obs[k] = self._feature_extractors[k].trans_obs(v)
    return obs

  def step(self, actions: Dict[int, Dict[str, Any]]):
    for k, v in actions.items():
      actions[k] = self._feature_extractors[k].trans_action(v)

    obs, rew, done, info = super().step(actions)
    for k, v in obs.items():
      obs[k] = self._feature_extractors[k].trans_obs(v)

    return obs, rew, done, info
