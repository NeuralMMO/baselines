from typing import Any, Dict

import numpy as np
import gym
import nmmo

from feature_extractor.feature_extractor import FeatureExtractor
from team_env import TeamEnv


class BaselineEnv(TeamEnv):
  def __init__(self, env, team_helper):
    assert isinstance(env, nmmo.Env)
    super().__init__(env, team_helper)
    self._feature_extractors = {
        team_id: FeatureExtractor(env.config, team_helper, team_id) for team_id in range(team_helper.num_teams)
    }

    ob = self.reset()[0]
    ob = sorted((k, v) for k, v in ob.items())
    self._observation_space = gym.spaces.Dict({
        k: gym.spaces.Box(
          low=-2**20, high=2**20,
          shape=v.shape,
          dtype=np.float32)
        for k, v in ob if len(v)
    })

  def observation_space(self, team):
    return self._observation_space
    return self._env.observation_space(team)

  def reset(self, **kwargs) -> Dict[int, Any]:
    obs = super().reset(**kwargs)
    for k, v in obs.items():
      self._feature_extractors[k].reset(v)
      obs[k] = self._feature_extractors[k].trans_obs(v)
    return obs

  def step(self, actions: Dict[int, Dict[str, Any]]):
    #for k, v in actions.items():
    #  actions[k] = self._feature_extractors[k].trans_action(v)

    obs, rew, done, info = super().step(actions)
    for k, v in obs.items():
      obs[k] = self._feature_extractors[k].trans_obs(v)

    return obs, rew, done, info
