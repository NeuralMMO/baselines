from typing import Any, Dict

import nmmo
from feature_extractor import FeatureExtractor
from team_env import TeamEnv
import collections
import copy

import nmmo.io.action as nmmo_act
import numpy as np

from model.const import *
from model.util import one_hot_generator, multi_hot_generator
from pettingzoo.utils.env import AgentID, ParallelEnv


class BaselineEnv(TeamEnv):
  def __init__(self, env, teams):
    assert isinstance(env, nmmo.Env)
    super().__init__(env, teams)
    self._feature_extractors = {
        team_id: FeatureExtractor(env.config, team_id) for team_id in range(len(teams))
    }

  def action_space(self, team):
    return self._env.action_space(team)

  def observation_space(self, team):
    return self._env.observation_space(team)

  def reset(self, **kwargs) -> Dict[int, Any]:
    obs = super().reset(**kwargs)
    for k, v in obs.items():
      self._feature_extractors[k].reset(v)
      obs[k] = self._feature_extractors[k].trans_obs(v)

  def step(self, actions: Dict[int, Dict[str, Any]]):
    for k, v in actions.items():
      actions[k] = self._feature_extractors[k].trans_action(v)

    obs, rew, done, info = super().step(actions)
    for k, v in obs.items():
      obs[k] = self._feature_extractors[k].trans_obs(v)

    return obs, rew, done, info
