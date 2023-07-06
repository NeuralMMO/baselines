
import nmmo
import numpy as np
from lib.model.util import multi_hot_generator

from model.realikun.model import ModelArchitecture


class GameState:
  def __init__(self, config: nmmo.config.Config, team_size: int):
    self.max_step = config.HORIZON
    self.team_size = team_size

    self.curr_step = None
    self.curr_obs = None
    self.prev_obs = None

    self.prev_atns = None

  def reset(self, init_obs):
    self.curr_step = 0
    self.curr_obs = init_obs

  def update(self, obs):
    self.prev_obs = self.curr_obs
    self.curr_obs = obs
    self.curr_step += 1

  def extract_game_feature(self, obs):
    n_progress_feat = ModelArchitecture.PROGRESS_NUM_FEATURES
    game_progress = self.curr_step / self.max_step
    n_alive = len(obs.keys())
    arr = np.array([
      game_progress,
      n_alive / self.team_size,
      *multi_hot_generator(n_feature=n_progress_feat,
                           index=int(game_progress*n_progress_feat)+1),
      *multi_hot_generator(n_feature=self.team_size, index=n_alive),
    ], dtype=np.float32)
    return arr

  def previous_actions(self):
    if self.prev_atns is None:
      atn_dim = len(ModelArchitecture.ACTION_NUM_DIM)
      return np.zeros((self.team_size, atn_dim), dtype=np.float32)

    return np.array(list(self.prev_atns.values()), dtype=np.float32).T
