
import nmmo
import numpy as np

from model.util import multi_hot_generator
from model.model import ModelArchitecture


class GameState:
  def __init__(self, config: nmmo.config.Config, team_size: int):
    self.max_step = config.HORIZON
    self.team_size = team_size

    self.curr_step = None
    self.curr_obs = None
    self.prev_obs = None

  def reset(self, init_obs):
    self.curr_step = 0
    self.prev_obs = init_obs

  def update(self, obs):
    self.curr_obs = obs

  def advance(self):
    self.curr_step += 1
    self.prev_obs = self.curr_obs

  def extract_game_feature(self, obs):
    n_progress_feat = ModelArchitecture.n_progress_feat
    game_progress = self.curr_step / self.max_step
    n_alive = len(obs.keys())
    arr = np.array([
      game_progress,
      n_alive / self.team_size,
      *multi_hot_generator(n_feature=n_progress_feat,
                           index=int(game_progress*n_progress_feat)+1),
      *multi_hot_generator(n_feature=self.team_size, index=n_alive),
    ])
    return arr
