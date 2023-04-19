
import nmmo
import numpy as np

from model.util import multi_hot_generator


class GameState:
  def __init__(self, config: nmmo.config.Config, team_size: int):
    self.MAX_GAME_LENGTH = config.HORIZON
    self.TEAM_SIZE = team_size

    self.curr_step = None
    self.curr_obs = None
    self.prev_obs = None

  def reset(self, init_obs):
    self.curr_step = 0
    self.prev_obs = init_obs

  def update(self, obs):
    self.prev_obs = self.curr_obs
    self.curr_obs = obs
    self.curr_step += 1

  def extract_game_feature(self, obs):
    game_progress = self.curr_step / self.MAX_GAME_LENGTH
    n_alive = len(obs.keys())
    arr = np.array([
      game_progress,
      n_alive / self.TEAM_SIZE,
      *multi_hot_generator(n_feature=16, index=int(game_progress*16)+1),
      *multi_hot_generator(n_feature=self.TEAM_SIZE, index=n_alive),
    ])
    return arr

  def previous_actions(self):
    # xcxc
    return np.zeros((self.TEAM_SIZE, 4), dtype=np.float32)
