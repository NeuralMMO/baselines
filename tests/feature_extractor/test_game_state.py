import unittest
import numpy as np
import nmmo
# pylint: disable=import-error
from feature_extractor.game_state import GameState
from feature_extractor.entity_helper import ATK_TYPE

class TestGameState(unittest.TestCase):
  def setUp(self):
    self.config = nmmo.config.Config()
    self.config.HORIZON = 1000
    self.team_size = 5
    self.game_state = GameState(self.config, self.team_size)

  def test_init(self):
    self.assertEqual(self.game_state.MAX_GAME_LENGTH, self.config.HORIZON)
    self.assertEqual(self.game_state.TEAM_SIZE, self.team_size)
    self.assertIsNone(self.game_state.curr_step)
    self.assertIsNone(self.game_state.curr_obs)
    self.assertIsNone(self.game_state.prev_obs)
    self.assertIsNone(self.game_state.professions)

  def test_reset(self):
    init_obs = {"player_1": "obs_1", "player_2": "obs_2"}
    self.game_state.reset(init_obs)
    self.assertEqual(self.game_state.curr_step, 0)
    self.assertEqual(self.game_state.prev_obs, init_obs)
    self.assertIsNotNone(self.game_state.professions)
    self.assertTrue(set(self.game_state.professions).issubset(set(ATK_TYPE)))
    self.assertEqual(len(self.game_state.professions), self.team_size)

  def test_update_advance(self):
    init_obs = {"player_1": "obs_1", "player_2": "obs_2"}
    self.game_state.reset(init_obs)

    obs = {"player_1": "obs_3", "player_2": "obs_4"}
    self.game_state.update(obs)
    self.assertEqual(self.game_state.curr_obs, obs)

    self.game_state.advance()
    self.assertEqual(self.game_state.prev_obs, obs)
    self.assertEqual(self.game_state.curr_step, 1)

  def test_extract_game_feature(self):
    init_obs = {"player_1": "obs_1", "player_2": "obs_2"}
    self.game_state.reset(init_obs)

    obs = {"player_1": "obs_3", "player_2": "obs_4", "player_3": "obs_5"}
    self.game_state.update(obs)

    game_features = self.game_state.extract_game_feature(obs)
    self.assertIsInstance(game_features, np.ndarray)

    expected_n_alive = len(obs.keys())
    self.assertEqual(game_features[1], expected_n_alive / self.team_size)

  def test_choose_profession(self):
    # pylint: disable=protected-access
    professions = self.game_state._choose_profession()
    self.assertEqual(len(professions), self.team_size)
    self.assertTrue(set(professions).issubset(set(ATK_TYPE)))


if __name__ == '__main__':
  unittest.main()
