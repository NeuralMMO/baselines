import unittest

import nmmo

# pylint: disable=import-error
from feature_extractor.map_helper import MapHelper
from feature_extractor.game_state import GameState
from feature_extractor.entity_helper import EntityHelper

from model.model import ModelArchitecture

from tests.feature_extractor.testhelpers import FeaturizerTestTemplate

TEST_HORIZON = 5
RANDOM_SEED = 0 # random.randint(0, 10000)


class TestMapHelper(FeaturizerTestTemplate):
  def test_map_helper_shape_check_only(self):
    # init map_helper for team 1
    team_id = 1
    team_size = self.team_helper.team_size[team_id]
    game_state = GameState(self.config, team_size)
    entity_helper = EntityHelper(self.config, self.team_helper, team_id)
    map_helper = MapHelper(self.config, entity_helper)

    # init the env
    env = nmmo.Env(self.config, RANDOM_SEED)
    init_obs = env.reset()
    team_obs = self._filter_obs(init_obs, team_id)

    # init the helpers
    game_state.reset(team_obs)
    map_helper.reset()
    entity_helper.reset(team_obs)

    # execute step and update the featurizers
    game_state.advance()
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)
    game_state.update(team_obs)
    entity_helper.update(team_obs)
    map_helper.update(team_obs, game_state)

    # check extract_tile_feature() output shape
    tile_img = map_helper.extract_tile_feature()
    self.assertEqual(tile_img.shape, (team_size,
                                      ModelArchitecture.n_img_ch,
                                      ModelArchitecture.img_size[0],
                                      ModelArchitecture.img_size[1]))

    # check nearyby_features() output shape
    for member_pos in range(team_size):
      nearby_feats = map_helper.nearby_features(*entity_helper.member_location[member_pos])
      self.assertTrue(len(nearby_feats) == ModelArchitecture.n_nearby_feat)

    # check legal_moves() output shape
    legal_moves = map_helper.legal_moves(team_obs) # 4 dirs + 1 for no move
    self.assertEqual(legal_moves.shape, (team_size, ModelArchitecture.n_legal['move']))

  # TODO: add correctness testing with actual values

if __name__ == '__main__':
  unittest.main()
