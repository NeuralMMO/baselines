import unittest

import nmmo

from feature_extractor.entity_helper import EntityHelper
from feature_extractor.game_state import GameState

# pylint: disable=import-error
from feature_extractor.map_helper import MapHelper
from model.realikun.model import ModelArchitecture

from tests.featurizer.testhelpers import FeaturizerTestTemplate

TEST_HORIZON = 5
RANDOM_SEED = 0  # random.randint(0, 10000)


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
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)
    game_state.update(team_obs)
    entity_helper.update(team_obs)
    map_helper.update(team_obs, game_state.curr_step)

    # check extract_tile_feature() output shape
    tile_img = map_helper.extract_tile_feature()
    self.assertEqual(
        tile_img.shape,
        (
            team_size,
            ModelArchitecture.TILE_NUM_CHANNELS,
            ModelArchitecture.TILE_IMG_SIZE[0],
            ModelArchitecture.TILE_IMG_SIZE[1],
        ),
    )

    # check nearyby_features() output shape
    for member_pos in range(team_size):
      nearby_feats = map_helper.nearby_features(
          *entity_helper.member_location[member_pos]
      )
      self.assertTrue(len(nearby_feats) ==
                      ModelArchitecture.NEARBY_NUM_FEATURES)

    # check legal_moves() output shape
    legal_moves = map_helper.legal_moves(team_obs)  # 4 dirs + 1 for no move
    self.assertEqual(
        legal_moves.shape, (team_size,
                            ModelArchitecture.ACTION_NUM_DIM["move"])
    )

  # TODO: add correctness testing with actual values


if __name__ == "__main__":
  unittest.main()
