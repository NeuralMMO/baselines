import unittest

import nmmo
from scripted import baselines

# pylint: disable=import-error
from feature_extractor.map_helper import MapHelper
from feature_extractor.game_state import GameState
from feature_extractor.entity_helper import EntityHelper
from feature_extractor.target_tracker import TargetTracker
from feature_extractor.feature_extractor import FeatureExtractor

from team_helper import TeamHelper

from model.model import ModelArchitecture

TEST_HORIZON = 5
RANDOM_SEED = 0 # random.randint(0, 10000)


class Config(nmmo.config.Medium, nmmo.config.AllGameSystems):
  RENDER = False
  SPECIALIZE = True
  PLAYERS = [
    baselines.Fisher, baselines.Herbalist, baselines.Prospector,
    baselines.Carver, baselines.Alchemist,
    baselines.Melee, baselines.Range, baselines.Mage]


class TestMapHelper(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = Config()

    # pylint: disable=invalid-name
    # CHECK ME: this provides action targets, at the cost of performance
    #   We may want to revisit this
    cls.config.PROVIDE_ACTION_TARGETS = True

    # 16 teams x 8 players
    cls.num_team = 16
    cls.team_size = 8
    # match the team definition to the default nmmo
    teams = {team_id: [cls.num_team*j+team_id+1 for j in range(cls.team_size)]
              for team_id in range(cls.num_team)}
    cls.team_helper = TeamHelper(teams)

  def test_featurizer_init_only(self):
    # pylint: disable=unused-variable
    feature_extractors = {
        team_id: FeatureExtractor(self.team_helper.teams, team_id, self.config)
        for team_id in self.team_helper.teams
    }

  def _filter_obs(self, obs, teammates):
    flt_obs = {}
    for ent_id, ent_obs in obs.items():
      if ent_id in teammates:
        flt_obs[ent_id] = ent_obs

    return flt_obs

  def test_map_helper_shape_check_only(self):
    # init map_helper for team 1
    team_id = 1
    map_helper = MapHelper(self.config, team_id, self.team_helper)
    target_tracker = TargetTracker(self.team_helper.team_size[team_id])
    entity_helper = EntityHelper(self.config, self.team_helper, team_id,
                                 target_tracker, map_helper)
    game_state = GameState(self.config, self.team_helper.team_size[team_id])

    # init the env
    env = nmmo.Env(self.config, RANDOM_SEED)
    init_obs = env.reset()
    team_obs = self._filter_obs(init_obs, self.team_helper.teams[team_id])

    # init the helpers
    game_state.reset(team_obs)
    map_helper.reset()
    target_tracker.reset(team_obs)
    entity_helper.reset(team_obs)

    # execute step and update the featurizers
    game_state.advance()
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, self.team_helper.teams[team_id])
    game_state.update(team_obs)
    entity_helper.update(team_obs)
    map_helper.update(team_obs, game_state)

    # check extract_tile_feature() output shape
    tile_img = map_helper.extract_tile_feature(entity_helper)
    self.assertEqual(tile_img.shape, (self.team_helper.team_size[team_id],
                                      ModelArchitecture.TILE_NUM_CHANNELS,
                                      ModelArchitecture.TILE_IMG_SIZE[0],
                                      ModelArchitecture.TILE_IMG_SIZE[1]))

    # check nearyby_features() output shape
    for member_pos in range(self.team_helper.team_size[team_id]):
      nearby_feats = map_helper.nearby_features(*entity_helper.member_location[member_pos])
      self.assertTrue(len(nearby_feats) == ModelArchitecture.NEARBY_NUM_FEATURES)

    # check legal_moves() output shape
    legal_moves = map_helper.legal_moves(team_obs) # 4 dirs + 1 for no move
    self.assertEqual(legal_moves.shape, (self.team_helper.team_size[team_id],
                                         ModelArchitecture.ACTION_NUM_DIM['move']))

  # TODO: add correctness testing with actual values

if __name__ == '__main__':
  unittest.main()
