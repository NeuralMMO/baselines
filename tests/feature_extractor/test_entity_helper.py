# TODO: remove the below line
# pylint: disable=all

import unittest
from unittest.mock import MagicMock
import numpy as np
from feature_extractor.entity_helper import EntityHelper, ATK_TYPE

from feature_extractor.map_helper import MapHelper
from feature_extractor.target_tracker import TargetTracker
from team_helper import TeamHelper
import nmmo
from nmmo.entity.entity import EntityState
from nmmo.core.tile import TileState
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.core.observation import Observation

from model.model import ModelArchitecture

EntityAttr = EntityState.State.attr_name_to_col


class MockMapHelper(MapHelper):
  def nearby_features(self, row: int, col: int):
    # TODO: gather all model-related consts in one place
    return np.zeros(ModelArchitecture.n_nearby_feat)

class TestEntityHelper(unittest.TestCase):
  def setUp(self):
    self.datastore = NumpyDatastore()
    self.datastore.register_object_type("Entity", EntityState.State.num_attributes)
    self.datastore.register_object_type("Tile", TileState.State.num_attributes)
    self.config = nmmo.config.Medium()

    self.num_npcs = 4
    self.num_team = 2
    self.team_size = 4
    teams = { tid: list(range(1+tid*self.team_size, 1+(tid+1)*self.team_size))
              for tid in range(self.num_team) }

    self.team_id = 0
    self.team_helper = TeamHelper(teams)
    self.target_tracker = TargetTracker(self.team_size)
    self.target_tracker.reset({})
    self.map_helper = MockMapHelper(self.config, self.team_id, self.team_helper)

    self.entity_helper = EntityHelper(
        self.config,
        self.team_helper,
        self.team_id,
        self.target_tracker,
        self.map_helper
    )

  def _make_entity(self, id, row=0, col=0):
    e = EntityState(self.datastore, EntityState.Limits(self.config))
    e.id.update(id)
    e.row.update(row)
    e.col.update(col)
    return e

  def create_sample_obs(self, team_id, num_npcs):
    entities = []
    for i in self.team_helper.team_and_position_for_agent.keys():
      entities.append(self._make_entity(i, row=2*i, col=3*i))

    for i in range(1, num_npcs+1):
      entities.append(self._make_entity(-i, row=2*i, col=3))

    # create team obs
    obs = {}
    for eid in self.team_helper.teams[team_id]:
      obs[eid] = {
        "Entity": EntityState.Query.by_ids(
          self.datastore, [e.id.val for e in entities])}

    return obs

  def test_reset(self):
    obs = self.create_sample_obs(self.team_id, self.num_npcs)
    self.entity_helper.reset(obs)

    self.assertIsNotNone(self.entity_helper.member_professions)
    self.assertTrue(set(self.entity_helper.member_professions).issubset(set(ATK_TYPE)))
    self.assertEqual(len(self.entity_helper.member_professions), self.team_size)

  def test_update(self):
    obs = self.create_sample_obs(self.team_id, self.num_npcs)
    self.entity_helper.reset(obs)

    num_ent = self.num_team * self.team_size + self.num_npcs

    self.assertEqual(len(self.entity_helper._entities), num_ent)
    self.assertEqual(len(self.entity_helper.member_location), self.team_size)
    self.assertEqual(len(self.entity_helper._entity_features), num_ent)
    self.assertEqual(len(self.entity_helper._entity_features[1]),
                     ModelArchitecture.n_ent_feat)

  def test_team_features_and_mask(self):
    obs = self.create_sample_obs(self.team_id, self.num_npcs)
    self.entity_helper.reset(obs)
    
    team_features, team_mask = self.entity_helper.team_features_and_mask()

    # n_player_feat = n_ent_feat + n_team + n_player_per_team 
    #                   + n_atk_type + n_nearby_feat
    n_feat = ModelArchitecture.n_ent_feat + \
             self.num_team + self.team_size + \
             ModelArchitecture.n_atk_type + \
             ModelArchitecture.n_nearby_feat

    self.assertEqual(team_features.shape, (self.team_size, n_feat))
    self.assertEqual(team_mask.shape, (self.team_size,))

  def test_npcs_features_and_mask(self):
    obs = self.create_sample_obs(self.team_id, self.num_npcs)
    self.entity_helper.reset(obs)
    npc_features, npc_mask = self.entity_helper.npcs_features_and_mask()

    self.assertEqual(npc_features.shape, (self.team_size,
                                          ModelArchitecture.n_npc_considered,
                                          ModelArchitecture.n_ent_feat))
    self.assertEqual(npc_mask.shape, (self.team_size,
                                      ModelArchitecture.n_npc_considered))

  def test_enemies_features_and_mask(self):
    obs = self.create_sample_obs(self.team_id, self.num_npcs)
    self.entity_helper.reset(obs)
    enemy_features, enemy_mask = self.entity_helper.enemies_features_and_mask()

    self.assertEqual(enemy_features.shape, (self.team_size,
                                            ModelArchitecture.n_enemy_considered,
                                            ModelArchitecture.n_ent_feat))
    self.assertEqual(enemy_mask.shape, (self.team_size,
                                        ModelArchitecture.n_enemy_considered))

  def test_choose_professions(self):
    # pylint: disable=protected-access
    self.entity_helper._choose_professions()
    professions = self.entity_helper.member_professions
    self.assertEqual(len(professions), self.team_size)
    self.assertTrue(set(professions).issubset(set(ATK_TYPE)))


if __name__ == '__main__':
  unittest.main()
