import unittest
from unittest.mock import MagicMock
import numpy as np
from feature_extractor.entity_helper import EntityHelper

from feature_extractor.map_helper import MapHelper
from feature_extractor.target_tracker import TargetTracker
from team_helper import TeamHelper
import nmmo
from nmmo.entity.entity import EntityState
from nmmo.core.tile import TileState
from nmmo.lib.datastore.numpy_datastore import NumpyDatastore
from nmmo.core.observation import Observation

EntityAttr = EntityState.State.attr_name_to_col

class TestEntityHelper(unittest.TestCase):
  def setUp(self):
    self.datastore = NumpyDatastore()
    self.datastore.register_object_type("Entity", EntityState.State.num_attributes)
    self.datastore.register_object_type("Tile", TileState.State.num_attributes)
    self.config = nmmo.config.Medium()
    self.team_id = 0
    self.team_helper = TeamHelper([[range(1,5), range(5,9)]])
    self.target_tracker = TargetTracker(0, self.team_helper)
    self.target_tracker.reset({})
    self.map_helper = MagicMock(spec=MapHelper)

    self.entity_helper = EntityHelper(
        self.config,
        self.team_id,
        self.team_helper,
        self.target_tracker,
        self.map_helper
    )

  def _make_entity(self, id, row=0, col=0):
    e = EntityState(self.datastore, EntityState.Limits(self.config))
    e.id.update(id)
    e.row.update(row)
    e.col.update(col)
    return e

  def create_sample_obs(self, num_npcs):
    entites = []
    for i in np.array(self.team_helper.teams).flatten():
      entites.append(self._make_entity(i, row=2*i, col=3*i))

    for i in range(1, num_npcs):
      entites.append(self._make_entity(-i, row=2*i, col=3))

    obs = {}
    for t in range(1, self.team_helper.num_teams):
      obs[t] = {}
      for i, id in enumerate(self.team_helper.teams[t]):
        obs[t][i] = {
          "Entity": EntityState.Query.by_ids(
            self.datastore, [e.id.val for e in entites])
        }
    return obs

def test_update(self):
  obs = self.create_sample_obs(4)
  self.entity_helper.update(obs)

  self.assertEqual(len(self.entity_helper._entities), 12)
  self.assertEqual(len(self.entity_helper._member_location), 4)
  self.assertEqual(len(self.entity_helper._entity_features), 12)

def test_team_features_and_mask(self):
  obs = self.create_sample_obs(4)
  self.entity_helper.update(obs)
  team_features, team_mask = self.entity_helper.team_features_and_mask(obs)

  self.assertEqual(team_features.shape, (4, 262))
  self.assertEqual(team_mask.shape, (4,))

def test_npcs_features_and_mask(self):
  obs = self.create_sample_obs(4)
  self.entity_helper.update(obs)
  npc_features, npc_mask = self.entity_helper.npcs_features_and_mask(obs)

  self.assertEqual(npc_features.shape, (4, 9, 29))
  self.assertEqual(npc_mask.shape, (4, 9))

def test_enemies_features_and_mask(self):
  obs = self.create_sample_obs(4)
  self.entity_helper.update(obs)
  enemy_features, enemy_mask = self.entity_helper.enemies_features_and_mask(obs)

  self.assertEqual(enemy_features.shape, (4, 9, 29))
  self.assertEqual(enemy_mask.shape, (4, 9))

if __name__ == '__main__':
  unittest.main()
