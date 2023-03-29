import unittest
from typing import Dict
from unittest.mock import MagicMock
import numpy as np
from feature_extractor.entity_helper import EntityHelper

from feature_extractor.map_helper import MapHelper
from feature_extractor.target_tracker import TargetTracker
from team_helper import TeamHelper
import nmmo
from nmmo.entity.entity import EntityState

EntityAttr = EntityState.State.attr_name_to_col

class TestEntityHelper(unittest.TestCase):
  def setUp(self):
    self.config = nmmo.config.Medium()
    self.team_id = 0
    self.team_helper = TeamHelper([[0, 1, 2]])
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

  def create_sample_obs(self, num_entities):
    entity_obs = [np.zeros(53) for _ in range(num_entities)]
    for i, obs in enumerate(entity_obs):
      obs[EntityAttr["id"]] = i + 1
    return {
        i: {
            'Entity': entity_obs
        } for i in range(3)
    }

  def test_reset_and_update(self):
    init_obs = self.create_sample_obs(4)
    self.entity_helper.reset(init_obs)
    self.assertIsNotNone(self.entity_helper.member_professions)

    obs = self.create_sample_obs(5)
    self.entity_helper.update(obs)
    self.assertNotEqual(len(self.entity_helper._entities), 0)
    self.assertNotEqual(len(self.entity_helper._member_location), 0)
    self.assertNotEqual(len(self.entity_helper._entity_features), 0)

  def test_team_features_and_mask(self):
    obs = self.create_sample_obs(4)
    self.entity_helper.update(obs)
    team_features, team_mask = self.entity_helper.team_features_and_mask(obs)

    self.assertEqual(team_features.shape, (self.entity_helper.TEAM_SIZE, 262))
    self.assertEqual(team_mask.shape, (self.entity_helper.TEAM_SIZE,))

  def test_npcs_features_and_mask(self):
    obs = self.create_sample_obs(6)
    self.entity_helper.update(obs)
    npc_features, npc_mask = self.entity_helper.npcs_features_and_mask(obs)

    self.assertEqual(npc_features.shape, (self.entity_helper.TEAM_SIZE, 9, 29))
    self.assertEqual(npc_mask.shape, (self.entity_helper.TEAM_SIZE, 9))

  def test_enemies_features_and_mask(self):
    obs = self.create_sample_obs(7)
    self.entity_helper.update(obs)
    enemy_features, enemy_mask = self.entity_helper.enemies_features_and_mask(obs)

    self.assertEqual(enemy_features.shape, (self.entity_helper.TEAM_SIZE, 9, 29))
    self.assertEqual(enemy_mask.shape, (self.entity_helper.TEAM_SIZE, 9))

if __name__ == '__main__':
  unittest.main()
