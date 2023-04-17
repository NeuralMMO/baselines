import unittest

import nmmo
from nmmo.systems import item as Item

# pylint: disable=import-error
from feature_extractor.inventory import Inventory
from feature_extractor.entity_helper import EntityHelper

from model.model import ModelArchitecture

from tests.feature_extractor.testhelpers import FeaturizerTestTemplate, provide_item

RANDOM_SEED = 0 # random.randint(0, 10000)


class TestInventory(FeaturizerTestTemplate):
  def test_extract_item_feature(self):
    # init map_helper for team 1
    team_id = 1
    entity_helper = EntityHelper(self.config, self.team_helper, team_id)
    inventory_helper = Inventory(self.config, entity_helper)

    # init the env
    env = nmmo.Env(self.config, RANDOM_SEED)
    init_obs = env.reset()
    team_obs = self._filter_obs(init_obs, team_id)

    # init the helpers
    entity_helper.reset(team_obs)
    inventory_helper.reset()

    # provide an item
    for member_pos in range(4):
      agent_id = self.team_helper.agent_id(team_id, member_pos)
      provide_item(env.realm, agent_id, Item.Bottom, 5)

    # kill the odd pos agents
    for member_pos in range(1, self.team_size, 2):
      agent_id = self.team_helper.agent_id(team_id, member_pos)
      env.realm.players[agent_id].resources.health.update(0)

    # step and get team obs
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)

    # check the item_extractor
    entity_helper.update(team_obs)
    item_types, item_arrs = inventory_helper.extract_item_feature(team_obs)

    # check the shapes
    self.assertEqual(item_types.shape, (self.team_size,
                                        self.config.ITEM_INVENTORY_CAPACITY))
    self.assertEqual(item_arrs.shape, (self.team_size,
                                       self.config.ITEM_INVENTORY_CAPACITY,
                                       ModelArchitecture.n_item_feat))

    # item_types: check the values
    for member_pos in range(self.team_size):
      if (member_pos % 2 == 0) and (member_pos < 4):
        # the first and only item is Bottom
        self.assertEqual(item_types[member_pos][0], Item.Bottom.ITEM_TYPE_ID)
        self.assertEqual(sum(item_types[member_pos][1:]), 0)
      else:
        # agent is dead or item not provided
        self.assertEqual(sum(item_types[member_pos]), 0)

    # item_arrs: check the values
    self.assertListEqual(list(item_arrs[0][0]), # level 1 Item.Bottom
                         [0.5, 0.1, 0, 0, 0, 1.25, 1.25, 1.25, 0, 0, 0])

  def test_legal_use_consumables(self):
    pass

  def test_legal_sell_consumables(self):
    pass

if __name__ == '__main__':
  unittest.main()
