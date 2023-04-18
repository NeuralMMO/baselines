import unittest

import nmmo
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState

# pylint: disable=import-error
from feature_extractor.item_helper import ItemHelper
from feature_extractor.entity_helper import EntityHelper

from model.model import ModelArchitecture

from tests.feature_extractor.testhelpers import FeaturizerTestTemplate, provide_item

ItemAttr = ItemState.State.attr_name_to_col

RANDOM_SEED = 0 # random.randint(0, 10000)

# pylint: disable=protected-access
class TestItemHelper(FeaturizerTestTemplate):
  def _create_test_env(self):
    # init map_helper for team 1
    team_id = 1
    entity_helper = EntityHelper(self.config, self.team_helper, team_id)
    item_helper = ItemHelper(self.config, entity_helper)

    # init the env
    env = nmmo.Env(self.config, RANDOM_SEED)
    init_obs = env.reset()
    team_obs = self._filter_obs(init_obs, team_id)

    # init the helpers
    entity_helper.reset(team_obs)
    item_helper.reset()

    return env, team_id, entity_helper, item_helper

  def test_extract_item_feature(self):
    env, team_id, entity_helper, item_helper = self._create_test_env()

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
    item_helper.update(team_obs)
    item_types, item_arrs = item_helper.extract_item_feature()

    # check the shapes
    self.assertEqual(item_types.shape, (self.team_size,
                                        self.config.ITEM_INVENTORY_CAPACITY))
    self.assertEqual(item_arrs.shape, (self.team_size,
                                       self.config.ITEM_INVENTORY_CAPACITY,
                                       ModelArchitecture.ITEM_NUM_FEATURES))

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
    expected_item_arr = [0.5, 0.1, 0., 0., 0., 1.25, 1.25, 1.25, 0., 0., 0.]
    for i, e in enumerate(item_arrs[0][0]):
      self.assertAlmostEqual(e, expected_item_arr[i])

  def _assert_same_item(self, itm, item_arr):
    self.assertEqual(item_arr[ItemAttr['type_id']], itm[0].ITEM_TYPE_ID)
    self.assertEqual(item_arr[ItemAttr['level']], itm[1])

  def test_update_equip_weapons_armors(self):
    env, team_id, entity_helper, item_helper = self._create_test_env()

    # set profession to melee
    entity_helper.member_professions = ['Melee', 'Melee', 'Range'] + ['Mage']*5

    # provide items: weapon, hats, tops, bottoms
    for member_pos in range(self.team_size):
      agent_id = entity_helper.pos_to_agent_id(member_pos)
      if member_pos < 4:
        provide_item(env.realm, agent_id, Item.Sword, 4) # melee
        provide_item(env.realm, agent_id, Item.Sword, 9) # melee
        provide_item(env.realm, agent_id, Item.Bow, 5) # range
        provide_item(env.realm, agent_id, Item.Wand, 6) # mage
      provide_item(env.realm, agent_id, Item.Hat, 4)
      provide_item(env.realm, agent_id, Item.Top, 3)
      provide_item(env.realm, agent_id, Item.Bottom, 2)
      if member_pos == 4: # test _sell_weapons_armors_profession, hat
        provide_item(env.realm, agent_id, Item.Hat, 2)
        provide_item(env.realm, agent_id, Item.Hat, 1)
      if member_pos == 5: # test _sell_weapons_armors_profession, top
        provide_item(env.realm, agent_id, Item.Top, 2)
        provide_item(env.realm, agent_id, Item.Top, 1)
      if member_pos == 6: # test _sell_weapons_armors_profession, bottom
        provide_item(env.realm, agent_id, Item.Bottom, 1)
        provide_item(env.realm, agent_id, Item.Bottom, 0)
      if member_pos == 7: # test _sell_ammos
        provide_item(env.realm, agent_id, Item.Scrap, 2)
        provide_item(env.realm, agent_id, Item.Shard, 3)

    # set levels
    env.realm.players[entity_helper.pos_to_agent_id(0)].skills.melee.level.update(10)
    env.realm.players[entity_helper.pos_to_agent_id(1)].skills.melee.level.update(7)
    env.realm.players[entity_helper.pos_to_agent_id(2)].skills.range.level.update(7)
    env.realm.players[entity_helper.pos_to_agent_id(3)].skills.mage.level.update(7)
    for pos in range(4, self.team_size):
      env.realm.players[entity_helper.pos_to_agent_id(pos)].skills.fishing.level.update(8-pos)

    # step, get team obs, update the entity helper
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)
    entity_helper.update(team_obs)

    # this scenario tests _equip_weapons_armors()
    item_helper.update(team_obs)

    # check the values
    exp_items = { 0: (Item.Sword, 9), 1: (Item.Sword, 4), 2: (Item.Bow, 5),
      3: (Item.Wand, 6), 4: (Item.Hat, 4), 5: (Item.Top, 3), 6:(Item.Bottom, 2) }

    for pos, (bw, bh, bt, bb) in enumerate(zip(item_helper._best_weapons,
                                             item_helper._best_hats,
                                             item_helper._best_tops,
                                             item_helper._best_bottoms)):
      agent_id = entity_helper.pos_to_agent_id(pos)
      obs_inv = team_obs[agent_id]['Inventory']
      force_use_idx = item_helper._force_use_idx[pos]
      force_sell_idx = item_helper._force_sell_idx[pos]

      if pos < 4:
        self._assert_same_item(exp_items[pos], bw)
        # test _sell_weapons
        if pos < 2: # melee
          self._assert_same_item((Item.Bow, 5), obs_inv[force_sell_idx]) # must not be sword
        else: # sell the lowest level weapon
          self._assert_same_item((Item.Sword, 4), obs_inv[force_sell_idx])
      else:
        self.assertTrue(bw is None)

      if pos == 4:
        self._assert_same_item(exp_items[pos], bh)
        # test _sell_weapons_armors_profession, hat
        self._assert_same_item((Item.Hat, 1), obs_inv[force_sell_idx])
      else:
        self.assertTrue(bh is None)

      if pos == 5:
        self._assert_same_item(exp_items[pos], bt)
        # test _sell_weapons_armors_profession, top
        self._assert_same_item((Item.Top, 1), obs_inv[force_sell_idx])
      else:
        self.assertTrue(bt is None)

      if pos == 6:
        self._assert_same_item(exp_items[pos], bb)
        # test _sell_weapons_armors_profession, bottom
        self._assert_same_item((Item.Bottom, 0), obs_inv[force_sell_idx])
      else:
        self.assertTrue(bb is None)

      if pos == 7: # only has ammos, and not using any of these
        self.assertTrue(force_use_idx is None)
        self._assert_same_item((Item.Scrap, 2), obs_inv[force_sell_idx]) # _sell_ammos
      else:
        self._assert_same_item(exp_items[pos], obs_inv[force_use_idx])

    # DONE

  def test_update_equip_tools(self):
    env, team_id, entity_helper, item_helper = self._create_test_env()

    # set profession to melee
    entity_helper.member_professions = ['Melee', 'Range'] + ['Mage']*6

    # provide items: weapons, tools
    item_level = 6
    item_list = [Item.Sword, Item.Bow, Item.Wand, Item.Rod,
                 Item.Gloves, Item.Pickaxe, Item.Chisel, Item.Arcane]
    for member_pos in range(self.team_size):
      agent_id = entity_helper.pos_to_agent_id(member_pos)
      for itm in item_list:
        provide_item(env.realm, agent_id, itm, item_level)

    # set levels
    sklvl = 7
    env.realm.players[entity_helper.pos_to_agent_id(0)].skills.melee.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(1)].skills.range.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(2)].skills.mage.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(3)].skills.fishing.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(4)].skills.herbalism.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(5)].skills.prospecting.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(6)].skills.carving.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(7)].skills.alchemy.level.update(sklvl)

    # step, get team obs, update the entity helper
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)
    entity_helper.update(team_obs)

    # this scenario tests _equip_tools()
    item_helper.update(team_obs)

    # check the values
    exp_items = { pos: (item, item_level)
                 for pos, item in enumerate(item_list) }
    for pos, (bw, bt) in enumerate(zip(item_helper._best_weapons,
                                       item_helper._best_tools)):
      agent_id = entity_helper.pos_to_agent_id(pos)
      obs_inv = team_obs[agent_id]['Inventory']
      force_use_idx = item_helper._force_use_idx[pos]

      self._assert_same_item(exp_items[pos], obs_inv[force_use_idx])
      if pos < 3: # has best weapons
        self._assert_same_item(exp_items[pos], bw)
        self.assertTrue(bt is None)
      else: # has best tools
        self._assert_same_item(exp_items[pos], bt)
        self.assertTrue(bw is None)

    # DONE

  def test_update_sell_tools_profession(self):
    env, team_id, entity_helper, item_helper = self._create_test_env()

    # set profession to melee
    entity_helper.member_professions = ['Melee', 'Range'] + ['Mage']*6

    # provide items: tools, rations, poultice
    item_list = [(Item.Rod, 1), (Item.Gloves, 1), (Item.Pickaxe, 0)]
    arms_list = [(Item.Hat, 6), (Item.Wand, 3)]
    consume_list = [(Item.Ration, 0), (Item.Poultice, 1)]
    for member_pos in range(self.team_size):
      agent_id = entity_helper.pos_to_agent_id(member_pos)
      if member_pos < 2:
        for itm, lvl in item_list:
          provide_item(env.realm, agent_id, itm, lvl)
      elif 2 <= member_pos < 4:
        # member_pos = 2 gets hat and wand, member_pos = 3 only gets wand
        for i in range(member_pos-2, len(arms_list)):
          provide_item(env.realm, agent_id, arms_list[i][0], arms_list[i][1]+1)
          provide_item(env.realm, agent_id, arms_list[i][0], arms_list[i][1])
          provide_item(env.realm, agent_id, arms_list[i][0], arms_list[i][1]-1)
      elif member_pos in [4, 5, 6]: # testing legal_use_consumables
        for itm, lvl in consume_list:
          provide_item(env.realm, agent_id, itm, lvl)
      elif member_pos == 7: # testing legal_sell_consumables: ration
        for _ in range(6):
          provide_item(env.realm, agent_id, Item.Ration, 0)
          provide_item(env.realm, agent_id, Item.Poultice, 0)

    # set agent attributes
    env.realm.players[entity_helper.pos_to_agent_id(0)].skills.fishing.level.update(5)
    env.realm.players[entity_helper.pos_to_agent_id(2)].skills.fishing.level.update(6)
    env.realm.players[entity_helper.pos_to_agent_id(3)].skills.fishing.level.update(6)
    # to force use poultice: 4 cannot use, 5 can
    env.realm.players[entity_helper.pos_to_agent_id(4)].resources.health.update(40)
    env.realm.players[entity_helper.pos_to_agent_id(5)].resources.health.update(40)
    env.realm.players[entity_helper.pos_to_agent_id(5)].skills.carving.level.update(3)
    # to force use ration
    env.realm.players[entity_helper.pos_to_agent_id(6)].resources.food.update(40)

    # step, get team obs, update the entity helper
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)
    entity_helper.update(team_obs)

    # this scenario tests _sell_tools(), _sell_weapons_armors_profession()
    #   legal_use_consumables(), legal_sell_consumables()
    item_helper.update(team_obs)
    legal_use = item_helper.legal_use_consumables()
    legal_sell = item_helper.legal_sell_consumables()

    for pos, (bt, bh) in enumerate(zip(item_helper._best_tools,
                                       item_helper._best_hats)):
      agent_id = entity_helper.pos_to_agent_id(pos)
      obs_inv = team_obs[agent_id]['Inventory']
      force_use_idx = item_helper._force_use_idx[pos]
      force_sell_idx = item_helper._force_sell_idx[pos]

      if pos == 0: # can use fishing rod
        # best tool to use, given skill
        self._assert_same_item(item_list[0], bt) # Item.Rod
        self._assert_same_item(item_list[0], obs_inv[force_use_idx]) # Item.Rod
        # lowest-level tool Item.Pickaxe is for sale
        self._assert_same_item(item_list[2], obs_inv[force_sell_idx])

      if pos == 1: # cannot use level-1 tools
        # best tool to use, given skill
        self._assert_same_item(item_list[2], bt) # Item.Pickaxe
        self._assert_same_item(item_list[2], obs_inv[force_use_idx]) # Item.Pickaxe
        # lowest-level tool (lvl=1), min id Item.Rod is for sale
        self._assert_same_item(item_list[0], obs_inv[force_sell_idx])

      if pos == 2:
        self._assert_same_item(arms_list[0], bh) # can wear level 6 hat, given skill
        self._assert_same_item(arms_list[0], obs_inv[force_use_idx])
        # _sell_weapons_armors_profession, hat, >= MAX_RESERVE_LEVEL
        self._assert_same_item((Item.Hat, 7), obs_inv[force_sell_idx])

      if pos == 3:
        # _sell_weapons_armors_profession, wand, worst level weapon of my profession
        self._assert_same_item((Item.Wand, 2), obs_inv[force_sell_idx])

      # check legal_use
      if pos == 5: # pos == 4 cannot use poultice due to low level
        self.assertListEqual(list(legal_use[pos]), [1, 0, 0])
      elif pos == 6:
        self.assertListEqual(list(legal_use[pos]), [0, 1, 0])
      else:
        self.assertListEqual(list(legal_use[pos]), [0, 0, 1])

      # check legal sell
      if pos == 7:
        self.assertListEqual(list(legal_sell[pos]), [1, 1, 0])
      else:
        self.assertListEqual(list(legal_sell[pos]), [0, 0, 1])

    # DONE


if __name__ == '__main__':
  unittest.main()
