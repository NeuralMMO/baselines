import unittest

import nmmo
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState

from feature_extractor.entity_helper import EntityHelper

# pylint: disable=import-error
from feature_extractor.item_helper import ItemHelper
from feature_extractor.market_helper import MarketHelper
from lib.team.team_helper import TeamHelper
from model.realikun.model import ModelArchitecture

from tests.featurizer.testhelpers import FeatureTestConfig, provide_item

ItemAttr = ItemState.State.attr_name_to_col

RANDOM_SEED = 0


# NOTE: these test cases assume a team_size of 8 agents
class TestItemHelper(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = FeatureTestConfig()
    cls.config.PLAYER_N = 128  # 16 teams x 8 agents

    # pylint: disable=invalid-name
    cls.config.PROVIDE_ACTION_TARGETS = True

    # default: 16 teams x 8 players
    cls.num_team = len(cls.config.PLAYERS)
    cls.team_size = int(cls.config.PLAYER_N / cls.num_team)

    # match the team definition to the default nmmo
    teams = {
        team_id: [cls.num_team * j + team_id + 1 for j in range(cls.team_size)]
        for team_id in range(cls.num_team)
    }
    cls.team_helper = TeamHelper(teams)

  def _filter_obs(self, obs, team_id):
    flt_obs = {}
    for ent_id, ent_obs in obs.items():
      if ent_id in self.team_helper.teams[team_id]:
        flt_obs[ent_id] = ent_obs
    return flt_obs

  # pylint: disable=protected-access
  def _create_test_env(self):
    # init map_helper for team 1
    team_id = 1
    entity_helper = EntityHelper(self.config, self.team_helper, team_id)
    item_helper = ItemHelper(self.config, entity_helper)
    market_helper = MarketHelper(self.config, entity_helper, item_helper)

    # init the env
    env = nmmo.Env(self.config, RANDOM_SEED)
    init_obs = env.reset()
    team_obs = self._filter_obs(init_obs, team_id)

    # init the helpers
    entity_helper.reset(team_obs)
    item_helper.reset()
    market_helper.reset()

    return env, team_id, entity_helper, item_helper, market_helper

  def test_extract_item_feature(self):
    env, team_id, entity_helper, item_helper, _ = self._create_test_env()

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
    self.assertEqual(
        item_types.shape, (self.team_size, self.config.ITEM_INVENTORY_CAPACITY)
    )
    self.assertEqual(
        item_arrs.shape,
        (
            self.team_size,
            self.config.ITEM_INVENTORY_CAPACITY,
            ModelArchitecture.ITEM_NUM_FEATURES,
        ),
    )

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
    expected_item_arr = [
        0.5,
        0.1,
        0.0,
        0.0,
        0.0,
        1.25,
        1.25,
        1.25,
        0.0,
        0.0,
        0.0]
    for i, e in enumerate(item_arrs[0][0]):
      self.assertAlmostEqual(e, expected_item_arr[i])

  def _assert_same_item(self, itm, item_arr):
    self.assertEqual(item_arr[ItemAttr["type_id"]], itm[0].ITEM_TYPE_ID)
    self.assertEqual(item_arr[ItemAttr["level"]], itm[1])

  def test_update_equip_weapons_armors(self):
    (
        env,
        team_id,
        entity_helper,
        item_helper,
        market_helper,
    ) = self._create_test_env()

    # set profession to melee
    entity_helper.member_professions = [
        "Melee", "Melee", "Range"] + ["Mage"] * 5

    # provide items: weapon, hats, tops, bottoms
    for member_pos in range(self.team_size):
      agent_id = entity_helper.pos_to_agent_id(member_pos)
      if member_pos < 4:
        provide_item(env.realm, agent_id, Item.Spear, 4)  # melee
        provide_item(env.realm, agent_id, Item.Spear, 9)  # melee
        provide_item(env.realm, agent_id, Item.Bow, 5)  # range
        provide_item(env.realm, agent_id, Item.Wand, 6)  # mage
      provide_item(env.realm, agent_id, Item.Hat, 4)
      provide_item(env.realm, agent_id, Item.Top, 3)
      provide_item(env.realm, agent_id, Item.Bottom, 2)
      if member_pos == 4:  # test _sell_weapons_armors_profession, hat
        provide_item(env.realm, agent_id, Item.Hat, 2)
        provide_item(env.realm, agent_id, Item.Hat, 1)
        provide_item(env.realm, agent_id, Item.Spear, 6, list_price=5)
      if member_pos == 5:  # test _sell_weapons_armors_profession, top
        provide_item(env.realm, agent_id, Item.Top, 2)
        provide_item(env.realm, agent_id, Item.Top, 1)
      if member_pos == 6:  # test _sell_weapons_armors_profession, bottom
        provide_item(env.realm, agent_id, Item.Bottom, 1)
        provide_item(env.realm, agent_id, Item.Bottom, 0)
        provide_item(env.realm, agent_id, Item.Bottom, 0, list_price=5)
      if member_pos == 7:  # test _sell_ammos
        provide_item(env.realm, agent_id, Item.Whetstone, 2)
        provide_item(env.realm, agent_id, Item.Runes, 3)

    # set levels
    env.realm.players[entity_helper.pos_to_agent_id(
        0)].skills.melee.level.update(10)
    env.realm.players[entity_helper.pos_to_agent_id(
        1)].skills.melee.level.update(7)
    env.realm.players[entity_helper.pos_to_agent_id(1)].gold.update(
        10
    )  # to buy level-6 spear
    env.realm.players[entity_helper.pos_to_agent_id(
        2)].skills.range.level.update(7)
    env.realm.players[entity_helper.pos_to_agent_id(
        3)].skills.mage.level.update(7)
    for pos in range(4, self.team_size):
      env.realm.players[
          entity_helper.pos_to_agent_id(pos)
      ].skills.fishing.level.update(8 - pos)
    env.realm.players[entity_helper.pos_to_agent_id(7)].gold.update(
        10
    )  # to buy level-0 bottom

    # step, get team obs, update the entity helper
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)
    entity_helper.update(team_obs)

    # this scenario tests
    #   item_helper._equip_weapons_armors()
    #   market_helper._calculate_combat_score()
    item_helper.update(team_obs)
    market_helper.update(team_obs, 1)  # 1 = curr_step

    # check the values
    exp_items = {
        0: (Item.Spear, 9),
        1: (Item.Spear, 4),
        2: (Item.Bow, 5),
        3: (Item.Wand, 6),
        4: (Item.Hat, 4),
        5: (Item.Top, 3),
        6: (Item.Bottom, 2),
    }

    for pos, (bw, bh, bt, bb) in enumerate(
        zip(
            item_helper.best_weapons,
            item_helper.best_hats,
            item_helper.best_tops,
            item_helper.best_bottoms,
        )
    ):
      agent_id = entity_helper.pos_to_agent_id(pos)
      obs_inv = team_obs[agent_id]["Inventory"]
      force_use_idx = item_helper.force_use_idx[pos]
      force_sell_idx = item_helper.force_sell_idx[pos]
      obs_mkt = team_obs[agent_id]["Market"]
      force_buy_idx = item_helper.force_buy_idx[pos]

      # market_helper._buy_weapons_armors(): spear, the highest priority
      if pos == 1:
        self._assert_same_item((Item.Spear, 6), obs_mkt[force_buy_idx])

      if pos < 4:
        self._assert_same_item(exp_items[pos], bw)
        # test _sell_weapons
        if pos < 2:  # melee
          self._assert_same_item(
              (Item.Bow, 5), obs_inv[force_sell_idx]
          )  # must not be spear
        else:  # sell the lowest level weapon
          self._assert_same_item((Item.Spear, 4), obs_inv[force_sell_idx])
      else:
        self.assertTrue(bw is None)

      if pos == 4:
        self._assert_same_item(exp_items[pos], bh)
        # test _sell_weapons_armors_profession, hat
        self._assert_same_item((Item.Hat, 1), obs_inv[force_sell_idx])

      if pos == 5:
        self._assert_same_item(exp_items[pos], bt)
        # test _sell_weapons_armors_profession, top
        self._assert_same_item((Item.Top, 1), obs_inv[force_sell_idx])

      if pos == 6:
        self._assert_same_item(exp_items[pos], bb)
        # test _sell_weapons_armors_profession, bottom
        self._assert_same_item((Item.Bottom, 0), obs_inv[force_sell_idx])

      if pos == 7:  # only has ammos, and not using any of these
        self.assertTrue(force_use_idx is None)
        self._assert_same_item(
            (Item.Whetstone, 2), obs_inv[force_sell_idx]
        )  # _sell_ammos
        # market_helper._buy_weapons_armors(): bottom, the highest priority
        self._assert_same_item((Item.Bottom, 0), obs_mkt[force_buy_idx])

    # check market_helper
    self.assertListEqual(
        list(market_helper._combat_score), [126, 76, 86, 96, 36, 20, 8, 0]
    )

    # DONE

  def test_update_equip_tools(self):
    (
        env,
        team_id,
        entity_helper,
        item_helper,
        market_helper,
    ) = self._create_test_env()

    # set profession to melee
    entity_helper.member_professions = ["Melee", "Range"] + ["Mage"] * 6

    # provide items: weapons, tools
    item_level = 6
    item_list = [
        Item.Spear,
        Item.Bow,
        Item.Wand,
        Item.Rod,
        Item.Gloves,
        Item.Pickaxe,
        Item.Axe,
        Item.Chisel,
    ]
    for member_pos in range(self.team_size):
      agent_id = entity_helper.pos_to_agent_id(member_pos)
      for itm in item_list:
        provide_item(env.realm, agent_id, itm, item_level)

    # set levels
    sklvl = 7
    env.realm.players[entity_helper.pos_to_agent_id(
        0)].skills.melee.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(
        1)].skills.range.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(
        2)].skills.mage.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(
        3)].skills.fishing.level.update(sklvl)
    env.realm.players[
        entity_helper.pos_to_agent_id(4)
    ].skills.herbalism.level.update(sklvl)
    env.realm.players[
        entity_helper.pos_to_agent_id(5)
    ].skills.prospecting.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(
        6)].skills.carving.level.update(sklvl)
    env.realm.players[entity_helper.pos_to_agent_id(
        7)].skills.alchemy.level.update(sklvl)

    # step, get team obs, update the entity helper
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)
    entity_helper.update(team_obs)

    # this scenario tests _equip_tools()
    item_helper.update(team_obs)
    market_helper.update(team_obs, 1)  # 1 = curr_step

    # check the values
    exp_items = {pos: (item, item_level) for pos, item in enumerate(item_list)}
    for pos, (bw, bt) in enumerate(
        zip(item_helper.best_weapons, item_helper.best_tools)
    ):
      agent_id = entity_helper.pos_to_agent_id(pos)
      obs_inv = team_obs[agent_id]["Inventory"]
      force_use_idx = item_helper.force_use_idx[pos]

      self._assert_same_item(exp_items[pos], obs_inv[force_use_idx])
      if pos < 3:  # has best weapons
        self._assert_same_item(exp_items[pos], bw)
        self.assertTrue(bt is None)
      else:  # has best tools
        self._assert_same_item(exp_items[pos], bt)
        self.assertTrue(bw is None)

    # DONE

  def test_update_sell_tools_profession(self):
    (
        env,
        team_id,
        entity_helper,
        item_helper,
        market_helper,
    ) = self._create_test_env()

    # set profession to melee
    entity_helper.member_professions = ["Melee", "Range"] + ["Mage"] * 6

    # provide items: tools, rations, poultice
    item_list = [(Item.Rod, 2), (Item.Gloves, 2), (Item.Pickaxe, 1)]
    arms_list = [(Item.Hat, 6), (Item.Wand, 3)]
    consume_list = [(Item.Ration, 1), (Item.Potion, 2)]
    for member_pos in range(self.team_size):
      agent_id = entity_helper.pos_to_agent_id(member_pos)
      if member_pos < 2:
        for itm, lvl in item_list:
          provide_item(env.realm, agent_id, itm, lvl)
      elif 2 <= member_pos < 4:
        # member_pos = 2 gets hat and wand, member_pos = 3 only gets wand
        for i in range(member_pos - 2, len(arms_list)):
          provide_item(
              env.realm, agent_id, arms_list[i][0], arms_list[i][1] + 1
          )
          provide_item(env.realm, agent_id, arms_list[i][0], arms_list[i][1])
          provide_item(
              env.realm, agent_id, arms_list[i][0], arms_list[i][1] - 1
          )
      elif member_pos in [4, 5, 6]:  # testing market_helper._restore_score
        for itm, lvl in consume_list:
          provide_item(env.realm, agent_id, itm, lvl)
      elif member_pos == 7:  # testing market_helper._emergency_buy_poultice()
        for i in range(6):
          provide_item(env.realm, agent_id, Item.Ration, 1)
          if i < 2:
            provide_item(env.realm, agent_id, Item.Potion, 1)
          else:
            provide_item(env.realm, agent_id, Item.Potion, 1, list_price=i)

    # set agent attributes
    env.realm.players[entity_helper.pos_to_agent_id(
        0)].skills.fishing.level.update(5)
    env.realm.players[entity_helper.pos_to_agent_id(0)].gold.update(
        20
    )  # _buy_consumables()
    env.realm.players[entity_helper.pos_to_agent_id(
        2)].skills.fishing.level.update(6)
    env.realm.players[entity_helper.pos_to_agent_id(
        3)].skills.fishing.level.update(6)
    # to force use poultice: 4 cannot use, 5 can
    #  also agent 4 goes into _emergency_buy_poultice()
    env.realm.players[entity_helper.pos_to_agent_id(
        4)].resources.health.update(20)
    env.realm.players[entity_helper.pos_to_agent_id(4)].gold.update(20)
    env.realm.players[entity_helper.pos_to_agent_id(
        5)].resources.health.update(20)
    env.realm.players[entity_helper.pos_to_agent_id(
        5)].skills.carving.level.update(3)
    # to force use ration
    env.realm.players[entity_helper.pos_to_agent_id(
        6)].resources.food.update(40)

    # step, get team obs, update the entity helper
    obs, _, _, _ = env.step({})
    team_obs = self._filter_obs(obs, team_id)
    entity_helper.update(team_obs)

    # this scenario tests
    #   item_helper._sell_tools(), _sell_weapons_armors_profession()
    #   market_helper._calculate_restore_score(), _emergency_buy_poultice()
    item_helper.update(team_obs)
    market_helper.update(team_obs, 1)  # 1 = curr_step

    for pos, (bt, bh, fb) in enumerate(zip(item_helper.best_tools,
                                           item_helper.best_hats, item_helper.force_buy_idx)):
      agent_id = entity_helper.pos_to_agent_id(pos)
      obs_inv = team_obs[agent_id]["Inventory"]
      force_use_idx = item_helper.force_use_idx[pos]
      force_sell_idx = item_helper.force_sell_idx[pos]
      obs_mkt = team_obs[agent_id]["Market"]

      if pos == 0:  # can use fishing rod
        # best tool to use, given skill
        self._assert_same_item(item_list[0], bt)  # Item.Rod
        self._assert_same_item(
            item_list[0], obs_inv[force_use_idx])  # Item.Rod
        # lowest-level tool Item.Pickaxe is for sale
        self._assert_same_item(item_list[2], obs_inv[force_sell_idx])
        # market_helper._buy_consumables()
        self.assertTrue(fb == 0)  # buying the cheapest poultice

      if pos == 1:  # cannot use level-2 tools
        # best tool to use, given skill
        self._assert_same_item(item_list[2], bt)  # Item.Pickaxe
        self._assert_same_item(
            item_list[2], obs_inv[force_use_idx]
        )  # Item.Pickaxe
        # lowest-level tool (lvl=2), min id Item.Rod is for sale
        self._assert_same_item(item_list[0], obs_inv[force_sell_idx])

      if pos == 2:
        self._assert_same_item(
            arms_list[0], bh
        )  # can wear level 6 hat, given skill
        self._assert_same_item(arms_list[0], obs_inv[force_use_idx])
        # _sell_weapons_armors_profession, hat, >= MAX_RESERVE_LEVEL
        self._assert_same_item((Item.Hat, 7), obs_inv[force_sell_idx])

      if pos == 3:
        # _sell_weapons_armors_profession, wand, worst level weapon of my
        # profession
        self._assert_same_item((Item.Wand, 2), obs_inv[force_sell_idx])

      # check market_helper._emergency_buy_poultice()
      if pos == 4:
        self._assert_same_item((Item.Potion, 1), obs_mkt[fb])

    # check market_helper
    self.assertListEqual(
        list(market_helper._restore_score), [0, 0, 0, 0, 2, 6, 2, 10]
    )

    # DONE


if __name__ == "__main__":
  unittest.main()
