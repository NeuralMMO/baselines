
import nmmo
import numpy as np

import pufferlib.emulation
from feature_extractor import map_helper

from feature_extractor.entity_helper import EntityHelper
from feature_extractor.game_state import GameState
from feature_extractor.map_helper import MapHelper
from feature_extractor.stats import Stats
from feature_extractor.target_tracker import TargetTracker
from model.model import ModelArchitecture

from team_helper import TeamHelper

class FeatureExtractor(pufferlib.emulation.Featurizer):
  def __init__(self, teams, team_id: int, config: nmmo.config.AllGameSystems):
    super().__init__(teams, team_id)
    self._config = config

    self._team_id = team_id
    self._team_helper = TeamHelper(teams)
    team_size = self._team_helper.team_size[team_id]

    self.game_state = GameState(config, team_size)
    self.map_helper = MapHelper(config, team_id, self._team_helper)
    self.target_tracker = TargetTracker(self.team_size)
    self.stats = Stats(config, self.team_size, self.target_tracker)

    self.entity_helper = EntityHelper(
      config,
      self._team_helper, team_id,
      self.target_tracker,
      self.map_helper
    )

    # self.inventory = Inventory(config)
    # self.market = Market(config)

  def reset(self, init_obs):
    self.game_state.reset(init_obs)
    self.map_helper.reset()
    self.target_tracker.reset(init_obs)
    self.stats.reset()
    self.entity_helper.reset(init_obs)
    # self.inventory.reset()
    # self.market.reset()

  def __call__(self, obs, step):
    self.game_state.update(obs)
    self.entity_helper.update(obs)
    self.stats.update(obs)
    self.map_helper.update(obs, self.game_state)

    # use & sell
    # self.inventory.update(obs)

    # buy
    # self.market.update(obs)

    tile = self.map_helper.extract_tile_feature(self.entity_helper)

    # item_type, item = self.inventory.extract_item_features(obs)
    item_type = np.zeros((self.team_size, 1), dtype=np.float32)
    item = np.zeros((self.team_size, 1, ModelArchitecture.ITEM_NUM_FEATURES), dtype=np.float32)

    team, team_mask = self.entity_helper.team_features_and_mask()
    npc, npc_mask = self.entity_helper.npcs_features_and_mask()
    enemy, enemy_mask = self.entity_helper.enemies_features_and_mask()

    game = self.game_state.extract_game_feature(obs)

    state = {
      'tile': tile,
      'item_type': item_type,
      'item': item,
      'team': team,
      'npc': npc,
      'enemy': enemy,
      'team_mask': team_mask,
      'npc_mask': npc_mask,
      'enemy_mask': enemy_mask,
      'game': game,
      'legal': {
        'move': self.map_helper.legal_moves(obs)[:,0:4],
        # 'target': np.zeros((self.team_size, 19), dtype=np.float32),
        # 'use': np.zeros((self.team_size, 3), dtype=np.float32),
        # 'sell': np.zeros((self.team_size, 3), dtype=np.float32),

        # xcxc
        # 'target': self.entity_helper.legal_target(obs, self.npc_tgt, self.enemy_tgt),
        # 'use': inventory.legal_use(),
        # 'sell': inventory.legal_sell(),
      },
      'prev_act': self.game_state.previous_actions(),
      'reset': np.array([self.game_state.curr_step == 0])  # for resetting RNN hidden,
    }
    return state
