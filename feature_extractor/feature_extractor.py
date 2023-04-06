
import nmmo
import numpy as np

from feature_extractor.entity_helper import EntityHelper
from feature_extractor.game_state import GameState
from feature_extractor.map_helper import MapHelper
from feature_extractor.stats import Stats
from feature_extractor.target_tracker import TargetTracker
from team_helper import TeamHelper


class FeatureExtractor():
  def __init__(self, config: nmmo.config.AllGameSystems, team_helper: TeamHelper, team_id: int):
    self.config = config
    self.team_id = team_id
    self.team_size = team_helper.team_size[team_id]
    self.num_teams = team_helper.num_teams
    self.team_helper = team_helper

    self.game_state = GameState(config, self.team_size)
    self.map_helper = MapHelper(config, team_id, self.team_size)
    self.target_tracker = TargetTracker(team_id, self.team_helper)
    self.stats = Stats(config, self.team_size, self.target_tracker)
    self.entity_helper = EntityHelper(
      config, team_id, self.team_helper, self.target_tracker, self.map_helper)
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

  def trans_obs(self, obs):
    self.game_state.update(obs)
    self.entity_helper.update(obs)
    self.stats.update(obs)
    self.map_helper.update(obs, self.game_state, self.entity_helper)

    # use & sell
    # self.inventory.update(obs)

    # buy
    # self.market.update(obs)

    tile = self.map_helper.extract_tile_feature(obs, self.entity_helper)
    # item_type, item = self.inventory.extract_item_features(obs)
    team, team_mask = self.entity_helper.team_features_and_mask(obs)
    npc, npc_mask = self.entity_helper.npcs_features_and_mask(obs)
    enemy, enemy_mask = self.entity_helper.enemies_features_and_mask(obs)
    # game = self.extract_game_feature(obs)

    state = {
      'tile': tile,
      # 'item_type': item_type,
      # 'item': item,
      'team': team,
      'npc': npc,
      'enemy': enemy,
      'team_mask': team_mask,
      'npc_mask': npc_mask,
      'enemy_mask': enemy_mask,
      # 'game': game,
      'legal': {
        # 'move': map.legal_move(obs),
        # 'target': self.entity_helper.legal_target(obs, self.npc_tgt, self.enemy_tgt),
        # 'use': inventory.legal_use(),
        # 'sell': inventory.legal_sell(),
      },
      # 'prev_act': self.game_state.prev_actions,
      'reset': np.array([self.game_state.curr_step == 0])  # for resetting RNN hidden,
    }
    return state
