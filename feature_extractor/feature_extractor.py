import nmmo
import numpy as np

from feature_extractor.entity_helper import EntityHelper
from feature_extractor.game_state import GameState
from feature_extractor.map_helper import MapHelper
from feature_extractor.item_helper import ItemHelper
from feature_extractor.market_helper import MarketHelper
from feature_extractor.stat_helper import StatHelper
from model.realikun.model import ModelArchitecture

from team_helper import TeamHelper

class FeatureExtractor():
  def __init__(self, teams, team_id: int, config: nmmo.config.AllGameSystems):
    # super().__init__(teams, team_id)
    self._config = config

    self._team_helper = TeamHelper(teams)
    self._team_id = team_id
    self.team_size = self._team_helper.team_size[team_id]

    self.game_state = GameState(config, self.team_size)

    self.entity_helper = EntityHelper(config, self._team_helper, team_id)
    self.stat_helper = StatHelper(config, self.entity_helper)

    self.map_helper = MapHelper(config, self.entity_helper)

    self.item_helper = ItemHelper(config, self.entity_helper)
    self.market_helper = MarketHelper(config, self.entity_helper, self.item_helper)

    # force_action = True overrides the policy outputs with the featurizer's decisions
    # TODO: check if using featurizer's decisions is actually better
    self.force_action = True

  def reset(self, init_obs):
    self.game_state.reset(init_obs)
    self.map_helper.reset()
    self.stat_helper.reset()
    self.entity_helper.reset(init_obs)
    self.item_helper.reset()
    self.market_helper.reset()

  def __call__(self, obs):
    # NOTE: these updates needs to be in this precise order
    self.game_state.update(obs)
    self.entity_helper.update(obs)
    self.map_helper.update(obs, self.game_state.curr_step)

    self.item_helper.update(obs) # use & sell
    self.market_helper.update(obs, self.game_state.curr_step) # buy

    # CHECK ME: we can get better stat from the event log. Do we need stat_helper?
    self.stat_helper.update(obs)

    # tile dim: (team_size, TILE_NUM_CHANNELS, *TILE_IMG_SIZE)
    tile = self.map_helper.extract_tile_feature()

    # item_type dim: (team_size, config.ITEM_INVENTORY_CAPACITY)
    # item dim: (team_size, config.ITEM_INVENTORY_CAPACITY, ITEM_NUM_FEATURES)
    item_type, item = self.item_helper.extract_item_feature()

    # team_mask dim: (team_size)
    team, team_mask = self.entity_helper.team_features_and_mask(self.map_helper)

    # game dim: (GAME_NUM_FEATURES)
    game = self.game_state.extract_game_feature(obs)

    state = {
      'tile': tile,

      'team': team,
      'team_mask': team_mask,

      'item_type': item_type,
      'item': item,

      # npc dim: (team_size, ENTITY_NUM_NPCS_CONSIDERED, ENTITY_NUM_FEATURES)
      'npc': self.entity_helper.npc_features,
      # npc_mask dim: (team_size, ENTITY_NUM_NPCS_CONSIDERED)
      'npc_mask': self.entity_helper.npc_mask,

      # enemy dim: (team_size, ENTITY_NUM_ENEMIES_CONSIDERED, ENTITY_NUM_FEATURES)
      'enemy': self.entity_helper.enemy_features,
      # enemy_mask dim: (team_size, ENTITY_NUM_ENEMIES_CONSIDERED)
      'enemy_mask': self.entity_helper.enemy_mask,

      'game': game,
      'legal': self._make_legal_moves(obs),
      'prev_act': self.game_state.previous_actions(),
      'reset': np.array([self.game_state.curr_step == 0]),  # for resetting RNN hidden,
    }

    return state

  def _make_legal_moves(self, obs):
    legal_moves = {
      action: np.zeros((self.team_size, dim), dtype=np.float32)
        for action, dim in ModelArchitecture.ACTION_NUM_DIM.items()
    }

    # move
    legal_moves["move"] = self.map_helper.legal_moves(obs)

    # attack
    legal_moves["style"] = np.ones((self.team_size,
                                       ModelArchitecture.ACTION_NUM_DIM["style"]))
    legal_moves["target"] = self.entity_helper.legal_target()

    # use, destroy
    if self._config.ITEM_SYSTEM_ENABLED:
      legal_moves["use"] = self.item_helper.legal_inventory(obs, nmmo.action.Use)
      legal_moves["destroy"] = self.item_helper.legal_inventory(obs, nmmo.action.Destroy)

    if self._config.EXCHANGE_SYSTEM_ENABLED:
      legal_moves["sell"] = self.item_helper.legal_inventory(obs, nmmo.action.Sell)
      # legal_moves["buy"] = self.market_helper.legal_buy(obs)

    # TODO: give, give-gold
    #   give-gold and sell should benefit from a continuous price policy head
    #   rather than a discrete action head (e.g., one-hot encoding out of 100 discrete prices)

    return legal_moves

  # pylint: disable=unsubscriptable-object
  def translate_actions(self, actions):
    # save actions to game_state.prev_atns, assuming this fn is called at the step
    self.game_state.prev_atns = actions

    key_to_action = {
      "use": nmmo.action.Use,
      "destroy": nmmo.action.Destroy,
      'sell': nmmo.action.Sell, }
    trans_actions = {}
    for member_pos in range(self.team_size):
      # NOTE: these keys are defined in ModelArchitecture.ACTION_NUM_DIM
      # 'move': nmmo.action.Move
      if "move" in actions:
        trans_actions[member_pos] = {
          nmmo.action.Move: {
            nmmo.action.Direction:
              nmmo.action.Direction.edges[actions['move'][member_pos]] }}

      # 'target', 'style': nmmo.action.Attack
      if "target" in actions:
        target_id = self.entity_helper.set_attack_target(member_pos,
                                                         actions['target'][member_pos])
        if target_id != 0:
          trans_actions[member_pos][nmmo.action.Attack] = {
            nmmo.action.Target: target_id,
            nmmo.action.Style: nmmo.action.Style.edges[actions['style'][member_pos]] }

      # 'use': is overrided by item_helper.force_use_idx, if force_action = True
      # 'destroy': is entirely from the policy
      if self._config.ITEM_SYSTEM_ENABLED:
        for key in ['use', 'destroy']:
          if key in actions:
            inv_idx = actions[key][member_pos]
            if self.item_helper.in_inventory(member_pos, inv_idx):
              trans_actions[member_pos][key_to_action[key]] = {
                nmmo.action.InventoryItem: inv_idx }

        # TODO: test if using force_use actually helps
        #   this can be disabled by setting force_action to False
        force_use = self.item_helper.force_use_idx[member_pos]
        if self.force_action and force_use is not None:
          trans_actions[member_pos][nmmo.action.Use] = {
            nmmo.action.InventoryItem: force_use }

      # 'sell' is overrided by item_helper.force_sell_idx, if force_action = True
      if self._config.EXCHANGE_SYSTEM_ENABLED and 'sell' in actions:
        # TODO: test if using force_sell actually helps
        inv_idx = self.item_helper.force_sell_idx[member_pos] if self.force_action \
                    else actions['sell'][member_pos]
        if self.item_helper.in_inventory(member_pos, inv_idx):
          trans_actions[member_pos][nmmo.action.Sell] = {
            nmmo.action.InventoryItem: inv_idx,
            nmmo.action.Price: self.item_helper.get_price(member_pos, inv_idx) }

      # 'buy' is entirely from the item_helper (TODO: let the policy decide)
      if self._config.EXCHANGE_SYSTEM_ENABLED and 'buy' in actions:
        buy_idx = self.item_helper.force_buy_idx[member_pos]
        if buy_idx is not None:
          trans_actions[member_pos][nmmo.action.Buy] = {
            nmmo.action.MarketItem: buy_idx }

    # TODO: give, give-gold

    # TODO: prioritize when there are multiple actions on the same item
    #   currently, the actions will be executed in the order of
    #   Use -> Give -> Destroy -> Sell
    return trans_actions
