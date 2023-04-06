
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState

from feature_extractor.target_tracker import TargetTracker

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col


class Stats:
  def __init__(self, config, team_size, target_tracker: TargetTracker):
    self.config = config
    self.player_kill_num = None  # for comparing with playerDefeat stat only
    self.npc_kill_num = None  # statistics for reward only
    self.step_onto_herb_cnt = None  # statistics for reward only
    self.team_size = team_size
    self.target_tracker = target_tracker

  def reset(self):
    self.player_kill_num = [0] * self.team_size
    self.npc_kill_num = {kind: [0] * self.team_size for kind in 'pnh'}
    self.step_onto_herb_cnt = [0] * self.team_size

  def update(self, obs):
    for player_id, player_obs in obs.items():
      if self.target_tracker.target_entity_id[player_id] is None:  # no target
          continue
      entity_obs = player_obs['Entity']
      entity_in_sight = entity_obs[:, EntityAttr["id"]]
      if self.target_tracker.target_entity_id[player_id] not in entity_in_sight:
        if self.target_tracker.target_entity_id[player_id] > 0:
          self.player_kill_num[player_id] += 1
        elif self.target_tracker.target_entity_id[player_id] < 0:
          if self.target_tracker.target_entity_pop[player_id] == -1:
            self.npc_kill_num['p'][player_id] += 1
          elif self.target_tracker.target_entity_pop[player_id] == -2:
            self.npc_kill_num['n'][player_id] += 1
          elif self.target_tracker.target_entity_pop[player_id] == -3:
            self.npc_kill_num['h'][player_id] += 1
          else:
            raise ValueError('Unknown npc pop:', self.target_tracker.target_entity_pop[player_id])



      # xcxc
      # my_prev_pos = self._entity_pos(self._my_entity(self.prev_obs, player_id))
      # # update herb gathering count
      # if self.tile_map[my_curr_pos[0], my_curr_pos[1]] == material.Herb.index and \
      #     (my_curr_pos[0] != my_prev_pos[0] or my_curr_pos[1] != my_prev_pos[1]):
      #   self.step_onto_herb_cnt[player_id] += 1
