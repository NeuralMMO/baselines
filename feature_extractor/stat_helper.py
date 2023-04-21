from typing import Dict, Any

import nmmo

from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState

from feature_extractor.entity_helper import EntityHelper

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col


class StatHelper:
  def __init__(self, config: nmmo.config.Config,
               entity_helper: EntityHelper) -> None:
    self._config = config
    self._entity_helper = entity_helper
    self._team_size = self._entity_helper.team_size

    self.step_onto_herb_cnt = None  # statistics for reward only
    self.player_kill_num = None  # for comparing with playerDefeat stat only

    # CHECK ME: original implementation kept separate record for different
    #   npc types, but here sums them all. DO WE need to separate those?
    self.npc_kill_num = None  # statistics for reward only

  def reset(self):
    self.player_kill_num = [0] * self._team_size
    self.npc_kill_num = [0] * self._team_size
    self.step_onto_herb_cnt = [0] * self._team_size

  def update(self, obs: Dict[int, Any]):
    for agent_id, agent_obs in obs.items():
      member_pos = self._entity_helper.agent_id_to_pos(agent_id)

      # NOTE: entity_helper.attack_target records the target agent id
      entity_in_sight = agent_obs['Entity'][:,EntityAttr["id"]]
      target_agent = self._entity_helper.attack_target[member_pos]
      if target_agent is not None and target_agent not in entity_in_sight:
        # NOTE: the target may have been killed by the other agent
        #   IF multiple agents target one, these kill_nums can be double-counted
        if target_agent > 0:
          self.player_kill_num[member_pos] += 1
        else:
          self.npc_kill_num[member_pos] += 1

    # CHECK ME: We can get precise step_onto_herb_cnt and other precise metrics
    #   from the event log. Do we need this?
