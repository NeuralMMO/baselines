
from typing import Dict, List

import nmmo
from pettingzoo.utils.env import AgentID


class NMMOEnv(nmmo.Env):
  def _compute_rewards(self, agents: List[AgentID], dones: Dict[AgentID, bool]):
    infos = {}
    rewards = { eid: -1 for eid in dones }

    for agent_id in agents:
      infos[agent_id] = {}
      agent = self.realm.players.get(agent_id)
      assert agent is not None, f'Agent {agent_id} not found'

      rewards[agent_id] = 0.1
      if agent.food.val / self.config.RESOURCE_BASE > 0.4:
        rewards[agent_id] += 0.1
      if agent.food.val / self.config.RESOURCE_BASE > 0.6:
        rewards[agent_id] += 0.5

      if agent.water.val / self.config.RESOURCE_BASE > 0.4:
        rewards[agent_id] += 0.1
      if agent.water.val / self.config.RESOURCE_BASE > 0.6:
        rewards[agent_id] += 0.5

      if agent.health.val / self.config.PLAYER_BASE_HEALTH > 0.4:
        rewards[agent_id] += 0.1
      if agent.health.val / self.config.PLAYER_BASE_HEALTH > 0.6:
        rewards[agent_id] += 0.5

    return rewards, infos
