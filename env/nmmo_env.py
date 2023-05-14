
from typing import Dict, List

import nmmo
from pettingzoo.utils.env import AgentID
import numpy as np

class NMMOEnv(nmmo.Env):
  def __init__(self, config, symlog_rewards=False):
    super().__init__(config)
    self._symlog_rewards = symlog_rewards

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

      if self._symlog_rewards:
        rewards[agent_id] = _symlog(rewards[agent_id])

      if dones.get(agent_id, False):
        infos[agent_id]["cod/starved"] = agent.food.val == 0
        infos[agent_id]["cod/dehydrated"] = agent.water.val == 0

    return rewards, infos

def _symlog(value):
    """Applies symmetrical logarithmic transformation to a float value."""
    sign = np.sign(value)
    abs_value = np.abs(value)

    if abs_value >= 1:
        log_value = np.log10(abs_value)
    else:
        log_value = abs_value

    symlog_value = sign * log_value
    return symlog_value
