from lib.agent.agent_env import AgentEnv
from pettingzoo.utils.env import AgentID, ParallelEnv
from typing import Any, Dict, List

from lib.agent.agent_pool import AgentPool

class AgentPoolEnv(AgentEnv):
  def __init__(self, env: ParallelEnv, agent_ids: List[AgentID], agent_pool: AgentPool):
    self._agent_pool = agent_pool
    super().__init__(env, {agent_id: None for agent_id in agent_ids})

  def reset(self, **kwargs) -> Dict[int, Any]:
    self._agents = {
      id: self._agent_pool.agent() for id in self._agent_keys
    }
    return super().reset(**kwargs)
