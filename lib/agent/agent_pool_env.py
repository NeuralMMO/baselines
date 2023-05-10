from collections import defaultdict
from traitlets import default
from lib.agent.agent_env import AgentEnv
from pettingzoo.utils.env import AgentID, ParallelEnv
from typing import Any, Dict, List

from lib.agent.policy_pool import PolicyPool

# Creates an AgentEnv that uses a PolicyPool to select an agent for each
# episode. This allows for training against a pool of opponents.
class OpponentPoolEnv(AgentEnv):
  def __init__(
      self,
      env: ParallelEnv,
      opponent_ids: List[AgentID],
      policy_pool: PolicyPool,
      make_agent_fn):

    self._policy_pool = policy_pool
    self._policy_ids = {}
    self._make_agent_fn = make_agent_fn
    self._opponent_ids = opponent_ids

    super().__init__(env, {id: None for id in opponent_ids})

  def reset(self, **kwargs) -> Dict[int, Any]:
    # Record the rewards from the previous episode
    policy_rewards = defaultdict(list)
    for id in self._policy_ids.keys():
      policy_rewards[self._policy_ids[id]].append(self._rewards.get(id, 0))
    self._policy_pool.update_rewards({
      policy_id: sum(rewards) / len(rewards) for policy_id, rewards in policy_rewards.items()
    })

    self._policy_ids = {}
    self._agents = {}
    for id in self._opponent_ids:
      policy_id = self._policy_pool.select_policy()
      self._policy_ids[id] = policy_id
      self._agents[id] = self._make_agent_fn(self._policy_pool.model_path(policy_id))
    return super().reset(**kwargs)

