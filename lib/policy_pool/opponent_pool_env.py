from collections import defaultdict
from lib.agent.agent import NoopAgent
from lib.agent.agent_env import AgentEnv
from pettingzoo.utils.env import AgentID, ParallelEnv
from typing import Any, Dict, List

from lib.policy_pool.policy_pool import PolicyPool

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

    # Select new policies for each opponent, use a NoopAgent if no policy is available
    new_policy_ids = self._policy_pool.select_policies(len(self._opponent_ids))
    if new_policy_ids is None:
       self._agents = {agent_id: NoopAgent() for agent_id in self._opponent_ids}
    else:
      for agent_id, policy_id in zip(self._opponent_ids, new_policy_ids):
        self._policy_ids[agent_id] = policy_id
        self._agents[agent_id] = self._make_agent_fn(self._policy_pool.model_path(policy_id))

    print("OpponentPoolEnv: Selected policies", self._policy_ids)

    return super().reset(**kwargs)

