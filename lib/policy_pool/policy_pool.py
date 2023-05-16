
import random
from typing import Dict

from lib.policy_pool.policy_pool_record import PolicyPoolRecord

# Maintains a pool of policies, and allows random sampling from the pool
# based on the mean reward of each policy.

class PolicyPool():
  def __init__(self):
    self._policies = {}

  def add_policy(self, model_path, reward=0):
    pr = PolicyPoolRecord(model_path)
    pr.record_reward(reward)
    self._policies[model_path] = pr

  def select_policies(self, num_to_select) -> PolicyPoolRecord:
    if len(self._policies) == 0:
      return None

    return random.choices(
      list(self._policies.keys()),
      weights=[0.00001 + policy.mean_reward() for policy in self._policies.values()],
      k=num_to_select
    )

  def model_path(self, policy_id: str) -> str:
    return self._policies[policy_id]._model_weights_path

  def update_rewards(self, rewards: Dict[str, float]):
    for id, reward in rewards.items():
      self._policies[id].record_reward(reward)

