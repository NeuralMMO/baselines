
from typing import Dict

import numpy as np
import pandas as pd
from pufferlib.rating import OpenSkillRating

from lib.agent.util import softmax
from lib.policy_pool.policy_pool_record import PolicyPoolRecord

# Maintains a pool of policies, and allows random sampling from the pool
# based on the mean reward of each policy.

class PolicyPool():
  def __init__(self):
    self._policies = {}
    self._skill_rating = OpenSkillRating(1000, 1500, 100/3)

  def add_policy(self, model_path, reward=0):
    if model_path not in self._policies:
      pr = PolicyPoolRecord(model_path)
      self._policies[model_path] = pr
      pr.record_reward(reward)

    if len(self._policies) == 1:
      self._skill_rating.set_anchor(model_path)
    elif model_path not in self._skill_rating.stats.keys():
      self._skill_rating.add_policy(model_path)

  def select_policies(self, num_to_select, temperature=1.0) -> PolicyPoolRecord:
    scores = np.array(list(
      self._skill_rating.stats.get(model, 1) for model in self._policies.keys()
    ))
    max_score = max(scores)
    probs = softmax((max_score-scores) / max_score, temperature)

    return np.random.choice(
      list(self._policies.keys()),
      size=num_to_select,
      p=probs)


  def model_path(self, policy_id: str) -> str:
    return self._policies[policy_id]._model_weights_path

  def update_rewards(self, rewards: Dict[str, float]):
    for id, reward in rewards.items():
      self._policies[id].record_reward(reward)
    self._skill_rating.update(rewards.keys(), scores=rewards.values())

  def to_table(self):
    stats = self._skill_rating.stats
    table = pd.DataFrame(self._policies.keys(), columns=["Model"])
    table["Rank"] = [stats[model] for model in table["Model"]]
    table = table.sort_values(by='Rank', ascending=True)
    return table
