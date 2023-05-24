
from typing import Dict

import numpy as np
import pandas as pd
from pufferlib.rating import OpenSkillRating

from lib.policy_pool.policy_pool_record import PolicyPoolRecord

# Maintains a pool of policies, and allows random sampling from the pool
# based on the mean reward of each policy.

class PolicyPool():
  def __init__(self):
    self._policies = {}
    self._skill_rating = OpenSkillRating(1000, 1500, 100/3)

  def add_policy(self, model_path):
    if model_path not in self._policies:
      pr = PolicyPoolRecord(model_path)
      self._policies[model_path] = pr
      pr.record_sample()

    if len(self._policies) == 1:
      self._skill_rating.set_anchor(model_path)
    elif model_path not in self._skill_rating.stats.keys():
      self._skill_rating.add_policy(model_path)

  def select_best_policies(self, num_to_select) -> PolicyPoolRecord:
    score = lambda x: self._skill_rating.stats.get(x, 1000)
    policies = sorted(self._policies.keys(), key=score, reverse=True)
    return policies[:num_to_select]

  def select_least_tested_policies(self, num_to_select) -> PolicyPoolRecord:
    score = lambda x: self._policies[x].num_samples()
    policies = sorted(self._policies.keys(), key=score)
    return policies[:num_to_select]

  def model_path(self, policy_id: str) -> str:
    return self._policies[policy_id]._model_weights_path

  def update_rewards(self, rewards: Dict[str, float]):
    for id in rewards.keys():
      self._policies[id].record_sample()
    self._skill_rating.update(rewards.keys(), scores=rewards.values())

  def to_table(self):
    stats = self._skill_rating.stats
    table = pd.DataFrame(self._policies.keys(), columns=["Model"])
    table["Rank"] = [stats[model] for model in table["Model"]]
    table["Num Samples"] = [self._policies[model].num_samples() for model in table["Model"]]
    table = table.sort_values(by='Rank')
    return table
