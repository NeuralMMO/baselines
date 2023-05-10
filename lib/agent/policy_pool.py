
from abc import abstractmethod
import glob
import json
import os
import random
from typing import Dict
from filelock import FileLock

from pettingzoo.utils.env import AgentID

from lib.agent.agent import Agent

class PolicyRecord():

  NUM_SAMPLES_TO_KEEP = 25

  def __init__(self, model_weights_path: str):
    self._model_weights_path = model_weights_path
    self._rewards = []

  def record_reward(self, reward: float):
    self._rewards.append(reward)
    if len(self._rewards) > self.NUM_SAMPLES_TO_KEEP:
      self._rewards.pop(0)

  def mean_reward(self) -> float:
    if len(self._rewards) == 0:
      return 0
    return sum(self._rewards) / len(self._rewards)

  def to_dict(self):
    return {
      'model_weights_path': self._model_weights_path,
      'rewards': self._rewards,
    }

  @classmethod
  def from_dict(cls, data):
    policy = cls(data['model_weights_path'])
    policy._rewards = data['rewards']
    return policy

class PolicyPool():
  def __init__(self, file_path: str):
    self._file_path = file_path
    self._policies = {}

  def select_policy(self) -> PolicyRecord:
    if len(self._policies) == 0:
      return None

    return random.choices(
      list(self._policies.keys()),
      weights=[0.00001 + policy.mean_reward() for policy in self._policies.values()],
      k=1
    )[0]

  def model_path(self, policy_id: str) -> str:
    return self._policies[policy_id]._model_weights_path

  def save(self):
    with open(self._file_path, 'w') as f:
      json.dump({
        id: policy.to_dict() for id, policy in self._policies.items()
      }, f)

  def load(self):
    if not os.path.exists(self._file_path):
      print(f"No policy pool file found {self._file_path}, skipping load")
      return

    with open(self._file_path, 'r') as f:
      self._policies = {
        id: PolicyRecord.from_dict(policy) for id, policy
        in json.load(f).items()
      }

  def update_rewards(self, rewards: Dict[str, float]):
    lock = FileLock(self._file_path + ".lock")
    with lock:
      self.load()
      for id, reward in rewards.items():
        self._policies[id].record_reward(reward)
      self.save()

