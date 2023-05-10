
from contextlib import contextmanager
import json
import os
from typing import Dict
from filelock import FileLock

from lib.policy_pool.policy_pool import PolicyPool
from lib.policy_pool.policy_pool_record import PolicyPoolRecord

# A PolicyPool that persists to a JSON file
class JsonPolicyPool(PolicyPool):
  def __init__(self, file_path: str):
    super().__init__()
    self._file_path = file_path

  def add_policy(self, model_path, reward=0):
    with self._persist():
      super().add_policy(model_path, reward)

  def update_rewards(self, rewards: Dict[str, float]):
    with self._persist():
      super().update_rewards(rewards)

  def _save(self):
    with open(self._file_path, 'w') as f:
      json.dump({
        id: policy.to_dict() for id, policy in self._policies.items()
      }, f)

  def _load(self):
    if not os.path.exists(self._file_path):
      print(f"No policy pool file found {self._file_path}, skipping load")
      return

    with open(self._file_path, 'r') as f:
      self._policies = {
        id: PolicyPoolRecord.from_dict(policy) for id, policy
        in json.load(f).items()
      }

  @contextmanager
  def _persist(self):
    lock = FileLock(self._file_path + ".lock")
    with lock:
      self._load()
      yield
      self._save()
