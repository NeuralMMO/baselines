
import random
from agent.agent import Agent
from typing import Any, Dict, List
from pettingzoo.utils.env import AgentID, ParallelEnv

from agent.baseline_agent import BaselineAgent


class PolicyPool():
  def __init__(self, model_paths: List[str]):
    self._model_paths = model_paths

  def add(self, path: str):
    self._model_paths.append(path)

  def agent(self):
    return BaselineAgent(random.choice(self._model_paths))
