
from abc import abstractmethod
import glob
import os
import random

from pettingzoo.utils.env import AgentID

from lib.agent.agent import Agent

class AgentPool():
  def __init__(self):
    self._agents = {}

  def add(self, id, policy):
    self._agents[id] = policy

  def agent(self):
    if len(self._agents) == 0:
      return None

    return random.choice(list(self._agents.values()))

class DirAgentPool(AgentPool):
  def __init__(self, model_weights_dir: str):
    self._model_weights_dir = model_weights_dir
    self._agents = {}

  def poll_dir(self, interval_s: int = 60):
    # list all .pt files in the directory
    agent_files = glob.glob(os.path.join(self._model_weights_dir, '*.pt'))

    for agent_path in agent_files:
      agent_id = os.path.splitext(agent_path)[-3:-1]

      # add the agent if it's not already present
      if agent_id not in self._agents:
        print(f"Adding agent {agent_id} to agent pool")
        self.add(agent_id, self.make_agent(agent_path))

  @abstractmethod
  def make_agent(self, agent_id: AgentID) -> Agent:
    pass
