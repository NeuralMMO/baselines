from abc import abstractmethod
from pettingzoo.utils.env import AgentID

class Agent():
  @abstractmethod
  def act(self, observation):
    pass

class NoopAgent(Agent):
  def act(self, observation):
    return {}
