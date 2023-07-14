from abc import abstractmethod


class Agent:
  @abstractmethod
  def act(self, observation):
    pass


class NoopAgent(Agent):
  def act(self, observation):
    return {}
