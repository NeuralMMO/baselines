from abc import abstractmethod
from pettingzoo.utils.env import AgentID, ParallelEnv

class Agent():
  @abstractmethod
  def act(self, observation):
    pass

class NeuralAgent(Agent):
  def __init__(self, agent_id: AgentID, policy_cls, model_path: str, binding):
    super().__init__(agent_id)
    self._model_path = model_path
    self._policy = policy_cls.create(model_path, binding)

  @abstractmethod
  def act(self, observation):
    pass
