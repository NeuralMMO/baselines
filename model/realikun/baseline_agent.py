
import torch
from lib import cleanrl_ppo_lstm
from lib.agent.agent import Agent
from lib.agent.policy_pool import PolicyPool
from model.realikun.policy import BaselinePolicy

class BaselineAgent(Agent):
  def __init__(self, weights_path, binding):
    super().__init__()
    self._weights_path = weights_path
    self._policy = BaselinePolicy.create_policy()(binding)
    with open(weights_path, 'rb') as f:
      cleanrl_ppo_lstm.load_matching_state_dict(
        self._policy,
        torch.load(f, map_location=torch.device("cpu"))["agent_state_dict"]
      )

  def act(self, observation):
    return {}

