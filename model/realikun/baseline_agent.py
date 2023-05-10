
import os

import torch
from lib.agent.agent import Agent

from model.realikun.policy import BaselinePolicy


class BaselineAgent(Agent):
  def __init__(self, weights_path, binding):
    super().__init__()
    self._weights_path = weights_path
    self._policy = BaselinePolicy.create_policy()(binding)
    self._load()

  def act(self, observation):
    return {}

  def _load(self):
    if not os.path.exists(self._weights_path):
      return

    with open(self._weights_path, 'rb') as f:
      self._load_matching_state_dict(
        torch.load(f, map_location=torch.device("cpu"))["agent_state_dict"]
      )

  def _load_matching_state_dict(self, state_dict):
    upgrade_required = False
    model_state_dict = self._policy.state_dict()
    for name, param in state_dict.items():
      if name in model_state_dict:
        if model_state_dict[name].shape == param.shape:
          model_state_dict[name].copy_(param)
        else:
          upgrade_required = True
          print(f"Skipping {name} due to shape mismatch. " \
                f"Model shape: {model_state_dict[name].shape}, checkpoint shape: {param.shape}")
      else:
        upgrade_required = True
        print(f"Skipping {name} as it is not found in the model's state_dict")
    self._policy.load_state_dict(model_state_dict, strict=False)
    return upgrade_required
