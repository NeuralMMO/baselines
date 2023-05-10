
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

    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._policy = self._policy.to(self._device)

    self._next_lstm_state = None
    self.reset()

  def reset(self, num_batch=1):
    self._next_lstm_state = (
        torch.zeros(self._policy.lstm.num_layers, num_batch,
                    self._policy.lstm.hidden_size).to(self._device),
        torch.zeros(self._policy.lstm.num_layers, num_batch,
                    self._policy.lstm.hidden_size).to(self._device))
    return self

  def act(self, observation, done=None):
    assert self._next_lstm_state is not None, "Must call reset() before act()"
    return {}

    # observation dim: (num_batch, num_features), done dim: (num_batch)
    t_obs = torch.Tensor(observation).to(self._device)
    if done is not None:
      t_done = torch.Tensor(done).to(self._device)

    # NOTE: pufferlib/frameworks/cleanrl.py: get_action_and_value takes in done
    #   but not using it for now. Marked as TODO, so revisit later.
    with torch.no_grad():
      action, _, _, _, self._next_lstm_state = \
        self._policy.get_action_and_value(t_obs, self._next_lstm_state)
    return action[0].cpu().numpy()

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
