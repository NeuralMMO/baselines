
import os
import gym

import numpy as np
import torch

from lib.agent.agent import Agent
from lib.agent.util import load_matching_state_dict
from model.realikun.policy import BaselinePolicy


class BaselineAgent(Agent):
  def __init__(self, weights_path, binding):
    super().__init__()
    self._weights_path = weights_path
    self._policy = BaselinePolicy.create_policy()(binding)
    self._binding = binding
    self._load()

    self._device = torch.device("cpu")
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

    # observation dim: (num_batch, num_features), done dim: (num_batch)
    t_obs = torch.Tensor(self._pack_unbatched_obs(observation)).to(self._device)

    # NOTE: pufferlib/frameworks/cleanrl.py: get_action_and_value takes in done
    #   but not using it for now. Marked as TODO, so revisit later.
    with torch.no_grad():
      action, _, _, _, self._next_lstm_state = \
        self._policy.get_action_and_value(t_obs, self._next_lstm_state)
    unpacked =  self._unpack_actions(action[0].cpu().numpy())
    return unpacked

  def _load(self):
    if not os.path.exists(self._weights_path):
      return

    with open(self._weights_path, 'rb') as f:
      load_matching_state_dict(
        self._policy,
        torch.load(f, map_location=torch.device("cpu"))["agent_state_dict"]
      )

  def _unpack_actions(self, packed_actions):
    flat_space = self._binding.raw_single_action_space

    actions = {}
    for action_name, space in flat_space.items():
      size = np.prod(space.shape)
      actions[action_name] = packed_actions[:size]
      packed_actions = packed_actions[size:]

    return actions

  def _pack_unbatched_obs(self, batched_obs):
      flat_space = self._binding._featurized_single_observation_space

      if not isinstance(flat_space, dict):
          return batched_obs.reshape(batched_obs.shape[0], -1)

      if () in flat_space:
          return batched_obs.reshape(batched_obs.shape[0], -1)

      packed_obs = []

      def pack_recursive(key_list, obs):
          nonlocal packed_obs
          if isinstance(obs, dict):
              for key, val in obs.items():
                  pack_recursive(key_list + [key], val)
          else:
              packed_obs.append(obs.flatten())

      pack_recursive([], batched_obs)

      return np.concatenate(packed_obs)
