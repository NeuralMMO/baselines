import numpy as np
import torch

from lib.agent.agent import Agent
from lib.agent.util import load_matching_state_dict
from model.basic.policy import BasicPolicy
from model.basic_teams.policy import BasicTeamsPolicy
from model.improved.policy import ImprovedPolicy
from model.random.policy import RandomPolicy
from model.realikun.policy import RealikunPolicy
from model.decode.policy import Policy as DecodePolicy
from model.realikun_simple.policy import RealikunSimplifiedPolicy


class BaselineAgent(Agent):
  def __init__(self, binding, weights_path=None, policy_cls=None):
    super().__init__()
    self._weights_path = weights_path
    self._binding = binding
    self._device = torch.device("cpu")

    assert (
        policy_cls is not None or weights_path is not None
    ), "Must provide either policy_cls or weights_path"
    if weights_path:
      self._load()
    else:
      self._policy = policy_cls(binding)
    self._policy = self._policy.to(self._device)

    self._next_lstm_state = None
    self.reset()

  def reset(self, num_batch=1):
    self._next_lstm_state = (
        torch.zeros(
            self._policy.lstm.num_layers,
            num_batch,
            self._policy.lstm.hidden_size).to(
            self._device),
        torch.zeros(
            self._policy.lstm.num_layers,
            num_batch,
            self._policy.lstm.hidden_size).to(
            self._device),
    )
    return self

  def act(self, observation, done=None):
    assert self._next_lstm_state is not None, "Must call reset() before act()"

    if observation is None:
      return {}

    # observation dim: (num_batch, num_features), done dim: (num_batch)
    t_obs = torch.Tensor(
        self._pack_unbatched_obs(observation)).to(
        self._device)

    # NOTE: pufferlib/frameworks/cleanrl.py: get_action_and_value takes in done
    #   but not using it for now. Marked as TODO, so revisit later.
    with torch.no_grad():
      action, _, _, _, self._next_lstm_state = self._policy.get_action_and_value(
          t_obs, self._next_lstm_state)
    unpacked = self._unpack_actions(action[0].cpu().numpy())
    return unpacked

  def _load(self):
    with open(self._weights_path, "rb") as f:
      model = torch.load(f, map_location=torch.device("cpu"))
      self._policy = self.policy_class(model.get("model_type", "realikun"))(
          self._binding
      )
      load_matching_state_dict(self._policy, model["agent_state_dict"])

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

  @staticmethod
  def policy_class(model_type: str):
    if model_type == "realikun":
      return RealikunPolicy.create_policy()
    if model_type == "realikun-simplified":
      return RealikunSimplifiedPolicy.create_policy()
    elif model_type == "random":
      return RandomPolicy.create_policy()
    elif model_type == "basic":
      return BasicPolicy.create_policy(num_lstm_layers=0)
    elif model_type == "basic-lstm":
      return BasicPolicy.create_policy(num_lstm_layers=1)
    elif model_type == "improved":
      return ImprovedPolicy.create_policy(num_lstm_layers=0)
    elif model_type == "improved-lstm":
      return ImprovedPolicy.create_policy(num_lstm_layers=1)
    elif model_type == "basic-teams":
      return BasicTeamsPolicy.create_policy(num_lstm_layers=0)
    elif model_type == "basic-teams-lstm":
      return BasicTeamsPolicy.create_policy(num_lstm_layers=0)
    elif model_type == "decode":
      return DecodePolicy.create_policy()
    else:
      raise ValueError(f"Unsupported model type: {model_type}")
