
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

    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._policy = self._policy.to(self._device)

    self._next_lstm_state = None

  def reset(self, num_batch=1):
    self._next_lstm_state = (
        torch.zeros(self._policy.lstm.num_layers, num_batch,
                    self._policy.lstm.hidden_size).to(self._device),
        torch.zeros(self._policy.lstm.num_layers, num_batch,
                    self._policy.lstm.hidden_size).to(self._device))

  def act(self, observation, done=None):
    assert self._next_lstm_state is not None, "Must call reset() before act()"

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
