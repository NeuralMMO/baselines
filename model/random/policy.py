

import pufferlib
import pufferlib.frameworks.cleanrl
import pufferlib.models
import pufferlib.registry.nmmo
import pufferlib.vectorization.multiprocessing
import pufferlib.vectorization.serial
import torch

class RandomPolicy(pufferlib.models.Policy):
  def __init__(self, binding):
    super().__init__(binding)
    self.decoders = torch.nn.ModuleList([torch.nn.Linear(1, n)
            for n in binding.single_action_space.nvec])

  def encode_observations(self, env_outputs):
    return torch.randn((env_outputs.shape[0], 1)).to(env_outputs.device), None

  def decode_actions(self, hidden, lookup, concat=True):
    torch.nn.init.xavier_uniform_(hidden)
    actions = [dec(hidden) for dec in self.decoders]
    if concat:
      return torch.cat(actions, dim=-1)
    return actions

  def critic(self, hidden):
    return torch.zeros((hidden.shape[0], 1)).to(hidden.device)

  @staticmethod
  def create_policy():
      return pufferlib.frameworks.cleanrl.make_policy(
          RandomPolicy,
          recurrent_args=[1, 1],
          recurrent_kwargs={'num_layers': 0})
