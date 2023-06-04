

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
    self.tile_conv_1 = torch.nn.Conv2d(3, 32, 3)

  def encode_observations(self, env_outputs):
    return torch.randn((env_outputs.shape[0], 1)), None

  def decode_actions(self, hidden, lookup, concat=True):
    actions = [
      torch.randint(low=0, high=n,
      size=(hidden.shape[0],)) for n in self.binding.single_action_space.nvec]
    if concat:
        return torch.cat(actions, dim=-1)
    return actions

  def critic(self, hidden):
    return torch.zeros((hidden.shape[0], 1))

  @staticmethod
  def create_policy():
      return pufferlib.frameworks.cleanrl.make_policy(
          RandomPolicy,
          recurrent_args=[1, 1],
          recurrent_kwargs={'num_layers': 0})
