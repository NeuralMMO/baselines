

import pufferlib
import pufferlib.frameworks.cleanrl
import pufferlib.models
import pufferlib.registry.nmmo
import pufferlib.vectorization.multiprocessing
import pufferlib.vectorization.serial
import torch
import torch.nn.functional as F
from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]


class RandomPolicy(pufferlib.models.Policy):
  def __init__(self, binding, input_size=128, hidden_size=256):
      '''Simple custom PyTorch policy subclassing the pufferlib BasePolicy

      This requires only that you structure your network as an observation encoder,
      an action decoder, and a critic function. If you use our LSTM support, it will
      be added between the encoder and the decoder.
      '''
      super().__init__(binding)
      self.raw_single_observation_space = binding.raw_single_observation_space

      self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden_size, n)
              for n in binding.single_action_space.nvec])
      self.value_head = torch.nn.Linear(hidden_size, 1)


  '''
  sample_in = torch.zeros(1, in_ch, *in_size)
  with torch.no_grad():
    self.h_size = len(self.tile_net(sample_in).flatten())

  def forward(self, x):
      bs, na = x['tile'].shape[:2]
      x_tile = x['tile'].contiguous().view(-1, *x['tile'].shape[2:])
      h_tile = self.tile_net(x_tile)
      h_tile = h_tile.view(bs, na, -1)  # flatten
      return h_tile
  '''


  def critic(self, hidden):
      return self.value_head(hidden)

  def encode_observations(self, env_outputs):
      # TODO: Change 0 for teams when teams are added
      env_outputs = self.binding.unpack_batched_obs(env_outputs)[0]

      tile = env_outputs['Tile']
      agents, tiles, features = tile.shape
      tile = tile.transpose(1, 2).view(agents, features, 15, 15)

      tile = self.tile_conv_1(tile)
      tile = F.relu(tile)
      tile = self.tile_conv_2(tile)
      tile = F.relu(tile)
      tile = tile.contiguous().view(agents, -1)
      tile = self.tile_fc(tile)
      tile = F.relu(tile)

      # Pull out rows corresponding to the agent
      agentEmb = env_outputs["Entity"]
      my_id = env_outputs["AgentId"][:,0]
      entity_ids = agentEmb[:,:,EntityId]
      mask = (entity_ids == my_id.unsqueeze(1)) & (entity_ids != 0)
      mask = mask.int()
      row_indices = torch.where(mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1)))
      entity = agentEmb[torch.arange(agentEmb.shape[0]), row_indices]

      #entity = env_outputs['Entity'][:, 0, :]
      entity = self.entity_fc(entity)
      entity = F.relu(entity)

      obs = torch.cat([tile, entity], dim=-1)
      return self.proj_fc(obs), None

  def decode_actions(self, hidden, lookup, concat=True):
      actions = [dec(hidden) for dec in self.decoders]
      if concat:
          return torch.cat(actions, dim=-1)
      return actions


  @staticmethod
  def create_policy(num_lstm_layers=1):
    return pufferlib.frameworks.cleanrl.make_policy(
      BasicPolicy,
      recurrent_args=[128, 256],
      recurrent_kwargs={'num_layers': num_lstm_layers})
