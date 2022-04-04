from pdb import set_trace as T
import numpy as np

import torch
from torch import nn
from torch.nn.utils import rnn

from neural import io, subnets

class Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden      = config.HIDDEN

        self.input = io.Input(config,
                embeddings=io.MixedEmbedding,
                attributes=subnets.SelfAttention)

        if config.EMULATE_FLAT_ATN:
            self.output = nn.Linear(hidden, 304)
        else:
            self.output = io.Output(config)

        self.model  = Simple(config)
        self.valueF = nn.Linear(hidden, 1)

    def _encode(self, obs):
        config = self.config

        if config.EMULATE_FLAT_OBS:
            import nmmo
            obs = nmmo.emulation.unpack_obs(config, obs)

        entityLookup  = self.input(obs)
        hidden, state = self.model(entityLookup, None, None)

        return entityLookup, hidden

    def compute_value(self, obs):
        _, hidden = self._encode(obs)
        return self.valueF(hidden).squeeze(1)

    def forward(self, obs):
        config = self.config

        entityLookup, hidden = self._encode(obs)
        
        if self.config.EMULATE_FLAT_ATN:
            return self.output(hidden), hidden

        logits = []
        output = self.output(hidden, entityLookup)
        return output
        for atnKey, atn in sorted(output.items()):
            for argKey, arg in sorted(atn.items()):
                logits.append(arg)

        return torch.cat(logits, dim=1)

class Simple(nn.Module):
   def __init__(self, config):
      '''Simple baseline model with flat subnetworks'''
      super().__init__()
      self.config = config
      h = config.HIDDEN

      self.ent    = nn.Linear(2*h, h)
      self.conv   = nn.Conv2d(h, h, 3)
      self.pool   = nn.MaxPool2d(2)
      self.fc     = nn.Linear(h*6*6, h)

      self.proj   = nn.Linear(2*h, h)
      self.attend = subnets.SelfAttention(config.EMBED, h)

   def forward(self, obs, state=None, lens=None):
      #Attentional agent embedding
      agentEmb  = obs['Entity']
      selfEmb   = agentEmb[:, 0:1].expand_as(agentEmb)
      agents    = torch.cat((selfEmb, agentEmb), dim=-1)
      agents    = self.ent(agents)
      agents, _ = self.attend(agents)
      #agents = self.ent(selfEmb)

      #Convolutional tile embedding
      tiles     = obs['Tile']
      self.attn = torch.norm(tiles, p=2, dim=-1)

      w      = self.config.WINDOW
      batch  = tiles.size(0)
      hidden = tiles.size(2)
      #Dims correct?
      tiles  = tiles.reshape(batch, w, w, hidden).permute(0, 3, 1, 2)
      tiles  = self.conv(tiles)
      tiles  = self.pool(tiles)
      tiles  = tiles.reshape(batch, -1)
      tiles  = self.fc(tiles)

      hidden = torch.cat((agents, tiles), dim=-1)
      hidden = self.proj(hidden)
      return hidden, state

class Recurrent(Simple):
   def __init__(self, config):
      '''Recurrent baseline model'''
      super().__init__(config)
      self.lstm = subnets.BatchFirstLSTM(
            input_size=config.HIDDEN,
            hidden_size=config.HIDDEN)

   #Note: seemingly redundant transposes are required to convert between 
   #Pytorch (seq_len, batch, hidden) <-> RLlib (batch, seq_len, hidden)
   def forward(self, obs, state, lens):
      #Attentional input preprocessor and batching
      lens = lens.cpu() if type(lens) == torch.Tensor else lens
      hidden, _ = super().forward(obs)
      config    = self.config
      h, c      = state

      TB  = hidden.size(0) #Padded batch of size (seq x batch)
      B   = len(lens)      #Sequence fragment time length
      TT  = TB // B        #Trajectory batch size
      H   = config.HIDDEN  #Hidden state size

      #Pack (batch x seq, hidden) -> (batch, seq, hidden)
      hidden        = rnn.pack_padded_sequence(
                         input=hidden.view(B, TT, H),
                         lengths=lens,
                         enforce_sorted=False,
                         batch_first=True)

      #Main recurrent network
      hidden, state = self.lstm(hidden, state)

      #Unpack (batch, seq, hidden) -> (batch x seq, hidden)
      hidden, _     = rnn.pad_packed_sequence(
                         sequence=hidden,
                         batch_first=True,
                         total_length=TT)

      return hidden.reshape(TB, H), state

class Attentional(nn.Module):
   def __init__(self, config):
      '''Transformer-based baseline model'''
      super().__init__(config)
      self.agents = nn.TransformerEncoderLayer(d_model=config.HIDDEN, nhead=4)
      self.tiles  = nn.TransformerEncoderLayer(d_model=config.HIDDEN, nhead=4)
      self.proj   = nn.Linear(2*config.HIDDEN, config.HIDDEN)

   def forward(self, obs, state=None, lens=None):
      #Attentional agent embedding
      agents    = self.agents(obs[Stimulus.Entity])
      agents, _ = torch.max(agents, dim=-2)

      #Attentional tile embedding
      tiles     = self.tiles(obs[Stimulus.Tile])
      self.attn = torch.norm(tiles, p=2, dim=-1)
      tiles, _  = torch.max(tiles, dim=-2)

      
      hidden = torch.cat((tiles, agents), dim=-1)
      hidden = self.proj(hidden)
      return hidden, state
