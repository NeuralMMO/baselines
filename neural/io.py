from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn

import nmmo

from neural import subnets

class MixedEmbedding(nn.Module):
   def __init__(self, continuous, discrete, config):
      super().__init__()

      self.continuous = torch.nn.ModuleList([
            torch.nn.Linear(1, config.EMBED) for _ in range(continuous)])
      self.discrete   = torch.nn.Embedding(discrete, config.EMBED)

   def forward(self, x):
      continuous = x['Continuous'].split(1, dim=-1)
      continuous = [net(e) for net, e in zip(self.continuous, continuous)]
      continuous = torch.stack(continuous, dim=-2)
      discrete   = self.discrete(x['Discrete'].long())

      return torch.cat((continuous, discrete), dim=-2)

class Input(nn.Module):
   def __init__(self, config, embeddings, attributes):
      '''Network responsible for processing observations

      Args:
         config     : A configuration object
         embeddings : An attribute embedding module
         attributes : An attribute attention module
      '''
      super().__init__()

      self.embeddings = nn.ModuleDict()
      self.attributes = nn.ModuleDict()

      for _, entity in nmmo.Serialized:
         continuous = len([e for e in entity if e[1].CONTINUOUS])
         discrete   = len([e for e in entity if e[1].DISCRETE])
         self.attributes[entity.__name__] = nn.Linear(
               (continuous+discrete)*config.HIDDEN, config.HIDDEN)
         self.embeddings[entity.__name__] = embeddings(
               continuous=continuous, discrete=4096, config=config)

      
      #TODO: implement obs scaling in a less hackey place
      self.tileWeight = torch.Tensor([1.0, 0.0, 0.02, 0.02])
      self.entWeight  = torch.Tensor([1.0, 0.0, 0.0, 0.05, 0.00, 0.02, 0.02, 0.1, 0.01, 0.1, 0.1, 0.1, 0.3])

      if torch.cuda.is_available():
          self.tileWeight = self.tileWeight.cuda()
          self.entWeight  = self.entWeight.cuda()

   def forward(self, inp):
      '''Produces tensor representations from an IO object

      Args:                                                                   
         inp: An IO object specifying observations                      
         

      Returns:
         entityLookup      : A fixed size representation of each entity
      ''' 
      #Pack entities of each attribute set
      entityLookup = {}

      device                       = inp['Tile']['Continuous'].device
      inp['Tile']['Continuous']   *= self.tileWeight
      inp['Entity']['Continuous'] *= self.entWeight
 
      entityLookup['N'] = inp['Entity'].pop('N')
      for name, entities in inp.items():
         #Construct: Batch, ents, nattrs, hidden
         embeddings = self.embeddings[name](entities)
         B, N, _, _ = embeddings.shape
         embeddings = embeddings.view(B, N, -1)

         #Construct: Batch, ents, hidden
         entityLookup[name] = self.attributes[name](embeddings)

      return entityLookup

class Output(nn.Module):
   def __init__(self, config):
      '''Network responsible for selecting actions

      Args:
         config: A Config object
      '''
      super().__init__()
      self.config = config
      self.h = config.HIDDEN

      self.net = DiscreteAction(self.config, self.h, self.h)
      self.arg = nn.Embedding(nmmo.Action.n, self.h)

   def names(self, nameMap, args):
      '''Lookup argument indices from name mapping'''
      return np.array([nameMap.get(e) for e in args])

   def forward(self, obs, lookup):
      '''Populates an IO object with actions in-place                         
                                                                              
      Args:                                                                   
         obs    : An IO object specifying observations
         lookup : A fixed size representation of each entity
      ''' 
      rets = defaultdict(dict)
      for atn in nmmo.Action.edges:
         for arg in atn.edges:
            lens  = None
            if arg.argType == nmmo.action.Fixed:
               batch = obs.shape[0]
               idxs  = [e.idx for e in arg.edges]
               cands = self.arg.weight[idxs]
               cands = cands.repeat(batch, 1, 1)
            else:
               cands = lookup['Entity']
               lens  = lookup['N']

            logits         = self.net(obs, cands, lens)
            rets[atn][arg] = logits

      return rets
      
#Root action class
class Action(nn.Module):
   pass

class DiscreteAction(Action):
   '''Head for making a discrete selection from
   a variable number of candidate actions'''
   def __init__(self, config, xdim, h):
      super().__init__()
      self.net = subnets.DotReluBlock(h)

   def forward(self, stim, args, lens):
      x = self.net(stim, args)

      if lens is not None:
         mask = torch.arange(x.shape[-1]).to(x.device).expand_as(x)
         x[mask >= lens] = 0

      return x
