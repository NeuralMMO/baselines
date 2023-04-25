from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch import nn

import nmmo
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState

from model.simple import subnets
from model.simple.model_architecture import EMBED_DIM, HIDDEN_DIM, INPUT_DISCRETE_EMBEDDING

class MixedEmbedding(nn.Module):
  def __init__(self, num_attributes):
    super().__init__()

    self.attr_embed_net = torch.nn.ModuleList([
      torch.nn.Linear(1, EMBED_DIM) for _ in range(num_attributes)])

  def forward(self, x):
    embeddings = x.split(1, dim=-1)
    embeddings = [net(e) for net, e in zip(self.attr_embed_net, embeddings)]
    embeddings = torch.stack(embeddings, dim=-2)

    return embeddings

class Input(nn.Module):
  def __init__(self, embeddings):
    '''Network responsible for processing observations
    Args:
        config     : A configuration object
        embeddings : An attribute embedding module
    '''
    super().__init__()

    self.embeddings = nn.ModuleDict()
    self.attributes = nn.ModuleDict()

    self.attributes["Tile"] = nn.Linear(TileState.State.num_attributes * EMBED_DIM, EMBED_DIM)
    self.embeddings["Tile"] = embeddings(TileState.State.num_attributes * EMBED_DIM)
    self.attributes["Entity"] = nn.Linear(EntityState.State.num_attributes * EMBED_DIM, EMBED_DIM)
    self.embeddings["Entity"] = embeddings(EntityState.State.num_attributes * EMBED_DIM)

    #TODO: implement obs scaling in a less hackey place
    # self.register_buffer('tileWeight', torch.Tensor([0.02, 0.02, 1]))
    # self.register_buffer('entWeight', torch.Tensor([1.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.02, 0.02, 0.1, 0.01, 0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    # self.register_buffer('itemWeight', torch.Tensor([0.0, 0.0, 0.1, 0.025, 0.025, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.02, 1.0]))
    # self.itemWeight = torch.Tensor([0.0, 0.0, 0.1, 0.025, 0.025, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.02, 1.0])

  def forward(self, inp):
    '''Produces tensor representations from an IO object
    Args:
        inp: An IO object specifying observations

    Returns:
        entityLookup      : A fixed size representation of each entity
    '''
    #Pack entities of each attribute set
    entityLookup = {}

    # inp['Tile']   *= self.tileWeight
    # inp['Entity'] *= self.entWeight

    for name in ('Tile', 'Entity'):
        #Construct: Batch, ents, nattrs, hidden
        embeddings = self.embeddings[name](inp[name])
        B, N, _, _ = embeddings.shape
        embeddings = embeddings.view(B, N, -1)

        #Construct: Batch, ents, hidden
        entityLookup[name] = self.attributes[name](embeddings)

    return entityLookup

class Output(nn.Module):
  def __init__(self):
    super().__init__()

    self.proj = None
    if HIDDEN_DIM != EMBED_DIM:
        self.proj = nn.Linear(HIDDEN_DIM, EMBED_DIM)
    self.net = DiscreteAction(EMBED_DIM)
    self.arg = nn.Embedding(nmmo.Action.n, EMBED_DIM)

  def names(self, nameMap, args):
    '''Lookup argument indices from name mapping'''
    return np.array([nameMap.get(e) for e in args])

  def forward(self, obs, lookup):
    '''Populates an IO object with actions in-place

    Args:
        obs    : An IO object specifying observations
        lookup : A fixed size representation of each entity
    '''
    if self.proj:
        obs = self.proj(obs)

    batch = obs.shape[0]

    rets = defaultdict(dict)
    for atn in [nmmo.action.Move, nmmo.action.Attack]:
      for arg in atn.edges:
        if arg.argType == nmmo.action.Fixed:
            batch = obs.shape[0]
            idxs  = [e.idx for e in arg.edges]
            cands = self.arg.weight[idxs]
            cands = cands.repeat(batch, 1, 1)
        elif arg == nmmo.action.Target:
            cands = lookup['Entity']
        mask = lookup["ActionTargets"][atn][arg]

        logits         = self.net(obs, cands, mask)
        rets[atn][arg] = logits

    return [
       rets[nmmo.action.Attack][nmmo.action.Style],
       rets[nmmo.action.Attack][nmmo.action.Target],
       rets[nmmo.action.Move][nmmo.action.Direction],
    ]

class Action(nn.Module):
  pass

class DiscreteAction(Action):
  '''Head for making a discrete selection from
  a variable number of candidate actions'''
  def __init__(self, h):
    super().__init__()
    self.net = subnets.DotReluBlock(h)

  def forward(self, stim, args, mask):
    logits  = self.net(stim, args)

    if mask is not None:
        mask_value = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype).to(logits.device)
        logits  = torch.where(mask.bool(), logits, mask_value)

    return logits
