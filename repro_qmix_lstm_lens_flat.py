from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import nmmo

import torch
from torch import nn
from torch.nn.utils import rnn
from torch.nn import functional as F

import functools, gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple

import unittest

import ray
from ray.tune import register_env
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

def modelSize(net):
   '''Print model size'''
   params = 0
   for e in net.parameters():
      params += np.prod(e.size())
   params = int(params/1000)
   print("Network has ", params, "K params")

def ModuleList(module, *args, n=1):
   '''Repeat module n times'''
   return nn.ModuleList([module(*args) for i in range(n)])

def Conv2d(fIn, fOut, k, stride=1):
   '''torch Conv2d with same padding'''
   assert k % 2 == 0
   pad = int((k-1)/2)
   return torch.nn.Conv2d(fIn, fOut, 
      k, stride=stride, padding=pad)

def Pool(k, stride=1, pad=0):
   return torch.nn.MaxPool2d(
      k, stride=stride, padding=pad)

class ConvReluPool(nn.Module):
   def __init__(self, fIn, fOut, 
         k, stride=1, pool=2):
      super().__init__()
      self.conv = Conv2d(
         fIn, fOut, k, stride)
      self.pool = Pool(k)

   def forward(self, x):
      x = self.conv(x)
      x = F.relu(x)
      x = self.pool(x)
      return x

class ReluBlock(nn.Module):
   def __init__(self, h, layers=2, postRelu=True):
      super().__init__()
      self.postRelu = postRelu

      self.layers = ModuleList(
         nn.Linear, h, h, n=layers)

   def forward(self, x):
      for idx, fc in enumerate(self.layers):
         if idx != 0:
            x = torch.relu(x)
         x = fc(x)

      if self.postRelu:
         x = torch.relu(x)

      return x

class DotReluBlock(nn.Module):
   def __init__(self, h, layers=2):
      super().__init__()
      self.key = ReluBlock(
         h, layers, postRelu=False)

      self.val = ReluBlock(
         h, layers, postRelu=False)
   
   def forward(self, k, v):
      k = self.key(k).unsqueeze(-2)
      v = self.val(v)
      x = torch.sum(k * v, -1)
      return x

class ScaledDotProductAttention(nn.Module):
   def __init__(self, h):
      super().__init__()
      self.scale = np.sqrt(h)

   def forward(self, Q, K, V):
      Kt  = K.transpose(-2, -1)
      QK  = torch.matmul(Q, Kt)

      #Original is attending over hidden dims?
      #QK  = torch.softmax(QK / self.scale, dim=-2)
      QK    = torch.softmax(QK / self.scale, dim=-1)
      score = torch.sum(QK, -2)
      QKV = torch.matmul(QK, V)
      return QKV, score

class SelfAttention(nn.Module):
   def __init__(self, xDim, yDim, flat=True):
      super().__init__()

      self.Q = torch.nn.Linear(xDim, yDim)
      self.K = torch.nn.Linear(xDim, yDim)
      self.V = torch.nn.Linear(xDim, yDim)

      self.attention = ScaledDotProductAttention(yDim)
      self.flat = flat

   def forward(self, q):
      Q = self.Q(q)
      K = self.K(q)
      V = self.V(q)

      attn, scores = self.attention(Q, K, V)

      if self.flat:
         attn, _ = torch.max(attn, dim=-2)

      return attn, scores

class BatchFirstLSTM(nn.LSTM):
   def __init__(self, *args, **kwargs):
      super().__init__(*args, batch_first=True, **kwargs)

   def forward(self, input, hx):
      h, c       = hx
      h          = h.transpose(0, 1)
      c          = c.transpose(0, 1)
      hidden, hx = super().forward(input, [h, c])
      h, c       = hx
      h          = h.transpose(0, 1)
      c          = c.transpose(0, 1)
      return hidden, [h, c]

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
      self.net = DotReluBlock(h)

   def forward(self, stim, args, lens):
      x = self.net(stim, args)

      if lens is not None:
         mask = torch.arange(x.shape[-1]).to(x.device).expand_as(x)
         x[mask >= lens] = 0

      return x

class Base(nn.Module):
   def __init__(self, config):
      '''Base class for baseline policies

      Args:
         config: A Configuration object
      '''
      super().__init__()
      self.embed  = config.EMBED
      self.config = config

      self.input  = Input(config,
            embeddings=MixedEmbedding,
            attributes=SelfAttention)
      self.output = Output(config)

      self.valueF = nn.Linear(config.HIDDEN, 1)

   def hidden(self, obs, state=None, lens=None):
      '''Abstract method for hidden state processing, recurrent or otherwise,
      applied between the input and output modules

      Args:
         obs: An observation dictionary, provided by forward()
         state: The previous hidden state, only provided for recurrent nets
         lens: Trajectory segment lengths used to unflatten batched obs
      ''' 
      raise NotImplementedError('Implement this method in a subclass')

   def forward(self, obs, state=None, lens=None):
      '''Applies builtin IO and value function with user-defined hidden
      state subnetwork processing. Arguments are supplied by RLlib
      ''' 
      entityLookup  = self.input(obs)
      hidden, state = self.hidden(entityLookup, state, lens)
      self.value    = self.valueF(hidden).squeeze(1)
      actions       = self.output(hidden, entityLookup)
      return actions, state

class Simple(Base):
   def __init__(self, config):
      '''Simple baseline model with flat subnetworks'''
      super().__init__(config)
      h = config.HIDDEN

      self.ent    = nn.Linear(2*h, h)
      self.conv   = nn.Conv2d(h, h, 3)
      self.pool   = nn.MaxPool2d(2)
      self.fc     = nn.Linear(h*6*6, h)

      self.proj   = nn.Linear(2*h, h)
      self.attend = SelfAttention(self.embed, h)

   def hidden(self, obs, state=None, lens=None):
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
      self.lstm = BatchFirstLSTM(
            input_size=config.HIDDEN,
            hidden_size=config.HIDDEN)

   #Note: seemingly redundant transposes are required to convert between 
   #Pytorch (seq_len, batch, hidden) <-> RLlib (batch, seq_len, hidden)
   def hidden(self, obs, state, lens):
      #Attentional input preprocessor and batching
      lens = lens.cpu() if type(lens) == torch.Tensor else lens
      hidden, _ = super().hidden(obs)
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

class Attentional(Base):
   def __init__(self, config):
      '''Transformer-based baseline model'''
      super().__init__(config)
      self.agents = nn.TransformerEncoderLayer(d_model=config.HIDDEN, nhead=4)
      self.tiles  = nn.TransformerEncoderLayer(d_model=config.HIDDEN, nhead=4)
      self.proj   = nn.Linear(2*config.HIDDEN, config.HIDDEN)

   def hidden(self, obs, state=None, lens=None):
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

#class RLlibPolicy(TorchModelV2, nn.Module):
class RLlibPolicy(RecurrentNetwork, nn.Module):
   def __init__(self, *args, **kwargs):
      self.config = kwargs.pop('config')
      config = self.config

      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)

      self.lstm = BatchFirstLSTM(
            input_size=config.HIDDEN,
            hidden_size=config.HIDDEN)


      self.proj_in  = nn.Linear(3277, config.HIDDEN)
      self.proj_out = nn.Linear(config.HIDDEN, 4)
      self.val = nn.Linear(config.HIDDEN, 1)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.val.weight.new(1, self.config.HIDDEN).zero_(),
      self.val.weight.new(1, self.config.HIDDEN).zero_()]

   def forward(self, input_dict, state, lens):
      obs = input_dict['obs']
      hidden = self.proj_in(obs)
      
      #Attentional input preprocessor and batching
      lens = lens.cpu() if type(lens) == torch.Tensor else lens
      #hidden, _ = super().hidden(obs)
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

      out = self.proj_out(hidden)
      self.value = self.val(hidden)
 
      return hidden.reshape(TB, H), state

   def value_function(self):
      return self.model.value

class QMixNMMO(nmmo.Env, MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      super().__init__(self.config)

   def observation_space(self, agent: int):
       return Dict(
         {
            'obs': super().observation_space(agent)
            #'obs': gym.spaces.Box(low=-2**20, high=2**20, shape=(3276,), dtype=np.float32),
         }
       )
   
   def action_space(self, agent):
      return gym.spaces.Discrete(4)

   def step(self, actions):
      ents = list(actions.keys())
      for ent in ents:
         if ent not in self.realm.players:
            del actions[ent]
            continue

         val = actions[ent]
         move = {nmmo.action.Move: {nmmo.action.Direction: val}}
         actions[ent] = move

      obs, rewards, dones, infos = super().step(actions)

      dones['__all__'] = False

      if self.realm.tick >= 32:
         dones['__all__'] = True

      for key in obs:
         flat = []
         for ent_name, ent_attrs in obs[key].items():
            for attr_name, attr in ent_attrs.items():
               flat.append(attr.ravel())
         flat = np.concatenate(flat)
    
         obs[key] = {'obs': flat}

      for key in ents:
         if key not in obs:
            obs[key] = {'obs': 0 * self.observation_space(key)['obs'].sample()}
            rewards[key] = 0
            dones[key] = 1
            infos[key] = {} 
            

      return obs, rewards, dones, infos

class Config(nmmo.config.Small):
    NENT = 2
    EMBED = 2
    HIDDEN = 2

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

class TestQMix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()

    def test_avail_actions_qmix(self):
        Env    = QMixNMMO
        config = Config()

        env       = Env({'config': config})
        idxs      = range(1, config.NENT+1)

        obs_space = Tuple([env.observation_space(i) for i in idxs])
        act_space = Tuple([env.action_space(i) for i in idxs])

        #Functional grouping requires patch
        #grouping = lambda e: 'group1'

        grouping = {'group1': list(range(1, config.NENT+1))}

        ray.tune.registry.register_env("Neural_MMO",
            lambda config: Env(config).with_agent_groups(grouping,
            obs_space=obs_space, act_space=act_space))

        ray.rllib.models.ModelCatalog.register_custom_model(
            'nmmo_policy', RLlibPolicy)

        policies = {}
        for i in range(1):
            params = {
                "agent_id": i,
                "obs_space_dict": obs_space,
                "act_space_dict": act_space}

            policies[f'policy_{i}'] = (None, obs_space, act_space, params) 

        agent = QMixTrainer(
            env="Neural_MMO",
            config={
                'num_envs_per_worker': 1,  # test with vectorization on
                'num_gpus': 0,
                'env_config': {
                    'config': config
                },
                'framework': 'torch',
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': lambda i: 'policy_0',
                },
               'model': {
                  'custom_model': 'nmmo_policy',
                  'custom_model_config': {'config': config},
                  'max_seq_len': 16,
               },
               'simple_optimizer': False,
            })

        agent.train()  # OK if it doesn't trip the action assertion error

        agent.stop()
        ray.shutdown()


if __name__ == "__main__":
    import pytest
    import sys
    #sys.exit(pytest.main(["-v", __file__]))

    TestQMix.setUpClass()
    qmix = TestQMix()
    qmix.test_avail_actions_qmix()
    TestQMix.tearDownClass()


