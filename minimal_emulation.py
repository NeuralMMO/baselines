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
from ray import rllib
from ray.tune import register_env
from ray.rllib.agents.ppo import PPOTrainer, DDPPOTrainer
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

from neural import policy

class SimpleAtns(policy.Simple):
   def __init__(self, config):
      super().__init__(config)
      self.proj_out = nn.Linear(config.HIDDEN, 304)

   def output(self, hidden, entityLookup):
      return self.proj_out(hidden)

class RLlibPolicy(TorchModelV2, nn.Module):
   def __init__(self, *args, **kwargs):
      self.config = kwargs.pop('config')
      config = self.config

      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)

      self.model = SimpleAtns(self.config)

   def forward(self, input_dict, state, seq_lens):
      obs = nmmo.emulation.unpack_obs(self.config, input_dict['obs'])

      #logitDict, state = self.model(obs, state, seq_lens) 
      actions, state = self.model(obs, state, seq_lens) 

      return actions, state

      logits = []
      for atnKey, atn in sorted(logitDict.items()):
         for argKey, arg in sorted(atn.items()):
            logits.append(arg)
            return torch.cat(logits, dim=1), state  

   def value_function(self):
      return self.model.value

   def attention(self):
      return self.model.attn

class Config(nmmo.config.Small):
    EMULATE_FLAT_OBS      = True
    EMULATE_FLAT_ATN      = True
    EMULATE_CONST_POP     = True
    EMULATE_CONST_HORIZON = 32

    HIDDEN = 10
    EMBED  = 10
    NENT   = 2

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

class Env(nmmo.Env, rllib.MultiAgentEnv):
    def __init__(self, config):
        self.config = config['config']
        super().__init__(self.config)

def test_rllib_integration():
    config    = Config()
    env       = Env({'config': config})
    idxs      = range(1, config.NENT+1)

    ray.tune.registry.register_env(
            "Neural_MMO",
            lambda config: Env(config))

    ray.rllib.models.ModelCatalog.register_custom_model(
            'nmmo_policy',
            RLlibPolicy)

    policies = {}
    for i in range(1):
        params = {
            "agent_id": i,
            "obs_space_dict": env.observation_space(i),
            "act_space_dict": env.action_space(i)}


        policies[f'policy_{i}'] = (None,
                env.observation_space(i),
                env.action_space(i), params) 

    agent = DDPPOTrainer(
         env="Neural_MMO",
         config={
             'num_envs_per_worker': 1,
             'num_gpus_per_worker': 0,
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

    agent.train()
    agent.stop()

if __name__ == "__main__":
    ray.init()
    test_rllib_integration()
    ray.shutdown()


