from pdb import set_trace as T
from collections import defaultdict
import numpy as np

from tqdm import tqdm

import torch
from torch import nn

import ray
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

import nmmo

from neural import io, subnets, policy
from scripted import baselines

def PPO(config):
   from ray.rllib.agents.ppo.ppo import PPOTrainer
   class PPO(Trainer, PPOTrainer): pass
   extra_config = {
            'train_batch_size': config.TRAIN_BATCH_SIZE,
            'sgd_minibatch_size': config.SGD_MINIBATCH_SIZE,
            'num_sgd_iter': 1}
   return PPO, extra_config

def APPO(config):
   from ray.rllib.agents.ppo.appo import APPOTrainer
   class APPO(Trainer, APPOTrainer): pass
   return APPO, {}

def DDPPO(config):
   from ray.rllib.agents.ppo.ddppo import DDPPOTrainer
   class DDPPO(Trainer, DDPPOTrainer): pass
   extra_config = {
           'sgd_minibatch_size': config.SGD_MINIBATCH_SIZE,
           'num_sgd_iter': 1,
           'num_gpus_per_worker': 0,
           'num_gpus': 0}
   return DDPPO, extra_config

def Impala(config):
   from ray.rllib.agents.impala.impala import ImpalaTrainer
   class Impala(Trainer, ImpalaTrainer): pass
   return Impala, {}

def QMix(config):
   from ray.rllib.agents.qmix.qmix import QMixTrainer
   class QMix(Trainer, QMixTrainer): pass
   return QMix, {}

###############################################################################
### Policy integration
class RLlibPolicy(RecurrentNetwork, nn.Module):
   '''Wrapper class for using our baseline models with RLlib'''
   #def __init__(self, observation_space, action_space, config):
   def __init__(self, *args, **kwargs):
      #self.config = config
      config = kwargs.pop('config')
      self.config = config 
      super().__init__(*args, **kwargs)
      nn.Module.__init__(self)

      self.input  = io.Input(config,
            embeddings=io.MixedEmbedding,
            attributes=subnets.SelfAttention)

      if config.EMULATE_FLAT_ATN:
         self.output = nn.Linear(self.config.HIDDEN, 304)
      else:
         self.output = io.Output(config)

      self.model  = policy.Recurrent(self.config)
      self.valueF = nn.Linear(config.HIDDEN, 1)

   #Initial hidden state for RLlib Trainer
   def get_initial_state(self):
      return [self.valueF.weight.new(1, self.config.HIDDEN).zero_(),
              self.valueF.weight.new(1, self.config.HIDDEN).zero_()]

   def forward(self, input_dict, state, seq_lens):
      obs = input_dict['obs']
      if self.config.EMULATE_FLAT_OBS:
         obs = nmmo.emulation.unpack_obs(self.config, obs)

      entityLookup  = self.input(obs)
      hidden, state = self.model(entityLookup, state, seq_lens)
      self.value    = self.valueF(hidden).squeeze(1)

      if self.config.EMULATE_FLAT_ATN:
         output = self.output(hidden)
         return output, state

      #Flatten structured logits for RLlib
      #TODO: better to use the space directly here in case of missing keys
      logits = []
      output = self.output(hidden, entityLookup)
      for atnKey, atn in sorted(output.items()):
         for argKey, arg in sorted(atn.items()):
            logits.append(arg)

      return torch.cat(logits, dim=1), state

   def value_function(self):
      return self.value

   def attention(self):
      return self.attn


###############################################################################
### Logging integration
class RLlibLogCallbacks(DefaultCallbacks):
   def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
      assert len(base_env.envs) == 1, 'One env per worker'
      env    = base_env.envs[0]#.env

      inv_map = {agent.policyID: agent for agent in env.config.AGENTS}

      stats      = env.terminal()['Stats']
      policy_ids = stats.pop('PolicyID')
 
      for key, vals in stats.items():
         policy_stat = defaultdict(list)

         # Per-population metrics
         for policy_id, v in zip(policy_ids, vals):
             policy_stat[policy_id].append(v)

         for policy_id, vals in policy_stat.items():
             policy = inv_map[policy_id].__name__

             k = f'{policy}_{policy_id}_{key}'
             episode.custom_metrics[k] = np.mean(vals)

      if not env.config.EVALUATE:
         return 

      episode.custom_metrics['Raw_Policy_IDs']   = policy_ids
      episode.custom_metrics['Raw_Task_Rewards'] = stats['Task_Reward']


###############################################################################
### Custom tournament evaluation computes SR
class Trainer:
   def __init__(self, config, env=None, logger_creator=None):
      super().__init__(config, env, logger_creator)
      self.env_config = config['env_config']['config']

      agents = self.env_config.EVAL_AGENTS

      err = 'Meander not in EVAL_AGENTS. Specify another agent to anchor to SR=0'
      assert baselines.Meander in agents, err
      self.sr = nmmo.OpenSkillRating(agents, baselines.Combat)

   @classmethod
   def name(cls):
      return cls.__bases__[0].__name__

   def post_mean(self, stats):
      for key, vals in stats.items():
          if type(vals) == list:
              stats[key] = np.mean(vals)

   def train(self):
      stats = super().train()
      self.post_mean(stats['custom_metrics'])
      return stats

   def evaluate(self):
      return {}
      stat_dict = super().evaluate()
      stats = stat_dict['evaluation']['custom_metrics']

      if __debug__:
         err = 'Missing evaluation key. Patch RLlib as per the installation guide'
         assert 'Raw_Policy_IDs' in stats, err

      policy_ids   = stats.pop('Raw_Policy_IDs')
      task_rewards = stats.pop('Raw_Task_Rewards')

      for ids, scores in zip(policy_ids, task_rewards):
          ratings = self.sr.update(policy_ids=ids, scores=scores)

          for pop, (agent, rating) in enumerate(ratings.items()):
              key = f'SR_{agent.__name__}_{pop}'

              if key not in stats:
                  stats[key] = []
 
              stats[key] = rating.mu
        
      return stat_dict

###############################################################################
### Custom overlays to hook in with the client
class RLlibOverlayRegistry(nmmo.OverlayRegistry):
   '''Host class for RLlib Map overlays'''
   def __init__(self, realm):
      super().__init__(realm.config, realm)

      self.overlays['values']       = Values
      self.overlays['attention']    = Attention
      self.overlays['tileValues']   = TileValues
      self.overlays['entityValues'] = EntityValues

class RLlibOverlay(nmmo.Overlay):
   '''RLlib Map overlay wrapper'''
   def __init__(self, config, realm, trainer, model):
      super().__init__(config, realm)
      self.trainer = trainer
      self.model   = model

class Attention(RLlibOverlay):
   def register(self, obs):
      '''Computes local attentional maps with respect to each agent'''
      tiles      = self.realm.realm.map.tiles
      players    = self.realm.realm.players

      attentions = defaultdict(list)
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         player = players[playerID]
         r, c   = player.pos

         rad     = self.config.NSTIM
         obTiles = self.realm.realm.map.tiles[r-rad:r+rad+1, c-rad:c+rad+1].ravel()

         for tile, a in zip(obTiles, self.model.attention()[idx]):
            attentions[tile].append(float(a))

      sz    = self.config.TERRAIN_SIZE
      data  = np.zeros((sz, sz))
      for r, tList in enumerate(tiles):
         for c, tile in enumerate(tList):
            if tile not in attentions:
               continue
            data[r, c] = np.mean(attentions[tile])

      colorized = nmmo.overlay.twoTone(data)
      self.realm.register(colorized)

class Values(RLlibOverlay):
   def update(self, obs):
      '''Computes a local value function by painting tiles as agents
      walk over them. This is fast and does not require additional
      network forward passes'''
      players = self.realm.realm.players
      for idx, playerID in enumerate(obs):
         if playerID not in players:
            continue
         r, c = players[playerID].base.pos
         self.values[r, c] = float(self.model.value_function()[idx])

   def register(self, obs):
      colorized = nmmo.overlay.twoTone(self.values[:, :])
      self.realm.register(colorized)

def zeroOb(ob, key):
   for k in ob[key]:
      ob[key][k] *= 0

class GlobalValues(RLlibOverlay):
   '''Abstract base for global value functions'''
   def init(self, zeroKey):
      if self.trainer is None:
         return

      print('Computing value map...')
      model     = self.trainer.get_policy('policy_0').model
      obs, ents = self.realm.dense()
      values    = 0 * self.values

      #Compute actions to populate model value function
      BATCH_SIZE = 128
      batch = {}
      final = list(obs.keys())[-1]
      for agentID in tqdm(obs):
         ob             = obs[agentID]
         batch[agentID] = ob
         zeroOb(ob, zeroKey)
         if len(batch) == BATCH_SIZE or agentID == final:
            self.trainer.compute_actions(batch, state={}, policy_id='policy_0')
            for idx, agentID in enumerate(batch):
               r, c         = ents[agentID].base.pos
               values[r, c] = float(self.model.value_function()[idx])
            batch = {}

      print('Value map computed')
      self.colorized = nmmo.overlay.twoTone(values)

   def register(self, obs):
      print('Computing Global Values. This requires one NN pass per tile')
      self.init()

      self.realm.register(self.colorized)

class TileValues(GlobalValues):
   def init(self, zeroKey='Entity'):
      '''Compute a global value function map excluding other agents. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)

class EntityValues(GlobalValues):
   def init(self, zeroKey='Tile'):
      '''Compute a global value function map excluding tiles. This
      requires a forward pass for every tile and will be slow on large maps'''
      super().init(zeroKey)

