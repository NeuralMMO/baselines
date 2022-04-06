'''Main file for NMMO baselines

rllib_wrapper.py contains all necessary RLlib wrappers to train and
evaluate capable policies on Neural MMO as well as rendering,
logging, and visualization tools.

TODO: Still need to add per pop / diff task demo and Tiled support

Associated docs and tutorials are hosted on neuralmmo.github.io.'''
from pdb import set_trace as T
from collections import defaultdict
from copy import deepcopy
from operator import attrgetter
import numpy as np
import os

from fire import Fire
from tqdm import tqdm

import torch
from torch import nn

import ray
from ray import rllib, tune
from ray.tune import CLIReporter
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

import nmmo

from neural import io, subnets, policy
from scripted import baselines

import config as base_config
from config import scale


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

class ConsoleLog(CLIReporter):
   def report(self, trials, done, *sys_info):
      #os.system('cls' if os.name == 'nt' else 'clear') 
      print(nmmo.motd + '\n')
      super().report(trials, done, *sys_info)

def run_tune_experiment(config, trainer_wrapper, rllib_env=nmmo.integrations.rllib_env_cls()):
   '''Ray[RLlib, Tune] integration for Neural MMO

   Setup custom environment, observations/actions, policies,
   and parallel training/evaluation'''

   #Round and round the num_threads flags go
   #Which are needed nobody knows!
   torch.set_num_threads(1)
   os.environ['MKL_NUM_THREADS']     = '1'
   os.environ['OMP_NUM_THREADS']     = '1'
   os.environ['NUMEXPR_NUM_THREADS'] = '1'
 
   ray.init(local_mode=config.LOCAL_MODE)

   #Register custom env and policies
   grouping = lambda e: 'group1'
   #{'group1': [i for i in range(1, config.NENT+1)]}
   ray.tune.registry.register_env("Neural_MMO",
         lambda config: rllib_env(config))

   rllib.models.ModelCatalog.register_custom_model(
         'godsword', RLlibPolicy)

   mapPolicy = lambda agentID : 'policy_{0}'

   #mapPolicy = lambda agentID : 'policy_{}'.format(
   #      agentID % config.NPOLICIES)

   policies = {}
   env = rllib_env({'config': config})
   for i in range(config.NPOLICIES):
      params = {
            "agent_id": i,
            "obs_space_dict": env.observation_space(i),
            "act_space_dict": env.action_space(i)}
      key           = mapPolicy(i)
      policies[key] = (None, env.observation_space(i), env.action_space(i), params)

   #Evaluation config
   eval_config = deepcopy(config)
   eval_config.EVALUATE = True
   eval_config.AGENTS   = eval_config.EVAL_AGENTS

   trainer_cls, extra_config = trainer_wrapper(config)

   #Create rllib config
   rllib_config = {
      'num_workers': config.NUM_WORKERS,
      'num_gpus_per_worker': config.NUM_GPUS_PER_WORKER,
      'num_gpus': config.NUM_GPUS,
      'num_envs_per_worker': 1,
      'simple_optimizer': True,
      'rollout_fragment_length': config.ROLLOUT_FRAGMENT_LENGTH,
      'timesteps_per_iteration': 10,
      'framework': 'torch',
      'horizon': np.inf,
      'soft_horizon': False, 
      'no_done_at_end': False,
      'env': 'Neural_MMO',
      'env_config': {
         'config': config
      },
      'evaluation_config': {
         'env_config': {
            'config': eval_config
         },
      },
      'multiagent': {
         'policies': policies,
         'policy_mapping_fn': mapPolicy,
         'count_steps_by': 'agent_steps'
      },
      'model': {
         'custom_model': 'godsword',
         'custom_model_config': {'config': config},
         'max_seq_len': config.LSTM_BPTT_HORIZON
      },
      'render_env': config.RENDER,
      'callbacks': RLlibLogCallbacks,
      'evaluation_interval': config.EVALUATION_INTERVAL,
      'evaluation_num_episodes': config.EVALUATION_NUM_EPISODES,
      'evaluation_num_workers': config.EVALUATION_NUM_WORKERS,
      'evaluation_parallel_to_training': config.EVALUATION_PARALLEL,
   }

   #Alg-specific params
   rllib_config = {**rllib_config, **extra_config}
 
   restore     = None
   config_name = config.__class__.__name__
   algorithm   = trainer_cls.name()
   if config.RESTORE:
      if config.RESTORE_ID:
         config_name = '{}_{}'.format(config_name, config.RESTORE_ID)

      restore   = '{0}/{1}/{2}/checkpoint_{3:06d}/checkpoint-{3}'.format(
            config.EXPERIMENT_DIR, algorithm, config_name, config.RESTORE_CHECKPOINT)

   callbacks = []
   wandb_api_key = 'wandb_api_key'
   if os.path.exists(wandb_api_key):
       callbacks=[WandbLoggerCallback(
               project = 'NeuralMMO',
               api_key_file = 'wandb_api_key',
               log_config = False)]
   else:
       print('Running without WanDB. Create a file baselines/wandb_api_key and paste your API key to enable')

   tune.run(trainer_cls,
      config    = rllib_config,
      name      = trainer_cls.name(),
      verbose   = config.LOG_LEVEL,
      stop      = {'training_iteration': config.TRAINING_ITERATIONS},
      restore   = restore,
      resume    = config.RESUME,
      local_dir = config.EXPERIMENT_DIR,
      keep_checkpoints_num = config.KEEP_CHECKPOINTS_NUM,
      checkpoint_freq = config.CHECKPOINT_FREQ,
      checkpoint_at_end = True,
      trial_dirname_creator = lambda _: config_name,
      progress_reporter = ConsoleLog(),
      reuse_actors = True,
      callbacks=callbacks,
      )


class CLI():
   '''Neural MMO CLI powered by Google Fire

   Main file for the RLlib demo included with Neural MMO.

   Usage:
      python main.py <COMMAND> --config=<CONFIG> --ARG1=<ARG1> ...

   The User API documents core env flags. Additional config options specific
   to this demo are available in projekt/config.py. 

   The --config flag may be used to load an entire group of options at once.
   Select one of the defaults from projekt/config.py or write your own.
   '''
   def __init__(self, **kwargs):
      if 'help' in kwargs:
         return 

      config = 'baselines.Medium'
      if 'config' in kwargs:
          config = kwargs.pop('config')

      config = attrgetter(config)(base_config)()
      config.override(**kwargs)

      if 'scale' in kwargs:
          config_scale = kwargs.pop('scale')
          config = getattr(scale, config_scale)()
          config.override(config_scale)

      assert hasattr(config, 'NUM_GPUS'), 'Missing NUM_GPUS (did you specify a scale?)'
      assert hasattr(config, 'NUM_WORKERS'), 'Missing NUM_WORKERS (did you specify a scale?)'
      assert hasattr(config, 'EVALUATION_NUM_WORKERS'), 'Missing EVALUATION_NUM_WORKERS (did you specify a scale?)'
      assert hasattr(config, 'EVALUATION_NUM_EPISODES'), 'Missing EVALUATION_NUM_EPISODES (did you specify a scale?)'

      self.config = config
      self.trainer_wrapper = PPO

   def generate(self, **kwargs):
      '''Manually generates maps using the current --config setting'''
      nmmo.MapGenerator(self.config).generate_all_maps()

   def train(self, **kwargs):
      '''Train a model using the current --config setting'''
      run_tune_experiment(self.config, self.trainer_wrapper)

   def evaluate(self, **kwargs):
      '''Evaluate a model against EVAL_AGENTS models'''
      self.config.TRAINING_ITERATIONS     = 0
      self.config.EVALUATE                = True
      self.config.EVALUATION_NUM_WORKERS  = self.config.NUM_WORKERS
      self.config.EVALUATION_NUM_EPISODES = self.config.NUM_WORKERS

      run_tune_experiment(self.config, self.trainer_wrapper)

   def render(self, **kwargs):
      '''Start a WebSocket server that autoconnects to the 3D Unity client'''
      self.config.RENDER                  = True
      self.config.NUM_WORKERS             = 1
      self.evaluate(**kwargs)


if __name__ == '__main__':
   def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

   from fire import core
   core.Display = Display
   Fire(CLI)
