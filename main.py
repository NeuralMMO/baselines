'''Main file for NMMO baselines

rllib_wrapper.py contains all necessary RLlib wrappers to train and
evaluate capable policies on Neural MMO as well as rendering,
logging, and visualization tools.

TODO: Still need to add per pop / diff task demo and Tiled support

Associated docs and tutorials are hosted on neuralmmo.github.io.'''
from pdb import set_trace as T

from fire import Fire
from copy import deepcopy
from operator import attrgetter
import os

import numpy as np
import torch

import ray
from ray import rllib, tune
from ray.tune import CLIReporter
from ray.tune.integration.wandb import WandbLoggerCallback

import nmmo

import rllib_wrapper as wrapper
import config as base_config
from config import scale


class ConsoleLog(CLIReporter):
   def report(self, trials, done, *sys_info):
      os.system('cls' if os.name == 'nt' else 'clear') 
      print(nmmo.motd + '\n')
      super().report(trials, done, *sys_info)


def run_tune_experiment(config, trainer_wrapper, rllib_env=wrapper.RLlibEnv):
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
   ray.tune.registry.register_env("Neural_MMO",
         lambda config: rllib_env(config))

   rllib.models.ModelCatalog.register_custom_model(
         'godsword', wrapper.RLlibPolicy)

   mapPolicy = lambda agentID : 'policy_{}'.format(
         agentID % config.NPOLICIES)

   policies = {}
   env = nmmo.Env(config)
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
      'train_batch_size': config.TRAIN_BATCH_SIZE,
      'rollout_fragment_length': config.ROLLOUT_FRAGMENT_LENGTH,
      'num_sgd_iter': config.NUM_SGD_ITER,
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
      'callbacks': wrapper.RLlibLogCallbacks,
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
      self.trainer_wrapper = wrapper.PPO

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
