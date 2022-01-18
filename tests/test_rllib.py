from pdb import set_trace as T

import ray

import config
import rllib_wrapper

from main import run_tune_experiment

def setup():
   ray.shutdown()

   conf = config.baselines.Debug()
   conf.TRAINING_ITERATIONS = 1
   conf.RESTORE             = False

   return conf

def test_ppo():
   run_tune_experiment(setup(), rllib_wrapper.PPO)

def test_appo():
   run_tune_experiment(setup(), rllib_wrapper.APPO)

def test_impala():
   run_tune_experiment(setup(), rllib_wrapper.Impala)



