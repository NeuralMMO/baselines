"""this is a place holder for dummy functions"""
# pylint: disable=all

import numpy as np
import nmmo

######################################################################
# to be provided by Joseph
#from cleanrl_ppo_lstm import train_on_tasks, evaluate_on_tasks
class DummyAgent:
  pass

dummy_stat = {'stat': np.nan}

def train_on_tasks(agent_model, curriculm_file_path):
  # env.__init__() checks whether the curriculum file is present
  config = nmmo.config.Default()
  config.CURRICULUM_FILE_PATH = curriculm_file_path
  env = nmmo.Env(config)

  # sampling training tasks is done via env.reset(sample_training_tasks=True)
  env.reset(sample_training_tasks=True)

  # check the tasks have embedding
  for task in env.tasks:
    assert np.sum(task.embedding) != 0, 'task embedding is not set'

  print('train_on_tasks(): using the curriculum file with the embedding')

  return DummyAgent(), dummy_stat

def evaluate_on_tasks(agent_model, curriculm_file_path):
  return dummy_stat

def load_agent_model(model_path):
  return DummyAgent()
