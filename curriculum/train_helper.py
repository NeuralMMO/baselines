"""this is a place holder for dummy functions"""
# pylint: disable=all

import random
import numpy as np


######################################################################
# to be provided by Joseph
#from cleanrl_ppo_lstm import train_on_tasks, evaluate_on_tasks
class DummyAgent:
  pass

dummy_stat = {'stat': np.nan}

def train_on_tasks(agent_model, task_spec_with_embedding):
  return DummyAgent(), dummy_stat

def evaluate_on_tasks(agent_model, task_spec_with_embedding):
  return dummy_stat

def load_agent_model(model_path):
  return DummyAgent()


######################################################################
# Ryan's syllabus task sampler, assuming something like this
class SyllabusTaskSampler:
  def __init__(self, task_spec_with_embedding):
    self.task_spec_with_embedding = task_spec_with_embedding
    # something like this? to indicate which tasks to focus on currently
    self.sample_weights = None
    self._dummy_update_weight()
    print('Number of tasks:', len(self.task_spec_with_embedding))

  def _dummy_update_weight(self):
    num_task = len(self.task_spec_with_embedding)
    weight = np.random.random(num_task)
    self.sample_weights = weight / np.sum(weight)

  # just adding new tasks, not replacing the whole tasks
  def add_new_tasks(self, task_spec_with_embedding):
    # TODO: deduplication, the embeddings must be different
    self.task_spec_with_embedding += task_spec_with_embedding
    print('Number of tasks:', len(self.task_spec_with_embedding))

  def sample_tasks(self, num_tasks):
    # TODO: return meaningful task specs
    return random.choices(self.task_spec_with_embedding, k=num_tasks)

  def update(self, task_spec_with_embedding, train_stats):
    self._dummy_update_weight()
