# pylint: disable=protected-access

'''Manual test for running a training epoch with curriculum'''
import wandb

from nmmo.core.config import Config

from curriculum.heuristic_task_gen import HeuristicTaskGenerator

# to be provided by Joseph
#from cleanrl_ppo_lstm import train_on_tasks, evaluate_on_tasks
def train_on_tasks():
  pass

def evaluate_on_tasks():
  pass

def load_model(pretrained_model_path):
  pass

# Nishaanth's OpenELM task generator
#from curriculum.openelm_task_gen.py_import OpenELMTaskGenerator
class OpenELMTaskGenerator:
  def permute_tasks(): # please suggest better name
    pass

epochs_per_elm_update = 10

# Ryan's syllabus task sampler
#from curriculum.syllabus_task_sampler.py_import SyllabusTaskSampler
class SyllabusTaskSampler:
  def update_tasks(): # please suggest better name
    pass

  def sample_tasks(self, num_tasks):
    pass

  def update(self, batch_of_tasks, train_stats):
    pass


def test_curriculum_learning(config: Config, use_elm=True):
  generator = HeuristicTaskGenerator(config)
  train_tasks = generator.generate_tasks(num_tasks=25_000)
  eval_tasks = generator.generate_tasks(num_tasks=1_000)

  if use_elm:
    generator = OpenELMTaskGenerator(train_tasks)

  sampler = SyllabusTaskSampler(train_tasks)

  # This will likely have to work off of a pretrained model
  model = load_model(pretrained_model_path='')

  for epoch in range(3): # epochs that run in 8 A100 hours
    if use_elm and epoch % epochs_per_elm_update == 0:
      # Coordinate with Ryan to figure out how to do this
      # The one issue is that changing the tasks will
      # mess up the existing curriculumg heuristics
      # Possibly ELM runs infrequently? Make sense,
      # we should get a lot out of each ELM update
      train_tasks = generator.permute_tasks(train_tasks)
      sampler.update_tasks(train_tasks)

      # Sample one task per team. Note that the trainer runs many
      # envs in parallel, and these are vectorized
      batch_of_tasks = sampler.sample_tasks(num_tasks=64)

    # Joseph will provide an API for training and evaluation with CleanRL
    model, train_stats = train_on_tasks(model, batch_of_tasks)

    # Sampler updates its heuristics based on performance delta
    sampler.update(batch_of_tasks, train_stats)

    # We also want to check performance on held-out tasks
    # This is just a split from the heuristics and is separate from
    # the competition evaluation tasks used to score models
    eval_stats = evaluate_on_tasks(model)
    wandb.log(train_stats, eval_stats)
