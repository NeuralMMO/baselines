# pylint: disable=protected-access

'''Manual test for running a training epoch with curriculum'''
import wandb

from nmmo.core.config import Config
import nmmo.task.base_predicates as prebuilt_eval_fn

import curriculum.submission_eval_fn as participant_eval_fn
import curriculum.submission_curriculum as participant_curriculum
import curriculum.manual_curriculum as eval_curriculum


# Task spec, which is a list tuple and pickle-able, is passed around
# task_spec: (reward_to, eval_fn, kwargs)
# task_spec_with_embedding: (reward_to, eval_fn, kwargs, task_embedding)

# assuming something like this
# CHECK ME: currently assuming each agent has only one task assigned during training,
#   so that we don't have to add multiple task embeddings to feed into one agent
# TODO: when given multiple tasks, can agents prioritize and/or multi-task?
#   It seems to be a research questions.
class TaskEmbeddingGenerator:
  def __init__(self, model, contexts):
    # hopefully, the model is stored locally and also used by OpenELM
    self.model = model

    # the context is the eval function definitions, both pre-built and submitted
    self.prompt_template = self._contruct_prompt_template(contexts)

  def _contruct_prompt_template(self, context):
    pass

  def _construct_prompt(self, reward_to, eval_fn, eval_fn_kwargs):
    pass

  def get_task_embedding(self, task_spec):
    task_embedding = []
    for reward_to, eval_fn, kwargs in task_spec:
      prompt = self._construct_prompt(reward_to, eval_fn, kwargs)
      embedding = self.model.get_hidden_layer(prompt) # something like this? 
      task_embedding.append(embedding)
    return task_embedding


# to be provided by Joseph
#from cleanrl_ppo_lstm import train_on_tasks, evaluate_on_tasks
def train_on_tasks(agent_mode, task_spec_with_embedding):
  pass

def evaluate_on_tasks(agent_mode, task_spec_with_embedding):
  pass

def load_agent_model(model_path):
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
  def add_new_tasks(self, task_spec_with_embedding):
    pass

  def sample_tasks(self, num_tasks):
    pass # return task_spec_with_embedding

  def update(self, task_spec_with_embedding, train_stats):
    pass


#####################################################################
# this is the main training loop

# LLM model path, which is shared by OpenELM and the task embedding generator
LLM_MODEL_PATH = '' 

# Joseph's comment: this will likely have to work off of a pretrained model
AGENT_MODEL_PATH = ''
agent_model = load_agent_model(AGENT_MODEL_PATH)

def test_curriculum_learning(config: Config, use_elm=True):
  # eval fn definitions are provided as the context for LLM
  task_encoder = TaskEmbeddingGenerator(LLM_MODEL_PATH, [prebuilt_eval_fn, participant_eval_fn])

  train_task_spec = participant_curriculum.task_spec
  train_task_spec_with_embedding = task_encoder.get_task_embedding(train_task_spec)
  task_sampler = SyllabusTaskSampler(train_task_spec_with_embedding)

  eval_task_spec = eval_curriculum.task_spec[:100] # for example
  eval_task_spec_with_embedding = task_encoder.get_task_embedding(eval_task_spec)

  if use_elm:
    task_generator = OpenELMTaskGenerator(LLM_MODEL_PATH)

  # epochs that run in 8 A100 hours
  for epoch in range(3):
    if use_elm and epoch % epochs_per_elm_update == 0:
      # Coordinate with Ryan to figure out how to do this
      # The one issue is that changing the tasks will
      # mess up the existing curriculumg heuristics
      # Possibly ELM runs infrequently? Make sense,
      # we should get a lot out of each ELM update
      train_task_spec = task_generator.permute_tasks(train_task_spec)
      train_task_spec_with_embedding = task_encoder.get_task_embedding(train_task_spec)
      task_sampler.add_new_tasks(train_task_spec_with_embedding)

      # Sample one task per team. Note that the trainer runs many
      # envs in parallel, and these are vectorized
      batch_task_spec_with_embedding = task_sampler.sample_tasks(num_tasks=64)

    # Joseph will provide an API for training and evaluation with CleanRL
    # David has specific ideas on how to pass around the embedding in and out of the env
    agent_model, train_stats = train_on_tasks(agent_model, batch_task_spec_with_embedding)

    # Sampler updates its heuristics based on performance delta
    task_sampler.update(batch_task_spec_with_embedding, train_stats)

    # We also want to check performance on held-out tasks
    # This is just a split from the heuristics and is separate from
    # the competition evaluation tasks used to score models
    eval_stats = evaluate_on_tasks(agent_model, eval_task_spec_with_embedding)
    wandb.log(train_stats, eval_stats)
