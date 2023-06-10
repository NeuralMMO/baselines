# pylint: disable=protected-access

'''Manual test for running a training epoch with curriculum'''
#import wandb

# train_helper provides the dummy functions/classes to make this run
# TODO: replace these with the actual functions/classes
from curriculum.train_helper import train_on_tasks, evaluate_on_tasks, load_agent_model
from curriculum.train_helper import TaskEmbeddingGenerator
from curriculum.train_helper import SimpleTaskGenerator, OpenELMTaskGenerator
from curriculum.train_helper import SyllabusTaskSampler

# simulation participant's custom functions
from submission import custom_curriculum as cc

# if you want more task specs, use
from curriculum import manual_curriculum as mc # and use mc.task_spec

# Task spec, which is a list tuple and pickle-able, is passed around
# task_spec: (reward_to, eval_fn, kwargs)
# task_spec_with_embedding: (reward_to, eval_fn, kwargs, task_embedding)

#####################################################################
# this is the main training loop

# LLM model path, which is shared by OpenELM and the task embedding generator
LLM_CHECKPOINT = "Salesforce/codegen-350M-mono" 

# Joseph's comment: this will likely have to work off of a pretrained model
AGENT_MODEL_PATH = ''

NUM_TRAIN_TASKS = 250
NUM_TEST_TASKS = 10
NUM_NEW_TASKS = 30
epochs_per_elm_update = 10


def train_with_curriculum(use_elm=True):
  # participant's custom task spec is provided in a separate file, custom_curriculum.py
  if use_elm:
    # NOTE: passing cc.task_spec just to get some tasks out
    # CHECK ME: would it be realistic to generate the training task on the fly here?
    #   I'd probably use OpenELM to supplement the training task
    task_generator = OpenELMTaskGenerator(cc.task_spec, LLM_CHECKPOINT)
  else:
    task_generator = SimpleTaskGenerator(cc.task_spec)
  train_task_spec = task_generator.generate_tasks(NUM_TRAIN_TASKS)
  eval_task_spec = task_generator.generate_tasks(NUM_TEST_TASKS)

  # get task embeddings
  task_encoder = TaskEmbeddingGenerator(LLM_CHECKPOINT)
  task_encoder.update_context(task_generator.eval_fn_code)
  train_task_spec_with_embedding = task_encoder.get_task_embedding(train_task_spec)
  eval_task_spec_with_embedding = task_encoder.get_task_embedding(eval_task_spec)

  # initialize task sampler
  task_sampler = SyllabusTaskSampler(train_task_spec_with_embedding)

  # This will likely have to work off of a pretrained model
  agent_model = load_agent_model(AGENT_MODEL_PATH)

  # epochs that run in 8 A100 hours
  for epoch in range(30):
    # if using elm, update the training tasks
    if use_elm and epoch % epochs_per_elm_update == 9:
      print('eval fn evol!, epoch:', epoch)
      new_task_spec = task_generator.evolve_tasks(NUM_NEW_TASKS,
                                                  task_sampler.task_spec_with_embedding,
                                                  task_sampler.sample_weights)
      task_encoder.update_context(task_generator.active_fn_code)
      new_task_spec_with_embedding = task_encoder.get_task_embedding(new_task_spec)
      task_sampler.add_new_tasks(new_task_spec_with_embedding)

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
    
    #wandb.log(train_stats, eval_stats)

if __name__ == '__main__':
  train_with_curriculum()
