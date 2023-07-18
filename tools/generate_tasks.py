import argparse

from curriculum.task_encoder import TaskEncoder
from elm_for_nmmo.elm_curriculum_gen import SimpleTaskGenerator
from submission import custom_curriculum as cc

LLM_CHECKPOINT = "Salesforce/codegen-350M-mono"

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--num_tasks",
      dest="num_tasks",
      type=str,
      default=1,
      help="number of tasks to generate (default: None)",
  )
  parser.add_argument(
      "--path",
      dest="path",
      type=str,
      default="tasks.pkl",
      help="path to save the tasks (default: tasks.pkl)",
  )

  args = parser.parse_args()

  task_generator = SimpleTaskGenerator(cc.task_spec)
  train_task_spec = task_generator.generate_tasks(args.num_tasks)

  task_encoder = TaskEncoder(LLM_CHECKPOINT, cc, batch_size=2)
  task_encoder.get_task_embedding(train_task_spec, save_to_file=args.path)
