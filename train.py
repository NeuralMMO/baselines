import argparse
import logging
import os
import time

import clean_pufferl
import torch
from pufferlib.policy_pool import PolicyPool
from pufferlib.policy_ranker import OpenSkillRanker
from pufferlib.policy_store import DirectoryPolicyStore, PolicySelector
from pufferlib.utils import PersistentObject
from pufferlib.vectorization.multiprocessing import VecEnv as MPVecEnv
from pufferlib.vectorization.serial import VecEnv as SerialVecEnv

import nmmo_env
from lib.training_run import TrainingRun
from nmmo_policy import NmmoPolicy

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--rollout.num_cores",
      dest="num_cores",
      type=int,
      default=None,
      help="number of cores to use for training (default: num_envs)",
  )
  parser.add_argument(
      "--rollout.num_envs",
      dest="num_envs",
      type=int,
      default=4,
      help="number of environments to use for training (default: 1)",
  )
  parser.add_argument(
      "--rollout.num_buffers",
      dest="num_buffers",
      type=int,
      default=4,
      help="number of buffers to use for training (default: 4)",
  )
  parser.add_argument(
      "--rollout.batch_size",
      dest="rollout_batch_size",
      type=int,
      default=2**14,
      help="number of steps to rollout (default: 2**14)",
  )
  parser.add_argument(
      "--train.num_steps",
      dest="train_num_steps",
      type=int,
      default=10_000_000,
      help="number of steps to train (default: 10_000_000)",
  )
  parser.add_argument(
      "--train.max_epochs",
      dest="train_max_epochs",
      type=int,
      default=10_000_000,
      help="number of epochs to train (default: 10_000_000)",
  )
  parser.add_argument(
      "--train.checkpoint_interval",
      dest="checkpoint_interval",
      type=int,
      default=10,
      help="interval to save models (default: 10)",
  )
  parser.add_argument(
      "--train.run_name",
      dest="run_name",
      type=str,
      default=None,
      help="run name (default: None)",
  )
  parser.add_argument(
      "--train.runs_dir",
      dest="runs_dir",
      type=str,
      default=None,
      help="runs_dir directory (default: runs)",
  )
  parser.add_argument(
      "--train.policy_store_dir",
      dest="policy_store_dir",
      type=str,
      default=None,
      help="policy_store directory (default: runs)",
  )
  parser.add_argument(
      "--train.use_serial_vecenv",
      dest="use_serial_vecenv",
      action="store_true",
      help="use serial vecenv impl (default: False)",
  )
  parser.add_argument(
      "--train.learner_weight",
      dest="learner_weight",
      type=float,
      default=1.0,
      help="weight of learner policy (default: 1.0)",
  )
  parser.add_argument(
      "--train.max_opponent_policies",
      dest="max_opponent_policies",
      type=int,
      default=0,
      help="maximum number of opponent policies to train against (default: 0)",
  )
  parser.add_argument(
      "--wandb.project",
      dest="wandb_project",
      type=str,
      default=None,
      help="wandb project name (default: None)",
  )
  parser.add_argument(
      "--wandb.entity",
      dest="wandb_entity",
      type=str,
      default=None,
      help="wandb entity name (default: None)",
  )

  parser.add_argument(
      "--ppo.bptt_horizon",
      dest="bptt_horizon",
      type=int,
      default=8,
      help="train on bptt_horizon steps of a rollout at a time. "
      "use this to reduce GPU memory (default: 16)",
  )

  parser.add_argument(
      "--ppo.training_batch_size",
      dest="ppo_training_batch_size",
      type=int,
      default=32,
      help="number of rows in a training batch (default: 32)",
  )
  parser.add_argument(
      "--ppo.update_epochs",
      dest="ppo_update_epochs",
      type=int,
      default=4,
      help="number of update epochs to use for training (default: 4)",
  )
  parser.add_argument(
      "--ppo.learning_rate",
      dest="ppo_learning_rate",
      type=float,
      default=0.0001,
      help="learning rate (default: 0.0001)",
  )
  nmmo_env.add_args(parser)
  args = parser.parse_args()

  if args.run_name is None:
    args.run_name = f"nmmo_{time.strftime('%Y%m%d_%H%M%S')}"

  training_run = TrainingRun(args.run_name, args.runs_dir, args)
  training_run.enable_wandb(args.wandb_project, args.wandb_entity)

  binding = nmmo_env.create_binding(args)
  device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

  if args.policy_store_dir is None:
    args.policy_store_dir = training_run.data_dir()
  logging.info("Using policy store from %s", args.policy_store_dir)
  policy_store = DirectoryPolicyStore(args.policy_store_dir)

  if training_run.has_policy_checkpoint():
    logging.info(
        "Train: resuming training from %s", training_run.latest_policy_name()
    )
    pr = policy_store.get_policy(training_run.latest_policy_name())
    learner_policy = pr.policy(NmmoPolicy.create_policy, binding)
  else:
    logging.info("No policy checkpoint found. Creating new policy.")
    learner_policy = NmmoPolicy.create_policy(
        {"policy_type": "nmmo", "num_lstm_layers": 0}, binding
    )

  policy_pool = PolicyPool(
      learner_policy,
      "learner",
      num_envs=args.num_envs,
      num_agents=args.num_agents,
      num_policies=args.max_opponent_policies + 1,
      learner_weight=args.learner_weight,
  )

  ps = PolicySelector(args.max_opponent_policies, exclude_names="learner")
  ranker = PersistentObject(
      os.path.join(training_run.data_dir(), "openskill.pickle"),
      OpenSkillRanker,
      "anchor",
  )

  trainer = clean_pufferl.CleanPuffeRL(
      binding,
      learner_policy,
      policy_pool=policy_pool,
      vec_backend=SerialVecEnv if args.use_serial_vecenv else MPVecEnv,
      total_timesteps=args.train_num_steps,
      num_envs=args.num_envs,
      num_cores=args.num_cores or args.num_envs,
      num_buffers=args.num_buffers,
      batch_size=args.rollout_batch_size,
      learning_rate=args.ppo_learning_rate,
      policy_ranker=ranker,
  )

  training_run.resume_training(trainer)
  if "learner" not in ranker.ratings():
    ranker.add_policy("learner")

  while not trainer.done_training():
    sp = policy_store.select_policies(ps)
    policy_pool.update_policies(
        {p.name: p.policy(NmmoPolicy.create_policy, binding) for p in sp}
    )
    trainer.evaluate()

    trainer.train(
        update_epochs=args.ppo_update_epochs,
        bptt_horizon=args.bptt_horizon,
        batch_rows=args.ppo_training_batch_size // args.bptt_horizon,
    )

    if trainer.update % args.checkpoint_interval == 1:
      training_run.save_checkpoint(trainer)
      policy_store.add_policy(
          training_run.latest_policy_name(),
          learner_policy,
          learner_policy.metadata(),
      )
      ranker.add_policy_copy(training_run.latest_policy_name(), "learner")

  trainer.close()

# lr: 0.0001 -> 0.00001
# ratio_clip: 0.2
# dual_clip_c: 3.
# pi_loss_weight: 1.0
# v_loss_weight: 0.5
# entropy_loss_weight: 0.03 -> 0.001
# grad_clip: 1.0
# bptt_trunc_len: 16
