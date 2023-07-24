# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import logging
import os
import time

import clean_pufferl
import pandas as pd
from pufferlib.policy_store import DirectoryPolicyStore, FilePolicyRecord, MemoryPolicyStore
from pufferlib.vectorization.multiprocessing import VecEnv as MPVecEnv
from pufferlib.vectorization.serial import VecEnv as SerialVecEnv

import nmmo_env
import nmmo_policy
import pufferlib

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--eval.run_name",
      dest="run_name",
      type=str,
      default=f"nmmo_{time.strftime('%Y%m%d_%H%M%S')}",
      help="run name (default: None)",
  )
  parser.add_argument(
      "--eval.runs_dir",
      dest="runs_dir",
      type=str,
      default="/tmp/nmmo_eval",
      help="runs_dir directory (default: runs)",
  )
  parser.add_argument(
      "--eval.policy_store_dir",
      dest="policy_store_dir",
      type=str,
      default=None,
      help="policy_store directory (default: runs)",
  )
  parser.add_argument(
      "--eval.num_rounds",
      dest="num_rounds",
      type=int,
      default=1,
      help="number of rounds to use for evaluation (default: 1)",
  )
  parser.add_argument(
      "--rollout.num_envs",
      dest="num_envs",
      type=int,
      default=1,
      help="number of environments to use for training (default: 1)",
  )
  parser.add_argument(
      "--eval.use_serial_vecenv",
      dest="use_serial_vecenv",
      action="store_true",
      help="use serial vecenv impl (default: False)",
  )
  parser.add_argument(
      "--eval.num_policies",
      dest="num_policies",
      type=int,
      default=2,
      help="number of policies to use for evaluation (default: 1)",
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
  nmmo_env.add_args(parser)
  nmmo_policy.add_args(parser)
  args = parser.parse_args()

  args = parser.parse_args()

  run_dir = os.path.join(args.runs_dir, args.run_name)
  os.makedirs(run_dir, exist_ok=True)

  logging.info("Evaluation run: %s (%s)", args.run_name, run_dir)
  logging.info("Training args: %s", args)
  binding = nmmo_env.create_binding(args)

  if args.policy_store_dir is not None:
    logging.info("Using policy store from %s", args.policy_store_dir)
    policy_store = DirectoryPolicyStore(args.policy_store_dir)

  # TODO: only pass the policy_args
  learner_policy = nmmo_policy.NmmoPolicy.create_policy(binding, args.__dict__)

  policy_selector = pufferlib.policy_ranker.PolicySelector(args.num_policies)

  evaluator = clean_pufferl.CleanPuffeRL(
      binding=binding,
      agent=learner_policy,
      data_dir=run_dir,
      exp_name=args.run_name,
      policy_store=policy_store,
      wandb_entity=args.wandb_entity,
      wandb_project=args.wandb_project,
      wandb_extra_data=args,
      vec_backend=SerialVecEnv if args.use_serial_vecenv else MPVecEnv,
      num_envs=args.num_envs,
      num_cores=args.num_envs,
      selfplay_learner_weight=0,
      selfplay_num_policies=args.num_policies + 1,
      policy_selector=policy_selector,
      batch_size=1024,
  )

  while True:
    evaluator.evaluate()
    ratings = evaluator.policy_ranker.ratings()
    dataframe = pd.DataFrame(
        {
            ("Rating"): [ratings.get(n).mu for n in ratings],
            ("Policy"): ratings.keys(),
        }
    )

    print(
        "\n\n"
        + dataframe.round(2)
        .sort_values(by=["Rating"], ascending=False)
        .to_string(index=False)
        + "\n\n"
    )

  evaluator.close()

# lr: 0.0001 -> 0.00001
# ratio_clip: 0.2
# dual_clip_c: 3.
# pi_loss_weight: 1.0
# v_loss_weight: 0.5
# entropy_loss_weight: 0.03 -> 0.001
# grad_clip: 1.0
# bptt_trunc_len: 16
