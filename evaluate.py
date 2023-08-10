# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import logging
import os
import time

import clean_pufferl
import pandas as pd

import pufferlib
from pufferlib.policy_store import DirectoryPolicyStore, FilePolicyRecord, MemoryPolicyStore
from pufferlib.vectorization.multiprocessing import VecEnv as MPVecEnv
from pufferlib.vectorization.serial import VecEnv as SerialVecEnv

import environment
import config

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  args = config.create_config(config.LocalConfig)

  run_dir = os.path.join(args.runs_dir, args.run_name)
  os.makedirs(run_dir, exist_ok=True)

  logging.info("Evaluation run: %s (%s)", args.run_name, run_dir)
  logging.info("Training args: %s", args)
  binding = environment.create_binding(args)

  if args.policy_store_dir is not None:
    logging.info("Using policy store from %s", args.policy_store_dir)
    policy_store = DirectoryPolicyStore(args.policy_store_dir)

  # TODO: only pass the policy_args
  learner_policy = policy.Baseline.create_policy(binding, args.__dict__)

  policy_selector = pufferlib.policy_ranker.PolicySelector(args.eval_num_policies)

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
      selfplay_learner_weight=0, # @daveey: is this correct? Seems to break policy pool
      selfplay_num_policies=args.eval_num_policies + 1,
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
