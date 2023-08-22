# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import logging
import os
import datetime
from types import SimpleNamespace
from pathlib import Path

import clean_pufferl
import pandas as pd

import pufferlib
from pufferlib.policy_store import DirectoryPolicyStore, FilePolicyRecord, MemoryPolicyStore
import pufferlib.policy_ranker
import pufferlib.utils
from pufferlib.vectorization.multiprocessing import VecEnv as MPVecEnv
from pufferlib.vectorization.serial import VecEnv as SerialVecEnv

import environment

from reinforcement_learning import config


# To check if a new replay file was generated
def get_new_files(directory, timestamp):
    new_files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            if modification_time > timestamp:
                new_files.append(file_path)
    return new_files

def save_replays(train_dir, save_dir):
    # NOTE: it uses only the latest checkpoint in the train_dir to generate replays
    #   The latest checkpoint is loaded as "learner"

    # Set up the replay directory
    assert os.path.exists(train_dir), "Train directory does not exist"
    run_name = os.path.basename(train_dir)
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    logging.info("Replays will be saved to %s", save_dir)

    # Use the local mode
    # TODO: task-condition agents when generating replays
    args = SimpleNamespace(**config.Config.asdict())
    args.replay_save_dir = save_dir
    args.num_envs = 1
    args.num_buffers = 1
    args.use_serial_vecenv = True
    args.rollout_batch_size = 1024 * 8  # CHECK ME: 1024 doesn't seem to generate replays
    args.learner_weight = 1  # use only the learner policy
    binding = environment.create_binding(args)

    # Check the policy store
    policy_store_dir = os.path.join(train_dir, "policy_store")
    if not os.path.exists(policy_store_dir):
        raise ValueError("Policy store does not exist")
    logging.info("Using policy store from %s", policy_store_dir)
    policy_store = DirectoryPolicyStore(policy_store_dir)

    # TODO: custom models will require different policy creation functions
    from reinforcement_learning import policy  # import your policy
    learner_policy = policy.Baseline.create_policy(binding, args.__dict__)

    # Setup the evaluator. No training during evaluation
    evaluator = clean_pufferl.CleanPuffeRL(
        binding=binding,
        agent=learner_policy,
        data_dir=train_dir,
        policy_store=policy_store,
        vec_backend=SerialVecEnv if args.use_serial_vecenv else MPVecEnv,
        num_envs=args.num_envs,
        num_cores=args.num_envs,
        num_buffers=args.num_buffers,
        selfplay_learner_weight=args.learner_weight,
        batch_size=args.rollout_batch_size,
    )

    # Generate replays
    start_ts = datetime.datetime.now()
    while True:
        # CHECK ME: don't know when env_done becomes True
        #   so running this loop until a new replay file is created
        evaluator.evaluate()
        # If a new replay file was created, stop
        if get_new_files(save_dir, start_ts):
            break

    evaluator.close()

def create_policy_ranker(policy_store_dir, ranker_file="openskill.pickle"):
    file = os.path.join(policy_store_dir, ranker_file)
    if os.path.exists(file):
        if os.path.exists(file + ".lock"):
            raise ValueError("Policy ranker file is locked.")
        logging.info("Using policy ranker from %s", file)
        policy_ranker = pufferlib.utils.PersistentObject(
            file,
            pufferlib.policy_ranker.OpenSkillRanker,
        )
    else:
        policy_ranker = pufferlib.utils.PersistentObject(
            file,
            pufferlib.policy_ranker.OpenSkillRanker,
            "anchor",
        )
    return policy_ranker

def rank_policies(policy_store_dir):
    # CHECK ME: can be custom models with different architectures loaded here?
    if not os.path.exists(policy_store_dir):
        raise ValueError("Policy store directory does not exist")
    if os.path.exists(os.path.join(policy_store_dir, "trainer.pt")):
        raise ValueError("Policy store directory should not contain trainer.pt")
    logging.info("Using policy store from %s", policy_store_dir)
    policy_store = DirectoryPolicyStore(policy_store_dir)
    policy_ranker = create_policy_ranker(policy_store_dir)

    # TODO: task-condition agents when generating replays
    args = SimpleNamespace(**config.Config.asdict())
    args.data_dir = policy_store_dir
    args.learner_weight = 0  # evaluate mode
    args.selfplay_num_policies = 16 + 1
    args.rollout_batch_size = 1024 * 32  # NOTE: # no ranking update until 360k steps

    assert args.replay_save_dir is None, "Replay save dir should not be specified during policy ranking"
    binding = environment.create_binding(args)

    # TODO: custom models will require different policy creation functions
    from reinforcement_learning import policy  # import your policy
    learner_policy = policy.Baseline.create_policy(binding, args.__dict__)

    # Setup the evaluator. No training during evaluation
    evaluator = clean_pufferl.CleanPuffeRL(
        binding=binding,
        agent=learner_policy,
        data_dir=policy_store_dir,
        vec_backend=SerialVecEnv if args.use_serial_vecenv else MPVecEnv,
        num_envs=args.num_envs,
        num_cores=args.num_envs,
        num_buffers=args.num_buffers,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.selfplay_num_policies,
        batch_size=args.rollout_batch_size,
        policy_store=policy_store,
        policy_ranker=policy_ranker, # so that a new ranker is created
    )

    for _ in range(20):  # 20 is arbitrary
        # CHECK ME: no ranking update until 360k steps?
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

        # CHECK ME: delete the policy_ranker lock file
        Path(evaluator.policy_ranker.lock.lock_file).unlink(missing_ok=True)

    evaluator.close()


if __name__ == "__main__":
    """Usage: python evaluate.py -r <run_dir> -s <replay_save_dir> -p <policy_store_dir>

    -r, --run-dir: Directory used for training and saving checkpoints. Usually contains run_name
    -p, --policy-store-dir: Directory to load policy checkpoints from
    -s, --replay-save-dir: Directory to save replays (Default: replays/)

    To generate replay from your training run, run the following command, and replays will be saved under the replays/:
    $ python evaluate.py -r <run_dir>

    To rank your checkpoints, run the following command, and the rankings will be printed out:
    $ python evaluate.py -p <policy_store_dir>

    If replay_save_dir is specified, the script will run in local mode and only use 1 environment.
    If policy_store_dir is specified, the script will NOT generate and save replays.

    TODO: generate replays using the checkpoints from policy_store_dir
    """
    logging.basicConfig(level=logging.INFO)

    # Process the evaluate.py command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run-dir",
        dest="run_dir",
        type=str,
        default=None,
        help="Directory used for training and saving checkpoints. Usually contains run_name",
    )
    parser.add_argument(
        "-s",
        "--replay-save-dir",
        dest="replay_save_dir",
        type=str,
        default="replays",
        help="Directory to save replays",
    )
    parser.add_argument(
        "-p",
        "--policy-store-dir",
        dest="policy_store_dir",
        type=str,
        default=None,
        help="Directory to load policy checkpoints from",
    )

    # Parse and check the arguments
    eval_args = parser.parse_args()
    if eval_args.run_dir is not None and eval_args.policy_store_dir is not None:
        raise ValueError("Only one of run_dir or policy_store_dir can be specified")

    if eval_args.run_dir is not None:
        logging.info("Generating replays from %s", eval_args.run_dir)
        save_replays(eval_args.run_dir, eval_args.replay_save_dir)
    elif eval_args.policy_store_dir is not None:
        logging.info("Evaluating checkpoints from %s", eval_args.policy_store_dir)
        logging.info("Replays will NOT be generated")
        rank_policies(eval_args.policy_store_dir)
    else:
        raise ValueError("Either run_dir or policy_store_dir must be specified")
