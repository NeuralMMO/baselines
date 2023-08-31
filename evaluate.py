# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import logging
import os
import datetime
from types import SimpleNamespace
from pathlib import Path

import torch
import pandas as pd

import pufferlib
from pufferlib.vectorization import Serial, Multiprocessing
from pufferlib.policy_store import DirectoryPolicyStore
from pufferlib.frameworks import cleanrl
from pufferlib.policy_store import DirectoryPolicyStore
import pufferlib.policy_ranker
import pufferlib.utils
import clean_pufferl

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
        vectorization=Serial if args.use_serial_vecenv else Multiprocessing,
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

    assert args.replay_save_dir is None, "Replay save dir should not be specified during policy ranking"

    # TODO: custom models will require different policy creation functions
    from reinforcement_learning import policy  # import your policy
    def make_policy(envs):
        learner_policy = policy.Baseline(
            envs,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            task_size=args.task_size
        )
        return cleanrl.Policy(learner_policy)

    # Setup the evaluator. No training during evaluation
    evaluator = clean_pufferl.CleanPuffeRL(
        device=torch.device(args.device),
        env_creator=environment.make_env_creator(args),
        env_creator_kwargs={},
        agent_creator=make_policy,
        data_dir=policy_store_dir,
        vectorization=Multiprocessing,
        num_envs=args.num_envs,
        num_cores=args.num_envs,
        num_buffers=args.num_buffers,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.selfplay_num_policies,
        batch_size=args.rollout_batch_size,
        policy_store=policy_store,
        policy_ranker=policy_ranker, # so that a new ranker is created
    )

    rank_file = os.path.join(policy_store_dir, "ranking.txt")
    with open(rank_file, "w") as f:
        pass

    while evaluator.global_step < args.train_num_steps:
        evaluator.evaluate()
        ratings = evaluator.policy_ranker.ratings()
        dataframe = pd.DataFrame(
            {
                ("Rating"): [ratings.get(n).mu for n in ratings],
                ("Policy"): ratings.keys(),
            }
        )

        with open(rank_file, "a") as f:
            f.write(
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
    """Usage: python evaluate.py -c <checkpoint_file> -s <replay_save_dir> -p <policy_store_dir>

    -c, --checkpoint: A single checkpoint file to generate replay
    -p, --policy-store-dir: Directory to load policy checkpoints from for evaluation/ranking
    -s, --replay-save-dir: Directory to save replays (Default: replays/)

    To generate replay from your checkpoint, run the following command, and replays will be saved under the replays/:
    $ python evaluate.py -c <checkpoint_file>

    To rank your checkpoints, put them together in policy_store_dir, run the following command, and the rankings will be printed out:
    $ python evaluate.py -p <policy_store_dir>

    If replay_save_dir is specified, the script will run in local mode and only use 1 environment.
    If policy_store_dir is specified, the script will NOT generate and save replays.

    TODO: generate replays using the checkpoints from policy_store_dir
    """
    logging.basicConfig(level=logging.INFO)

    # Process the evaluate.py command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint_file",
        type=str,
        default=None,
        help="A single checkpoint file to generate replay",
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
    # if eval_args.run_dir is not None and eval_args.policy_store_dir is not None:
    #     raise ValueError("Only one of checkpoint or policy-store-dir can be specified.")

    eval_args.policy_store_dir = "puf12_late"

    if eval_args.checkpoint_file is not None:
        logging.info("Generating replays from %s", eval_args.checkpoint_file)
        save_replays(eval_args.checkpoint_file, eval_args.replay_save_dir)
    elif eval_args.policy_store_dir is not None:
        logging.info("Evaluating checkpoints from %s", eval_args.policy_store_dir)
        logging.info("Replays will NOT be generated")
        rank_policies(eval_args.policy_store_dir)
    else:
        raise ValueError("Either run_dir or policy_store_dir must be specified")
