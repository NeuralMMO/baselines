# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import logging
import os
import time
from types import SimpleNamespace
from pathlib import Path

import torch
import pandas as pd

from nmmo.render.replay_helper import FileReplayHelper

import pufferlib
from pufferlib.vectorization import Serial, Multiprocessing
from pufferlib.policy_store import DirectoryPolicyStore
from pufferlib.frameworks import cleanrl
import pufferlib.policy_ranker
import pufferlib.utils
import clean_pufferl

import environment

from reinforcement_learning import config

def setup_policy_store(policy_store_dir):
    # CHECK ME: can be custom models with different architectures loaded here?
    if not os.path.exists(policy_store_dir):
        raise ValueError("Policy store directory does not exist")
    if os.path.exists(os.path.join(policy_store_dir, "trainer.pt")):
        raise ValueError("Policy store directory should not contain trainer.pt")
    logging.info("Using policy store from %s", policy_store_dir)
    policy_store = DirectoryPolicyStore(policy_store_dir)
    return policy_store

def save_replays(policy_store_dir, save_dir):
    # load the checkpoints into the policy store
    policy_store = setup_policy_store(policy_store_dir)
    num_policies = len(policy_store._all_policies())

    # setup the replay path
    save_dir = os.path.join(save_dir, policy_store_dir)
    os.makedirs(save_dir, exist_ok=True)
    logging.info("Replays will be saved to %s", save_dir)

    # Use 1 env and 1 buffer for replay generation
    # TODO: task-condition agents when generating replays
    args = SimpleNamespace(**config.Config.asdict())
    args.num_envs = 1
    args.num_buffers = 1
    args.use_serial_vecenv = True
    args.learner_weight = 0  # evaluate mode
    args.selfplay_num_policies = num_policies + 1
    args.early_stop_agent_num = 0  # run the full episode

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
        seed=args.seed,
        env_creator=environment.make_env_creator(args),
        env_creator_kwargs={},
        agent_creator=make_policy,
        vectorization=Serial,
        num_envs=args.num_envs,
        num_cores=args.num_envs,
        num_buffers=args.num_buffers,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.selfplay_num_policies,
        policy_store=policy_store,
        data_dir=save_dir,
    )

    # Load the policies into the policy pool
    evaluator.policy_pool.update_policies({
        p.name: p.policy(make_policy, evaluator.buffers[0], evaluator.device)
        for p in policy_store._all_policies().values()
    })

    # Set up the replay helper
    o, r, d, i = evaluator.buffers[0].recv()  # reset the env
    replay_helper = FileReplayHelper()
    nmmo_env = evaluator.buffers[0].envs[0].envs[0].env
    nmmo_env.realm.record_replay(replay_helper)
    replay_helper.reset()

    # Run an episode to generate the replay
    while True:
        with torch.no_grad():
            actions, logprob, value, _ = evaluator.policy_pool.forwards(
                torch.Tensor(o).to(evaluator.device),
                None,  # dummy lstm state
                torch.Tensor(d).to(evaluator.device),
            )
            value = value.flatten()
        evaluator.buffers[0].send(actions.cpu().numpy(), None)
        o, r, d, i = evaluator.buffers[0].recv()

        num_alive = len(nmmo_env.realm.players)
        print('Tick:', nmmo_env.realm.tick, ", alive agents:", num_alive)
        if num_alive == 0 or nmmo_env.realm.tick == args.max_episode_length:
            break

    # Save the replay file
    replay_file = os.path.join(save_dir, f"replay_{time.strftime('%Y%m%d_%H%M%S')}")
    logging.info("Saving replay to %s", replay_file)
    replay_helper.save(replay_file, compress=False)
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

def rank_policies(policy_store_dir, device):
    # CHECK ME: can be custom models with different architectures loaded here?
    policy_store = setup_policy_store(policy_store_dir)
    num_policies = len(policy_store._all_policies())
    policy_ranker = create_policy_ranker(policy_store_dir)

    # TODO: task-condition agents when generating replays
    args = SimpleNamespace(**config.Config.asdict())
    args.data_dir = policy_store_dir
    args.learner_weight = 0  # evaluate mode
    args.selfplay_num_policies = num_policies + 1
    args.early_stop_agent_num = 0  # run the full episode

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
        device=torch.device(device),
        seed=args.seed,
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
    """Usage: python evaluate.py -p <policy_store_dir> -s <replay_save_dir>

    -p, --policy-store-dir: Directory to load policy checkpoints from for evaluation/ranking
    -s, --replay-save-dir: Directory to save replays (Default: replays/)
    -e, --eval-mode: Evaluate mode (Default: False)
    -d, --device: Device to use for evaluation/ranking (Default: cuda if available, otherwise cpu)

    To generate replay from your checkpoints, put them together in policy_store_dir, run the following command, 
    and replays will be saved under the replays/. The script will only use 1 environment.
    $ python evaluate.py -p <policy_store_dir>

    To rank your checkpoints, set the eval-mode to true, and the rankings will be printed out.
    The replay files will NOT be generated in the eval mode.:
    $ python evaluate.py -p <policy_store_dir> -e true

    TODO: Pass in the task embedding?
    """
    logging.basicConfig(level=logging.INFO)

    # Process the evaluate.py command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--policy-store-dir",
        dest="policy_store_dir",
        type=str,
        default=None,
        help="Directory to load policy checkpoints from",
    )
    parser.add_argument(
        "-s",
        "--replay-save-dir",
        dest="replay_save_dir",
        type=str,
        default="replays",
        help="Directory to save replays (Default: replays/)",
    )
    parser.add_argument(
        "-e",
        "--eval-mode",
        dest="eval_mode",
        type=bool,
        default=False,
        help="Evaluate mode (Default: False)",
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation/ranking (Default: cuda if available, otherwise cpu)",
    )

    # Parse and check the arguments
    eval_args = parser.parse_args()
    assert eval_args.policy_store_dir is not None, "Policy store directory must be specified"

    if eval_args.eval_mode:
        logging.info("Ranking checkpoints from %s", eval_args.policy_store_dir)
        logging.info("Replays will NOT be generated")
        rank_policies(eval_args.policy_store_dir, eval_args.device)
    else:
        logging.info("Generating replays from the checkpoints in %s", eval_args.policy_store_dir)
        save_replays(eval_args.policy_store_dir, eval_args.replay_save_dir)
