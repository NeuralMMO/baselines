# pylint: disable=bad-builtin, no-member, protected-access
import argparse
from email.policy import Policy
import logging
import os
import time
from venv import logger

import pandas as pd

from env.nmmo_config import NmmoConfig
from nmmo.render.replay_helper import DummyReplayHelper

from env.nmmo_env import RewardsConfig
from lib.policy_pool.json_policy_pool import JsonPolicyPool
from lib.policy_pool.policy_pool import PolicyPool
from lib.rollout import Rollout
from lib.team.team_helper import TeamHelper
from lib.team.team_replay_helper import TeamReplayHelper

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--model.checkpoints",
    dest="model_checkpoints", type=str,
    default=None,
    help="comma seperated list of paths to model checkpoints to load")
  parser.add_argument(
    "--model.policy_pool", dest="policy_pool", type=str, default=None,
    help="path to policy pool to use for evaluation (default: None)")

  parser.add_argument(
    "--env.seed", dest="seed", type=int, default=1,
    help="random seed to initialize the env (default: 1)")
  parser.add_argument(
    "--env.num_teams", dest="num_teams", type=int, default=16,
    help="number of teams to use for replay (default: 16)")
  parser.add_argument(
    "--env.team_size", dest="team_size", type=int, default=8,
    help="number of agents per team to use for replay (default: 8)")
  parser.add_argument(
    "--env.num_npcs", dest="num_npcs", type=int, default=0,
    help="number of NPCs to use for replay (default: 0)")
  parser.add_argument(
    "--env.max_episode_length", dest="max_episode_length", type=int, default=1024,
    help="number of steps per episode (default: 1024)")
  parser.add_argument(
    "--env.death_fog_tick", dest="death_fog_tick", type=int, default=None,
    help="number of ticks before death fog starts (default: None)")

  parser.add_argument(
    "--eval.num_rounds", dest="num_rounds", type=int, default=1,
    help="number of rounds to use for evaluation (default: 1)")
  parser.add_argument(
    "--eval.num_maps", dest="num_maps", type=int, default=128,
    help="number of maps to use for evaluation (default: 128)")
  parser.add_argument(
    "--eval.num_policies", dest="num_policies", type=int, default=2,
    help="number of policies to use for evaluation (default: 2)")

  parser.add_argument(
    "--replay.save_dir", dest="replay_save_dir", type=str, default=None,
    help="path to save replay files (default: auto-generated)")

  args = parser.parse_args()

  config = NmmoConfig(
    num_teams=args.num_teams,
    team_size=args.team_size,
    num_npcs=args.num_npcs,
    max_episode_length=args.max_episode_length,
    death_fog_tick=args.death_fog_tick,
    num_maps=args.num_maps
  )

  config.MAP_PREVIEW_DOWNSCALE = 8
  config.MAP_CENTER = 64

  team_helper = TeamHelper({
    i: [i*args.team_size+j+1 for j in range(args.team_size)]
    for i in range(args.num_teams)}
  )

  replay_helper = DummyReplayHelper()
  if args.replay_save_dir is not None:
    os.makedirs(args.replay_save_dir, exist_ok=True)
    replay_helper = TeamReplayHelper(team_helper)


  rewards_config = RewardsConfig(
    achievements=True
  )

  policy_pool = PolicyPool()
  if args.policy_pool is not None:
      policy_pool = JsonPolicyPool(args.policy_pool)
      policy_pool._load()

  if args.model_checkpoints is not None:
    for p in args.model_checkpoints.split(","):
      policy_pool.add_policy(p)

  while len(policy_pool._policies) < args.num_policies:
    logger.warn("Not enough policies to evaluate, waiting...")
    policy_pool._load()
    time.sleep(60)

  for ri in range(args.num_rounds):
    models = list(set(
       policy_pool.select_least_tested_policies(args.num_policies*2)[:args.num_policies]))

    if len(models) < 2:
       logger.warn("Not enough models to evaluate, skipping round")
       continue

    rollout = Rollout(
      config, team_helper, rewards_config,
      models,
      replay_helper
    )

    logger.info(f"Evaluating models: {models} with seed {args.seed+ri}")
    agent_rewards, model_rewards = rollout.run_episode(args.seed+ri)

    if args.replay_save_dir is not None:
      replay_helper.save(
        os.path.join(args.replay_save_dir, f"replay_{ri}"), compress=False)

    old_ranks = policy_pool._skill_rating.stats
    policy_pool.update_rewards(model_rewards)
    new_ranks = policy_pool._skill_rating.stats

    table = pd.DataFrame(models, columns=["Model"])
    table["Reward"] = [model_rewards[model] for model in table["Model"]]
    table["Old Rank"] = [old_ranks.get(model, 1000) for model in table["Model"]]
    table["New Rank"] = [new_ranks,get[model] for model in table["Model"]]
    table["Delta"] = [new_ranks[model]-old_ranks.get(model, 1000) for model in table["Model"]]

    table = table.sort_values(by='Reward')
    logger.info("\n" + table.to_string(index=False))




