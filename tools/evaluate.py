# pylint: disable=bad-builtin, no-member, protected-access
import argparse
from email.policy import Policy
import logging
import os
import time
from venv import logger
import nmmo

import pandas as pd

from env.nmmo_config import NmmoConfig
from nmmo.render.replay_helper import DummyReplayHelper

from env.nmmo_env import RewardsConfig
import lib
from lib.policy_pool.json_policy_pool import JsonPolicyPool
from lib.policy_pool.policy_pool import PolicyPool
from lib.rollout import Rollout
from lib.team.team_helper import TeamHelper
from lib.team.team_replay_helper import TeamReplayHelper

import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.registry.nmmo
import torch
from env.nmmo_config import NmmoConfig
from env.nmmo_env import NMMOEnv, RewardsConfig
from env.postprocessor import Postprocessor
from lib.policy_pool.json_policy_pool import JsonPolicyPool

from lib.agent.baseline_agent import BaselineAgent
from lib.policy_pool.policy_pool import PolicyPool
from lib.policy_pool.opponent_pool_env import OpponentPoolEnv
from nmmo.render.replay_helper import DummyReplayHelper

import cleanrl_ppo_lstm as cleanrl_ppo_lstm
from env.nmmo_team_env import NMMOTeamEnv
from lib.team.team_env import TeamEnv
from lib.team.team_helper import TeamHelper

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
    "--env.num_maps", dest="num_maps", type=int, default=128,
    help="number of maps to use for evaluation (default: 128)")
  parser.add_argument(
    "--env.maps_path", dest="maps_path", type=str, default="maps/eval/medium",
    help="path to maps to use for evaluation (default: None)")

  parser.add_argument(
    "--eval.num_rounds", dest="num_rounds", type=int, default=1,
    help="number of rounds to use for evaluation (default: 1)")

  parser.add_argument(
    "--eval.num_policies", dest="num_policies", type=int, default=2,
    help="number of policies to use for evaluation (default: 2)")

  parser.add_argument(
    "--replay.save_dir", dest="replay_save_dir", type=str, default=None,
    help="path to save replay files (default: auto-generated)")

  args = parser.parse_args()

  team_helper = TeamHelper({
    i: [i*args.team_size+j+1 for j in range(args.team_size)]
    for i in range(args.num_teams)}
  )

  config = NmmoConfig(
    team_helper,
    num_npcs=args.num_npcs,
    max_episode_length=args.max_episode_length,
    death_fog_tick=args.death_fog_tick,
    num_maps=args.num_maps,
    maps_path=args.maps_path
  )

  replay_helper = DummyReplayHelper()
  if args.replay_save_dir is not None:
    os.makedirs(args.replay_save_dir, exist_ok=True)
    replay_helper = TeamReplayHelper(team_helper)

  rewards_config = RewardsConfig(
    environment=True
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

  puffer_teams = None
  if args.team_size != 1:
    puffer_teams = team_helper.teams

  def make_env():
    env = nmmo.Env(config)
    env.realm.record_replay(replay_helper)
    return env

  binding = pufferlib.emulation.Binding(
    env_creator=make_env,
    env_name="Neural MMO",
    suppress_env_prints=False,
    emulate_const_horizon=args.max_episode_length,
    teams=puffer_teams,
    postprocessor_cls=Postprocessor,
    postprocessor_args=[rewards_config]
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  for ri in range(args.num_rounds):
    models = list(set(
       policy_pool.select_least_tested_policies(args.num_policies*2)[:args.num_policies]))

    assert len(models) == 1
    model_data = torch.load(models[0], map_location=device)
    agent = BaselineAgent.policy_class(
      model_data.get("model_type", "realikun"))(binding)
    lib.agent.util.load_matching_state_dict(
      agent,
      model_data["agent_state_dict"])

    evaluator = cleanrl_ppo_lstm.CleanPuffeRL(
      binding,
      agent,

      cuda=torch.cuda.is_available(),
      vec_backend=pufferlib.vectorization.serial.VecEnv,
      total_timesteps=10000000,

      num_envs=1,
      num_cores=1,
      num_buffers=1,
      num_steps=args.max_episode_length,

      num_agents=args.num_teams,
      seed=args.seed+ri,
    )
    eval_state = evaluator.allocate_storage()

    logger.info(f"Evaluating models: {models} with seed {args.seed+ri}")
    evaluator.evaluate(agent, eval_state, max_episodes=1)

    if args.replay_save_dir is not None:
      replay_helper.save(
        os.path.join(args.replay_save_dir, f"replay_{ri}"), compress=False)

    logger.info(f"Model rewards: {sum(eval_state.rewards)}")

    # old_ranks = policy_pool._skill_rating.stats
    # policy_pool.update_rewards(model_rewards)
    # new_ranks = policy_pool._skill_rating.stats

    # table = pd.DataFrame(models, columns=["Model"])
    # table["Reward"] = [model_rewards[model] for model in table["Model"]]
    # table["Old Rank"] = [old_ranks.get(model, 1000) for model in table["Model"]]
    # table["New Rank"] = [new_ranks.get(model, 1000) for model in table["Model"]]
    # table["Delta"] = [new_ranks[model]-old_ranks.get(model, 1000) for model in table["Model"]]

    # table = table.sort_values(by='Reward')
    # logger.info("\n" + table.to_string(index=False))




