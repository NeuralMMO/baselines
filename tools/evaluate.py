# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import logging
import os
from typing import Any, Dict
from venv import logger

import clean_pufferl
import nmmo
import pandas as pd
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.policy_pool
import pufferlib.registry.nmmo
from env.nmmo import nmmo
from env.postprocessor import Postprocessor

import model
from lib.team.team_helper import TeamHelper
from lib.team.team_replay_helper import TeamReplayHelper

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--model.checkpoints",
      dest="model_checkpoints",
      type=str,
      default=None,
      help="comma seperated list of paths to model checkpoints to load",
  )
  parser.add_argument(
      "--model.policy_pool",
      dest="policy_pool",
      type=str,
      default=None,
      help="path to policy pool to use for evaluation (default: None)",
  )

  parser.add_argument(
      "--env.seed",
      dest="seed",
      type=int,
      default=1,
      help="random seed to initialize the env (default: 1)",
  )
  parser.add_argument(
      "--env.num_teams",
      dest="num_teams",
      type=int,
      default=128,
      help="number of teams to use for replay (default: 16)",
  )
  parser.add_argument(
      "--env.team_size",
      dest="team_size",
      type=int,
      default=1,
      help="number of agents per team to use for replay (default: 8)",
  )
  parser.add_argument(
      "--env.num_npcs",
      dest="num_npcs",
      type=int,
      default=0,
      help="number of NPCs to use for replay (default: 0)",
  )
  parser.add_argument(
      "--env.max_episode_length",
      dest="max_episode_length",
      type=int,
      default=1024,
      help="number of steps per episode (default: 1024)",
  )
  parser.add_argument(
      "--env.death_fog_tick",
      dest="death_fog_tick",
      type=int,
      default=None,
      help="number of ticks before death fog starts (default: None)",
  )
  parser.add_argument(
      "--env.num_maps",
      dest="num_maps",
      type=int,
      default=128,
      help="number of maps to use for evaluation (default: 128)",
  )
  parser.add_argument(
      "--env.maps_path",
      dest="maps_path",
      type=str,
      default="maps/eval/medium",
      help="path to maps to use for evaluation (default: None)",
  )
  parser.add_argument(
      "--env.map_size",
      dest="map_size",
      type=int,
      default=128,
      help="size of maps to use for training (default: 128)",
  )
  parser.add_argument(
      "--env.combat_enabled",
      dest="combat_enabled",
      action="store_true",
      default=False,
      help="only allow moves (default: False)",
  )

  parser.add_argument(
      "--eval.num_rounds",
      dest="num_rounds",
      type=int,
      default=1,
      help="number of rounds to use for evaluation (default: 1)",
  )
  parser.add_argument(
      "--eval.num_envs",
      dest="num_envs",
      type=int,
      default=1,
      help="number of environments to use for evaluation (default: 1)",
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
      help="number of policies to use for evaluation (default: 2)",
  )

  parser.add_argument(
      "--replay.save_dir",
      dest="replay_save_dir",
      type=str,
      default=None,
      help="path to save replay files (default: auto-generated)",
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

  args = parser.parse_args()

  team_helper = TeamHelper(
      {
          i: [i * args.team_size + j + 1 for j in range(args.team_size)]
          for i in range(args.num_teams)
      }
  )

  config = nmmo(
      team_helper,
      dict(
          num_maps=args.num_maps,
          maps_path=f"{args.maps_path}/{args.map_size}/",
          map_size=args.map_size,
          max_episode_length=args.max_episode_length,
          death_fog_tick=args.death_fog_tick,
          combat_enabled=args.combat_enabled,
          num_npcs=args.num_npcs,
      ),
  )

  puffer_teams = None
  if args.team_size != 1:
    puffer_teams = team_helper.teams

  if args.replay_save_dir is not None:
    os.makedirs(args.replay_save_dir, exist_ok=True)

  class ReplayEnv(nmmo.Env):
    num_replays_saved = 0

    def __init__(self, config):
      super().__init__(config)
      self._replay_helper = None
      if args.replay_save_dir is not None:
        self._replay_helper = TeamReplayHelper(team_helper)
        self.realm.record_replay(self._replay_helper)

    def step(self, actions: Dict[int, Dict[str, Dict[str, Any]]]):
      return super().step(actions)

    def reset(self, **kwargs):
      if self.realm.tick and self._replay_helper is not None:
        ReplayEnv.num_replays_saved += 1
        self._replay_helper.save(
            f"{args.replay_save_dir}/{ReplayEnv.num_replays_saved}",
            compress=False,
        )
      return super().reset()

  def make_env():
    return ReplayEnv(config)

  binding = pufferlib.emulation.Binding(
      env_creator=make_env,
      env_name="Neural MMO",
      suppress_env_prints=False,
      emulate_const_horizon=args.max_episode_length,
      teams=puffer_teams,
      postprocessor_cls=Postprocessor,
      postprocessor_args=[],
  )

  policies = []
  if args.model_checkpoints is not None:
    for policy_path in args.model_checkpoints.split(","):
      logging.info(f"Loading model from {policy_path}...")
      policy = model.load_policy(policy_path, binding)
      policies.append(policy)

  policy_pool = pufferlib.policy_pool.PolicyPool(
      evaluation_batch_size=args.num_teams * args.team_size * args.num_envs,
      sample_weights=[1] * len(policies),
      active_policies=len(policies),
      path="pool",
  )

  vec_env_cls = pufferlib.vectorization.multiprocessing.VecEnv
  if args.use_serial_vecenv:
    vec_env_cls = pufferlib.vectorization.serial.VecEnv

  for ri in range(args.num_rounds):
    evaluator = clean_pufferl.CleanPuffeRL(
        binding,
        policies[0],
        policy_pool=policy_pool,
        vec_backend=vec_env_cls,
        total_timesteps=10000000,
        num_envs=args.num_envs,
        num_cores=args.num_envs,
        num_buffers=1,
        batch_size=args.num_envs
        * args.num_teams
        * args.team_size
        * args.max_episode_length,
        seed=args.seed + ri,
    )

    eval_state = evaluator.allocate_storage()
    if args.wandb_project is not None:
      evaluator._init_wandb(
          args.wandb_project, args.wandb_entity, extra_data=vars(args)
      )

    # logger.info(f"Evaluating models: {models} with seed {args.seed+ri}")
    evaluator.evaluate(policies[0], eval_state, show_progress=True)

    logger.info(f"Model rewards: {sum(eval_state.rewards)}")

    stats = policy_pool.tournament.stats

    table = pd.DataFrame(stats.keys(), columns=["Model"])
    table["New Rank"] = [stats.get(model, 1000) for model in table["Model"]]
    # table["Delta"] = [new_ranks[model]-old_ranks.get(model, 1000) for model in table["Model"]]

    # table = table.sort_values(by='Reward')
    logger.info("\n" + table.to_string(index=False))
