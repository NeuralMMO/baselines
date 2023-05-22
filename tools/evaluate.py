# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import logging
import os

from env.nmmo_config import NmmoConfig
from nmmo.render.replay_helper import DummyReplayHelper

from env.nmmo_env import RewardsConfig
from lib.rollout import Rollout
from lib.team.team_helper import TeamHelper
from lib.team.team_replay_helper import TeamReplayHelper

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--model.checkpoints",
    dest="model_checkpoints", type=str,
    default="model_weights/realikun.001470.pt",
    help="comma seperated list of paths to model checkpoints to load")

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
    "--replay.save_dir", dest="replay_save_dir", type=str, default=None,
    help="path to save replay files (default: auto-generated)")

  args = parser.parse_args()

  config = NmmoConfig(
    num_teams=args.num_teams,
    team_size=args.team_size,
    num_npcs=args.num_npcs,
    max_episode_length=args.max_episode_length,
    death_fog_tick=args.death_fog_tick
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


  rewards_config = RewardsConfig()

  rollout = Rollout(
    config, team_helper, rewards_config,
    args.model_checkpoints.split(","),
    replay_helper
  )

  for ri in range(args.num_rounds):
    print(f"Generating the replay for round {ri+1} with seed {args.seed+ri}")
    agent_rewards, model_rewards = rollout.run_episode(args.seed+ri)
    print("Episode reward", agent_rewards)
    print("Model reward", model_rewards)
    replay_helper.save(
      os.path.join(args.replay_save_dir, f"replay_{ri}"), compress=False)



