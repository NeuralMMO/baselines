import argparse
import logging
import os
import re
import sys

import nmmo
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.registry.nmmo
import torch
from env.nmmo_config import NmmoConfig
from env.nmmo_env import RewardsConfig
from lib.policy_pool.json_policy_pool import JsonPolicyPool

from model.realikun.baseline_agent import BaselineAgent
from lib.policy_pool.policy_pool import PolicyPool
from lib.policy_pool.opponent_pool_env import OpponentPoolEnv

import cleanrl_ppo_lstm as cleanrl_ppo_lstm
from model.realikun.policy import BaselinePolicy
from env.nmmo_team_env import NMMOTeamEnv
from lib.team.team_helper import TeamHelper

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--model.init_from_path",
    dest="model_init_from_path", type=str, default=None,
    help="path to model to load (default: None)")

  parser.add_argument(
    "--env.num_teams", dest="num_teams", type=int, default=16,
    help="number of teams to use for training (default: 16)")
  parser.add_argument(
    "--env.team_size", dest="team_size", type=int, default=8,
    help="number of agents per team to use for training (default: 8)")
  parser.add_argument(
    "--env.num_npcs", dest="num_npcs", type=int, default=0,
    help="number of NPCs to use for training (default: 256)")
  parser.add_argument(
    "--env.num_learners", dest="num_learners", type=int, default=16,
    help="number of agents running he learner policy (default: 16)")
  parser.add_argument(
    "--env.max_episode_length", dest="max_episode_length", type=int, default=1024,
    help="number of steps per episode (default: 1024)")
  parser.add_argument(
    "--env.death_fog_tick", dest="death_fog_tick", type=int, default=None,
    help="number of ticks before death fog starts (default: None)")
  parser.add_argument(
    "--env.moves_only", dest="moves_only",
    action="store_true", default=False,
    help="only allow moves (default: False)")


  parser.add_argument(
    "--reward.hunger", dest="rewards_hunger",
    action="store_true", default=False,
    help="enable hunger rewards (default: False)")
  parser.add_argument(
    "--reward.thirst", dest="rewards_thirst",
    action="store_true", default=False,
    help="enable thirst rewards (default: False)")
  parser.add_argument(
    "--reward.health", dest="rewards_health",
    action="store_true", default=False,
    help="enable health rewards (default: False)")
  parser.add_argument(
    "--reward.achievements", dest="rewards_achievements",
    action="store_true", default=False,
    help="enable achievement rewards (default: False)")

  parser.add_argument(
    "--reward.disable_symlog", dest="symlog_rewards",
    action="store_false", default=True,
    help="disable symlog rewards (default: True)")

  parser.add_argument(
    "--rollout.num_cores", dest="num_cores", type=int, default=None,
      help="number of cores to use for training (default: num_envs)")
  parser.add_argument(
    "--rollout.num_envs", dest="num_envs", type=int, default=4,
    help="number of environments to use for training (default: 1)")
  parser.add_argument(
    "--rollout.num_buffers", dest="num_buffers", type=int, default=4,
    help="number of buffers to use for training (default: 4)")
  parser.add_argument(
    "--rollout.num_steps", dest="num_steps", type=int, default=128,
    help="number of steps to rollout (default: 16)")

  parser.add_argument(
    "--train.num_steps",
    dest="train_num_steps", type=int, default=10_000_000,
    help="number of steps to train (default: 10_000_000)")
  parser.add_argument(
    "--train.num_maps", dest="num_maps", type=int, default=128,
    help="number of maps to use for training (default: 1)")
  parser.add_argument(
    "--train.checkpoint_interval",
    dest="checkpoint_interval", type=int, default=10,
    help="interval to save models (default: 10)")
  parser.add_argument(
    "--train.experiment_name",
    dest="experiment_name", type=str, default=None,
    help="experiment name (default: None)")
  parser.add_argument(
    "--train.experiments_dir",
    dest="experiments_dir", type=str, default="experiments",
    help="experiments directory (default: experiments)")
  parser.add_argument(
    "--train.use_serial_vecenv",
    dest="use_serial_vecenv", action="store_true",
    help="use serial vecenv impl (default: False)")
  parser.add_argument(
    "--train.opponent_pool", dest="opponent_pool", type=str, default=None,
    help="json file containing the opponent pool (default: None)")

  parser.add_argument(
    "--wandb.project", dest="wandb_project", type=str, default=None,
      help="wandb project name (default: None)")
  parser.add_argument(
    "--wandb.entity", dest="wandb_entity", type=str, default=None,
      help="wandb entity name (default: None)")

  parser.add_argument(
    "--ppo.bptt_horizon", dest="bptt_horizon", type=int, default=8,
    help="train on bptt_horizon steps of a rollout at a time. "
     "use this to reduce GPU memory (default: 16)")

  parser.add_argument(
    "--ppo.num_minibatches",
    dest="ppo_num_minibatches", type=int, default=16,
    help="number of minibatches to use for training")
  parser.add_argument(
    "--ppo.update_epochs",
    dest="ppo_update_epochs", type=int, default=4,
    help="number of update epochs to use for training (default: 4)")
  parser.add_argument(
    "--ppo.learning_rate", dest="ppo_learning_rate",
    type=float, default=0.0001,
    help="learning rate (default: 0.0001)")

  args = parser.parse_args()

  config = NmmoConfig(
    num_teams=args.num_teams,
    team_size=args.team_size,
    num_npcs=args.num_npcs,
    num_maps=args.num_maps,
    max_episode_length=args.max_episode_length,
    death_fog_tick=args.death_fog_tick
  )

  # Historic self play is not yet working, so we require
  # all the players to be learners
  assert args.num_teams == args.num_learners

  # Set up the teams
  team_helper = TeamHelper({
    i: [i*args.team_size+j+1 for j in range(args.team_size)]
    for i in range(args.num_teams)}
  )

  # Create a pool of opponents
  if args.opponent_pool is None:
    opponent_pool = PolicyPool()
  else:
    opponent_pool = JsonPolicyPool(args.opponent_pool)

  binding = None

  # Create an environment factory that uses the opponent pool
  # for some of the agents, while letting the rest be learners
  def make_agent(model_weights):
    if binding is None:
      return None
    return BaselineAgent(model_weights, binding)

  def make_env():
    return OpponentPoolEnv(
      NMMOTeamEnv(config, team_helper,
                  RewardsConfig(
                    symlog_rewards=args.symlog_rewards,
                    hunger=args.rewards_hunger,
                    thirst=args.rewards_thirst,
                    health=args.rewards_health,
                    achievements=args.rewards_achievements
                  ),
                  moves_only=args.moves_only),
      range(args.num_learners, team_helper.num_teams),
      opponent_pool,
      make_agent
    )

  # Create a pufferlib binding, and use it to initialize the
  # opponent pool and create the learner agent
  binding = pufferlib.emulation.Binding(
    env_creator=make_env,
    env_name="Neural MMO",
    suppress_env_prints=False,
    emulate_const_horizon=args.max_episode_length,
  )
  opponent_pool.binding = binding
  learner_agent = BaselinePolicy.create_policy()(binding)

  # Initialize the learner agent from a pretrained model
  if args.model_init_from_path is not None:
    logging.info(f"Initializing model from {args.model_init_from_path}...")
    cleanrl_ppo_lstm.load_matching_state_dict(
      learner_agent,
      torch.load(args.model_init_from_path)["agent_state_dict"]
    )

  # Create an experiment directory for saving model checkpoints
  os.makedirs(args.experiments_dir, exist_ok=True)
  if args.experiment_name is None:
    prefix = f"{args.num_teams}x{args.team_size}_"
    existing = os.listdir(args.experiments_dir)
    prefix_pattern = re.compile(f'^{prefix}(\\d{{4}})$')
    existing_numbers = [int(match.group(1)) for name in existing for match in [prefix_pattern.match(name)] if match]
    next_number = max(existing_numbers, default=0) + 1
    args.experiment_name = f"{prefix}{next_number:04}"

  experiment_dir = os.path.join(args.experiments_dir, args.experiment_name)

  os.makedirs(experiment_dir, exist_ok=True)
  logging.info(f"Experiment directory {experiment_dir}")
  resume_from_path = None
  checkpoins = os.listdir(experiment_dir)
  if len(checkpoins) > 0:
    resume_from_path = os.path.join(experiment_dir, max(checkpoins))

  def epoch_end_callback(state):
    if experiment_dir is not None and state["update"] % args.checkpoint_interval == 0:
        save_path = os.path.join(experiment_dir, f'{state["update"]:06d}.pt')
        temp_path = os.path.join(experiment_dir, f'.{state["update"]:06d}.pt.tmp')
        logging.info(f'Saving checkpoint to {save_path}')
        torch.save(state, temp_path)
        os.rename(temp_path, save_path)
        logging.info(f"Adding {save_path} to policy pool. reward={state['mean_reward']}")
        opponent_pool.add_policy(save_path, state["mean_reward"])

  logging.info("Starting training...")
  try:
    cleanrl_ppo_lstm.train(
      binding,
      learner_agent,
      run_name = args.experiment_name,

      cuda=torch.cuda.is_available(),
      total_timesteps=args.train_num_steps,
      track=(args.wandb_project is not None),

      num_envs=args.num_envs,
      num_cores=args.num_cores or args.num_envs,
      num_buffers=args.num_buffers,
      use_serial_vecenv=args.use_serial_vecenv,

      num_minibatches=args.ppo_num_minibatches,
      update_epochs=args.ppo_update_epochs,

      num_agents=args.num_learners,
      num_steps=args.num_steps,
      bptt_horizon=args.bptt_horizon,

      wandb_project_name=args.wandb_project,
      wandb_entity=args.wandb_entity,

      epoch_end_callback=epoch_end_callback,
      resume_from_path=resume_from_path,

      # PPO
      learning_rate=args.ppo_learning_rate,
      # clip_coef=0.2, # ratio_clip
      # dual_clip_c=3.,
      # ent_coef=0.001 # entropy_loss_weight,
      # grad_clip=1.0,
      # bptt_trunc_len=16,
    )

  except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        logging.info("Exitting due to CUDA out of memory")
        sys.exit(101)
    else:
      raise e

# lr: 0.0001 -> 0.00001
# ratio_clip: 0.2
# dual_clip_c: 3.
# pi_loss_weight: 1.0
# v_loss_weight: 0.5
# entropy_loss_weight: 0.03 -> 0.001
# grad_clip: 1.0
# bptt_trunc_len: 16
