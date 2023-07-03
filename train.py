import argparse
import imp
import logging
import os
import re

import nmmo
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.registry.nmmo
import torch

import clean_pufferl
from env.nmmo_config import nmmo_config
from env.nmmo_env import RewardsConfig
from env.postprocessor import Postprocessor
from lib.agent.baseline_agent import BaselineAgent
from lib.team.team_helper import TeamHelper
import lib.agent.util

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--model.init_from_path",
    dest="model_init_from_path", type=str, default=None,
    help="path to model to load (default: None)")
  parser.add_argument(
    "--model.type",
    dest="model_type", type=str, default="realikun",
    help="model type (default: realikun)")

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
    "--env.max_episode_length", dest="max_episode_length", type=int, default=1024,
    help="number of steps per episode (default: 1024)")
  parser.add_argument(
    "--env.death_fog_tick", dest="death_fog_tick", type=int, default=None,
    help="number of ticks before death fog starts (default: None)")
  parser.add_argument(
    "--env.combat_enabled", dest="combat_enabled",
    action="store_true", default=False,
    help="only allow moves (default: False)")
  parser.add_argument(
    "--env.reset_on_death", dest="reset_on_death",
    action="store_true", default=False,
    help="reset on death (default: False)")
  parser.add_argument(
    "--env.num_maps", dest="num_maps", type=int, default=128,
    help="number of maps to use for training (default: 1)")
  parser.add_argument(
    "--env.maps_path", dest="maps_path", type=str, default="maps/train/",
    help="path to maps to use for training (default: None)")
  parser.add_argument(
    "--env.map_size", dest="map_size", type=int, default=128,
    help="size of maps to use for training (default: 128)")

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
    "--reward.environment", dest="rewards_environment",
    action="store_true", default=False,
    help="enable environment rewards (default: False)")

  parser.add_argument(
    "--reward.symlog", dest="symlog_rewards",
    action="store_true", default=False,
    help="symlog rewards (default: True)")

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
    "--rollout.batch_size", dest="rollout_batch_size", type=int, default=2**14,
    help="number of steps to rollout (default: 2**14)")

  parser.add_argument(
    "--train.num_steps",
    dest="train_num_steps", type=int, default=10_000_000,
    help="number of steps to train (default: 10_000_000)")
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
    "--ppo.training_batch_size",
    dest="ppo_training_batch_size", type=int, default=32,
    help="number of rows in a training batch (default: 32)")
  parser.add_argument(
    "--ppo.update_epochs",
    dest="ppo_update_epochs", type=int, default=4,
    help="number of update epochs to use for training (default: 4)")
  parser.add_argument(
    "--ppo.learning_rate", dest="ppo_learning_rate",
    type=float, default=0.0001,
    help="learning rate (default: 0.0001)")

  args = parser.parse_args()

  # Set up the teams
  team_helper = TeamHelper({
    i: [i*args.team_size+j+1 for j in range(args.team_size)]
    for i in range(args.num_teams)}
  )

  config = nmmo_config(
    team_helper,
    dict(
      # num_npcs=args.num_npcs,
      num_maps=args.num_maps,
      maps_path=f"{args.maps_path}/{args.map_size}/",
      map_size=args.map_size,
      max_episode_length=args.max_episode_length,
      death_fog_tick=args.death_fog_tick,
      combat_enabled=args.combat_enabled,
      num_npcs=args.num_npcs,
    )
  )
  config.RESET_ON_DEATH = args.reset_on_death

  binding = None

  rewards_config = RewardsConfig(
    symlog_rewards=args.symlog_rewards,
    hunger=args.rewards_hunger,
    thirst=args.rewards_thirst,
    health=args.rewards_health,
    achievements=args.rewards_achievements,
    environment=args.rewards_environment
  )

  def make_env():
    return nmmo.Env(config)
    # if args.model_type in ["realikun", "realikun-simplified"]:
    #   env = NMMOTeamEnv(
    #     config, team_helper, rewards_config, moves_only=args.moves_only)

  binding = pufferlib.emulation.Binding(
    env_creator=make_env,
    env_name="Neural MMO",
    suppress_env_prints=False,
    emulate_const_horizon=args.max_episode_length,
    # teams=team_helper.teams,
    postprocessor_cls=Postprocessor,
    postprocessor_args=[rewards_config]
  )

  # Initialize the learner agent from a pretrained model
  learner_policy = None
  if args.model_init_from_path is not None:
    logging.info(f"Initializing model from {args.model_init_from_path}...")
    model = torch.load(args.model_init_from_path)
    learner_policy = BaselineAgent.policy_class(
      model.get("model_type", "realikun"))(binding)
    lib.agent.util.load_matching_state_dict(
      learner_policy,
      model["agent_state_dict"]
    )
  else:
    learner_policy = BaselineAgent.policy_class(args.model_type)(binding)

  device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
  learner_policy = learner_policy.to(device)
  opponent_pool = pufferlib.policy_pool.PolicyPool(
    args.num_teams * args.team_size * args.num_envs,
    policies=[learner_policy],
    names=['baseline'],
    tenured=[True],
    sample_weights=[1, 1],
    max_policies=8,
    path='pool'
  )
  opponent_pool.add_policy_copy('baseline', 'anchor', anchor=True)


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

  vec_env_cls = pufferlib.vectorization.multiprocessing.VecEnv
  if args.use_serial_vecenv:
    vec_env_cls = pufferlib.vectorization.serial.VecEnv

  logging.info("Starting training...")
  trainer = clean_pufferl.CleanPuffeRL(
    binding,
    learner_policy,

    run_name = args.experiment_name,

    vec_backend=vec_env_cls,
    total_timesteps=args.train_num_steps,

    num_envs=args.num_envs,
    num_cores=args.num_cores or args.num_envs,
    num_buffers=args.num_buffers,

    batch_size=args.rollout_batch_size,

    policy_pool=opponent_pool,

    # PPO
    learning_rate=args.ppo_learning_rate,
    # clip_coef=0.2, # ratio_clip
    # dual_clip_c=3.,
    # ent_coef=0.001 # entropy_loss_weight,
    # grad_clip=1.0,
    # bptt_trunc_len=16,
  )

  resume_from_path = None
  checkpoins = os.listdir(experiment_dir)
  if len(checkpoins) > 0:
    resume_from_path = os.path.join(experiment_dir, max(checkpoins))
    trainer.resume_model(resume_from_path)

  trainer_state = trainer.allocate_storage()
  if args.wandb_project is not None:
    trainer.init_wandb(args.wandb_project, args.wandb_entity, extra_data=vars(args))

  num_updates = 1000000
  for update in range(trainer.update+1, num_updates + 1):
    trainer.evaluate(learner_policy, trainer_state)
    trainer.train(
      learner_policy,
      trainer_state,
      update_epochs=args.ppo_update_epochs,
      bptt_horizon=args.bptt_horizon,
      batch_rows=args.ppo_training_batch_size
    )
    if experiment_dir is not None and update % args.checkpoint_interval == 1:
      save_path = os.path.join(experiment_dir, f'{update:06d}.pt')
      trainer.save_model(save_path,
                         model_type=args.model_type)
      opponent_pool.add_policy_copy('baseline', f'baseline{update}')


  trainer.close()

# lr: 0.0001 -> 0.00001
# ratio_clip: 0.2
# dual_clip_c: 3.
# pi_loss_weight: 1.0
# v_loss_weight: 0.5
# entropy_loss_weight: 0.03 -> 0.001
# grad_clip: 1.0
# bptt_trunc_len: 16
