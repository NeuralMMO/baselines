import argparse
import os

import nmmo
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.registry.nmmo
import torch

import cleanrl_ppo_lstm
from model.policy import BaselinePolicy
from model.simple.simple_policy import SimplePolicy
from nmmo_env import NMMOEnv
from nmmo_team_env import NMMOTeamEnv
from team_helper import TeamHelper

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--model.arch",
    dest="model_arch", choices=["realikun", "simple"],
    default="realikun",
    help="model architecture (default: realikun)")

  parser.add_argument(
    "--env.num_teams", dest="num_teams", type=int, default=16,
    help="number of teams to use for training (default: 16)")
  parser.add_argument(
    "--env.team_size", dest="team_size", type=int, default=8,
    help="number of agents per team to use for training (default: 8)")

  parser.add_argument(
    "--rollout.num_cores", dest="num_cores", type=int, default=None,
      help="number of cores to use for training (default: num_envs)")
  parser.add_argument(
    "--rollout.num_envs", dest="num_envs", type=int, default=1,
    help="number of environments to use for training (default: 1)")
  parser.add_argument(
    "--rollout.num_buffers", dest="num_buffers", type=int, default=4,
    help="number of buffers to use for training (default: 4)")
  parser.add_argument(
    "--rollout.num_steps", dest="num_steps", type=int, default=16,
    help="number of steps to rollout (default: 16)")

  parser.add_argument(
    "--train.num_steps",
    dest="train_num_steps", type=int, default=10_000_000,
    help="number of steps to train (default: 10_000_000)")

  parser.add_argument(
    "--wandb.project", dest="wandb_project", type=str, default=None,
      help="wandb project name (default: None)")
  parser.add_argument(
    "--wandb.entity", dest="wandb_entity", type=str, default=None,
      help="wandb entity name (default: None)")

  parser.add_argument("--model_path", type=str, default=None,
      help="path to model to load (default: None)")
  parser.add_argument("--checkpoint_dir", type=str, default=None,
      help="path to save models (default: None)")
  parser.add_argument("--checkpoint_interval", type=int, default=10,
                      help="interval to save models (default: 10)")
  parser.add_argument("--resume_from", type=str, default=None,
      help="path to resume from (default: None)")

  parser.add_argument(
    "--ppo.num_minibatches",
    dest="ppo_num_minibatches", type=int, default=4,
    help="number of minibatches to use for training (default: 4)")
  parser.add_argument(
    "--ppo.update_epochs",
    dest="ppo_update_epochs", type=int, default=4,
    help="number of update epochs to use for training (default: 4)")
  parser.add_argument(
    "--ppo.learning_rate", dest="ppo_learning_rate",
    type=float, default=0.0001,
    help="learning rate (default: 0.0001)")

  args = parser.parse_args()

  class TrainConfig(
    nmmo.config.Medium,
    nmmo.config.Terrain,
    nmmo.config.Resource,
    nmmo.config.Progression,
    # nmmo.config.Profession
    # nmmo.config.Combat
  ):

    PROVIDE_ACTION_TARGETS = True
    # MAP_N = 20
    # MAP_FORCE_GENERATION = False
    PLAYER_N = args.num_teams * args.team_size

  config = TrainConfig()
  team_helper = TeamHelper({
    i: [i*args.team_size+j+1 for j in range(args.team_size)]
    for i in range(args.num_teams)}
  )

  if args.model_arch == "simple":
    assert args.team_size == 1

  def make_env():
    if args.model_arch == "simple":
      return NMMOEnv(config)

    return NMMOTeamEnv(config, team_helper)

  binding = pufferlib.emulation.Binding(
    env_creator=make_env,
    env_name="Neural MMO",
    suppress_env_prints=False,
  )

  if args.model_arch == "simple":
    agent = SimplePolicy.create_policy()(binding)
    num_agents = args.num_teams * args.team_size
  else:
    agent = BaselinePolicy.create_policy()(binding)
    num_agents = args.num_teams

  if args.model_path is not None:
    print(f"Loading model from {args.model_path}...")
    agent.load_state_dict(torch.load(args.model_path)["agent_state_dict"])

  if args.checkpoint_dir is not None:
    os.makedirs(args.checkpoint_dir, exist_ok=True)

  if args.resume_from == "latest":
    checkpoins = os.listdir(args.checkpoint_dir)
    if len(checkpoins) > 0:
      args.resume_from = os.path.join(args.checkpoint_dir, max(checkpoins))
    else :
      args.resume_from = None

  assert binding is not None
  train = lambda: cleanrl_ppo_lstm.train(
      binding,
      agent,
      cuda=torch.cuda.is_available(),
      total_timesteps=args.train_num_steps,
      track=(args.wandb_project is not None),

      num_envs=args.num_envs,
      num_cores=args.num_cores or args.num_envs,
      num_buffers=args.num_buffers,

      num_minibatches=args.ppo_num_minibatches,
      update_epochs=args.ppo_update_epochs,

      num_agents=num_agents,
      num_steps=args.num_steps,
      wandb_project_name=args.wandb_project,
      wandb_entity=args.wandb_entity,

      checkpoint_dir=args.checkpoint_dir,
      checkpoint_interval=args.checkpoint_interval,
      resume_from_path=args.resume_from,

      # PPO
      learning_rate=args.ppo_learning_rate,
      # clip_coef=0.2, # ratio_clip
      # dual_clip_c=3.,
      # ent_coef=0.001 # entropy_loss_weight,
      # grad_clip=1.0,
      # bptt_trunc_len=16,
    )

  train()

# lr: 0.0001 -> 0.00001
# ratio_clip: 0.2
# dual_clip_c: 3.
# pi_loss_weight: 1.0
# v_loss_weight: 0.5
# entropy_loss_weight: 0.03 -> 0.001
# grad_clip: 1.0
# bptt_trunc_len: 16
