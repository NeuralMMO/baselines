import torch
import cleanrl_ppo_lstm
import pufferlib.emulation
import pufferlib.registry.nmmo
import pufferlib.frameworks.cleanrl
import nmmo
from model.policy import Policy
from model.model import MemoryBlock
from feature_extractor.feature_extractor import FeatureExtractor
import argparse
import os
import subprocess

def get_gpu_memory():
  result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, text=True)
  return [int(x) for x in result.stdout.strip().split('\n')]

def get_least_utilized_gpu():
  gpu_memory = get_gpu_memory()
  return gpu_memory.index(min(gpu_memory))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu_id", type=int, default=None,
      help="GPU ID to use for training, or -1 to pick the least utilized (default: None)")
  parser.add_argument("--num_cores", type=int, default=1,
      help="number of cores to use for training (default: 1)")
  parser.add_argument("--num_steps", type=int, default=16,
      help="number of steps to use for training (default: 16)")

  parser.add_argument("--num_envs", type=int, default=1,
      help="number of environments to use for training (default: 1)")
  parser.add_argument("--num_buffers", type=int, default=4,
      help="number of buffers to use for training (default: 4)")
  parser.add_argument("--num_minibatches", type=int, default=4,
      help="number of minibatches to use for training (default: 4)")
  parser.add_argument("--update_epochs", type=int, default=4,
      help="number of update epochs to use for training (default: 4)")

  parser.add_argument("--num_agents", type=int, default=16,
      help="number of agents to use for training (default: 16)")

  parser.add_argument("--wandb_project", type=str, default=None,
      help="wandb project name (default: None)")
  parser.add_argument("--wandb_entity", type=str, default=None,
      help="wandb entity name (default: None)")

  parser.add_argument("--model_path", type=str, default=None,
      help="path to model to load (default: None)")
  parser.add_argument("--checkpoint_dir", type=str, default=None,
      help="path to save models (default: None)")
  parser.add_argument("--checkpoint_interval", type=int, default=10,
                      help="interval to save models (default: 10)")
  parser.add_argument("--resume_from", type=str, default=None,
      help="path to resume from (default: None)")

  args = parser.parse_args()

  if torch.cuda.is_available():
    if args.gpu_id == -1:
      args.gpu_id = get_least_utilized_gpu()
      print(f"Selected GPU with least memory utilization: {args.gpu_id}")

  class TrainConfig(
    nmmo.config.Medium,
    nmmo.config.Terrain,
    nmmo.config.Resource,
    nmmo.config.Combat):

    PROVIDE_ACTION_TARGETS = True
    MAP_N = args.num_cores*4
    MAP_FORCE_GENERATION = False

  config = TrainConfig()

  def make_env():
    return nmmo.Env(config)

  binding = pufferlib.emulation.Binding(
    env_creator=make_env,
    env_name="Neural MMO",
    teams = {i: [i*8+j+1 for j in range(8)] for i in range(16)},
    featurizer_cls=FeatureExtractor,
    featurizer_args=[config],
    suppress_env_prints=False,
  )

  agent = pufferlib.frameworks.cleanrl.make_policy(
      Policy,
      recurrent_cls=MemoryBlock,
      recurrent_args=[2048, 4096],
      recurrent_kwargs={'num_layers': 1},
      )(
    binding
  )

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
      total_timesteps=10_000_000,
      track=(args.wandb_project is not None),

      num_envs=args.num_envs,
      num_cores=args.num_cores,
      num_buffers=args.num_buffers,

      num_minibatches=args.num_minibatches,
      update_epochs=args.update_epochs,

      num_agents=16,
      num_steps=args.num_steps,
      wandb_project_name=args.wandb_project,
      wandb_entity=args.wandb_entity,

      checkpoint_dir=args.checkpoint_dir,
      checkpoint_interval=args.checkpoint_interval,
      resume_from_path=args.resume_from,

      # PPO
      learning_rate=0.00001,
      clip_coef=0.2, # ratio_clip
      # dual_clip_c=3.,
      ent_coef=0.001 # entropy_loss_weight,
      # grad_clip=1.0,
      # bptt_trunc_len=16,
    )

  if torch.cuda.is_available():
    with torch.cuda.device(args.gpu_id):
      train()
  else:
    train()

# lr: 0.0001 -> 0.00001
# ratio_clip: 0.2
# dual_clip_c: 3.
# pi_loss_weight: 1.0
# v_loss_weight: 0.5
# entropy_loss_weight: 0.03 -> 0.001
# grad_clip: 1.0
# bptt_trunc_len: 16
