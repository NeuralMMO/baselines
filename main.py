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
  parser.add_argument("--num_envs", type=int, default=1,
      help="number of environments to use for training (default: 1)")
  parser.add_argument("--num_buffers", type=int, default=4,
      help="number of buffers to use for training (default: 4)")
  parser.add_argument("--num_minibatches", type=int, default=4,
      help="number of minibatches to use for training (default: 4)")
  parser.add_argument("--num_agents", type=int, default=16,
      help="number of agents to use for training (default: 16)")
  parser.add_argument("--wandb_project", type=str, default=None,
      help="wandb project name (default: None)")
  parser.add_argument("--wandb_entity", type=str, default=None,
      help="wandb entity name (default: None)")


  args = parser.parse_args()

  if torch.cuda.is_available():
    if args.gpu_id == -1:
      args.gpu_id = get_least_utilized_gpu()
      print(f"Selected GPU with least memory utilization: {args.gpu_id}")

  config = nmmo.config.Medium()

  def make_env():
    return nmmo.Env(config)

  config = nmmo.Env().config
  config.MAP_N = args.num_cores*4
  config.MAP_FORCE_GENERATION = False

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

  assert binding is not None
  train = lambda: cleanrl_ppo_lstm.train(
      binding,
      agent,
      cuda=torch.cuda.is_available(),
      total_timesteps=10_000_000,
      track=(args.wandb_project is not None),
      num_envs=args.num_envs,
      num_cores=args.num_cores,
      num_buffers=4,
      num_minibatches=4,
      num_agents=16,
      wandb_project_name=args.wandb_project,
      wandb_entity=args.wandb_entity,
    )

  if torch.cuda.is_available() and args.gpu_id is not None:
    with torch.cuda.device(args.gpu_id):
      train()
  else:
    train()
