import numpy as np
import torch
from baseline_env import BaselineEnv
import cleanrl_ppo_lstm
import pufferlib.emulation
import pufferlib.registry.nmmo
import pufferlib.frameworks.cleanrl
import gym
import nmmo
from team_env import TeamEnv
from model.policy import Policy
from model.model import MemoryBlock
from feature_extractor.feature_extractor import FeatureExtractor
from team_helper import TeamHelper


if __name__ == "__main__":
  num_cores = 1

  config = nmmo.Env().config

  binding = pufferlib.emulation.Binding(
    env_cls=nmmo.Env,
    env_name="Neural MMO",
    teams = {i+1: [i*8+j+1 for j in range(8)] for i in range(16)},
    featurizer_cls=FeatureExtractor,
    featurizer_args=[config],
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
  cleanrl_ppo_lstm.train(
    binding,
    agent,
    cuda=torch.cuda.is_available(),
    total_timesteps=10_000_000,
    track=False,
    num_envs=num_cores,
    num_cores=num_cores,
    num_buffers=4,
    num_minibatches=4,
    num_agents=16,
    wandb_project_name="pufferlib",
    wandb_entity="platypus",
  )
