'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T
import numpy as np
import os

import supersuit as ss
import gym

import torch as th 
from torch import nn
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import nmmo

import tasks
from neural import io, subnets, policy

#################################################
# Wrapping the NMMO baseline architecture for SB3
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, Config.HIDDEN)

        self.extractor = io.Input(Config,
                embeddings=io.MixedEmbedding,
                attributes=subnets.SelfAttention)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = nmmo.emulation.unpack_obs(config, observations)
        return self.extractor(observations)

class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = Config.HIDDEN

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = hidden
        self.latent_dim_vf = hidden

        # Policy network
        self.policy_net = policy.Simple(config)
        
    def forward(self, features):
        hidden, _ = self.policy_net(features)
        return hidden, hidden

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)[0]

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.forward(features)[1]

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork()

##################################################
# Integrate logging with WanDB through tensorboard
class TensorboardToWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            if 'logs' not in info:
                continue
            
            for k, v in info['logs'].items():
                self.logger.record(k, np.mean(v).item())

        return True

#######################
# Configure environment
class Config(nmmo.config.Medium, nmmo.config.AllGameSystems):
    # Cheat network params into env config
    HIDDEN = 32
    EMBED  = 32

    # Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

    # Force terrain generation -- avoids unexpected behavior from caching
    FORCE_MAP_GENERATION = True
    NMAPS = 256

    # Enable tasks
    TASKS = tasks.All


if __name__ == '__main__':
    num_epochs = 1000 # Number of updates of size num_envs * num_steps * NENT
    num_cpu    = 16   # Number of CPU cores to use
    num_envs   = 16   # Number of environments to simulate in parallel
    n_steps    = 32   # Steps to simulate each environment per batch

    config = Config()

    # Wrap environments for SB3
    env = nmmo.integrations.sb3_vec_envs(Config, num_envs, num_cpu)

    # WanDB integration
    with open('wandb_api_key') as key:
        os.environ['WANDB_API_KEY'] = key.read().rstrip('\n')

    run = wandb.init(
        project='nmmo-sb3',
        sync_tensorboard=True)

    # Learn with SB3
    model = PPO(
        CustomActorCriticPolicy,
        env,
        n_steps=n_steps,
        batch_size=128,
        n_epochs=1,
        tensorboard_log=f'runs/{run.id}',
        policy_kwargs={
            'features_extractor_class': CustomFeatureExtractor})

    model.learn(
        total_timesteps=num_epochs * num_envs * n_steps * Config.NENT,
        callback=TensorboardToWandbCallback())

    # End WanDB logging
    run.finish()

    # TODO: Evaluate and render properly
    #obs = env.reset()
    #for i in range(1000):
    #    action, _state = model.predict(obs, deterministic=True)
    #    obs, reward, done, info = env.step(action)
