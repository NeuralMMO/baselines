'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo

from neural import io, subnets, policy

import gym
import os
import supersuit as ss
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th 
from torch import nn

import wandb
from wandb.integration.sb3 import WandbCallback

from pettingzoo.utils.env import ParallelEnv
from supersuit.vector import MarkovVectorEnv

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


##################################################
# Wrapping the NMMO baseline archietecture for SB3
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
        
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
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

################################################
# Configure environment with compatibility mixin
class Config(nmmo.config.CompatibilityMixin, nmmo.config.Small):
    NENT               = 4
    HORIZON            = 32
    HIDDEN             = 32
    EMBED              = 32

    #Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'

    #Force terrain generation -- avoids unexpected behavior from caching
    FORCE_MAP_GENERATION = True


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    
    def _on_rollout_end(self) -> bool:
        logs = self.training_env.terminal()
        for k, v in logs['Stats'].items():
            self.logger.record(k, np.mean(v).item())
        return True


if __name__ == '__main__':
    num_cpu  = 4
    num_envs = 1

    config = Config()

    # Wrap environments for SB3
    env = nmmo.integrations.sb3_vec_envs(Config, num_envs, num_cpu)

    #check_env(env)

    with open('wandb_api_key') as key:
        os.environ['WANDB_API_KEY'] = key.read().rstrip('\n')

    run = wandb.init(
        project='nmmo-sb3',
        sync_tensorboard=True)

    policy_kwargs = {'features_extractor_class': CustomFeatureExtractor}

    model = PPO(CustomActorCriticPolicy, env, tensorboard_log=f'runs/{run.id}', policy_kwargs=policy_kwargs)
    model.learn(
        total_timesteps=81,
        callback=WandbCallback(
            model_save_path=f'models/{run.id}',
            verbose=2
        ),
    )

    run.finish()

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        #env.render()
