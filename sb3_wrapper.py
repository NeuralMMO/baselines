'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo

import gym
import os

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

'''
from gym.envs.registration import register
# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="CartPole-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="gym.envs.classic_control:CartPoleEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=500,
)
'''

class Env(nmmo.Env, gym.Env):
    def __init__(self, config):
        super().__init__(config)
        self.observation_space = self.observation_space(1)
        self.action_space = self.action_space(1)

    def step(self, actions):
        if type(actions) != dict:
            actions = {1: actions}
        
        obs, rewards, dones, infos = super().step(actions)
        return obs[1], rewards[1], dones[1], infos[1]

class Config(nmmo.config.Small):
    EMULATE_FLAT_OBS   = True
    EMULATE_FLAT_ATN   = True
    EMUALTE_CONST_NENT = True

    NENT               = 1

    #Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'

    #Force terrain generation -- avoids unexpected behavior from caching
    FORCE_MAP_GENERATION = True

if __name__ == '__main__':
    num_cpu = 4

    
    #make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv
    make_env = lambda: Env(Config())

    #env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    env = make_env()
    check_env(make_env())

    with open('wandb_api_key') as key:
        os.environ['WANDB_API_KEY'] = key.read().rstrip('\n')

    run = wandb.init(
        project='nmmo-sb3',
        sync_tensorboard=True)
 
    model = A2C('MlpPolicy', env)
    model.learn(
        total_timesteps=1000,
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
        if done:
          obs = env.reset()
