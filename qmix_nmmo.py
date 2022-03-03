from pdb import set_trace as T

import nmmo

import functools, gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple

import numpy as np
import unittest

import rllib_wrapper

import ray
from ray.tune import register_env
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class QMixNMMO(nmmo.Env, MultiAgentEnv):
   def __init__(self, config):
      self.config = config['config']
      super().__init__(self.config)

   def observation_space(self, agent: int):
      return Dict(
         {
            'obs': super().observation_space(agent)
            #'obs': gym.spaces.Box(low=0, high=1, shape=(1,1), dtype=np.float32),
         }
      )
   
   def action_space(self, agent):
      return gym.spaces.Discrete(2)

   def step(self, actions):
      actions = {}

      obs, rewards, dones, infos = super().step(actions)

      dones['__all__'] = False
      if self.realm.tick >= 32:
         dones['__all__'] = True

      for key in list(obs.keys()):
         obs[key] = {'obs': self.observation_space(key)['obs'].sample()}

      return obs, rewards, dones, infos

class Config(nmmo.config.Small):
    NENT = 2
    EMBED = 2
    HIDDEN = 2

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

class TestQMix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()

    def test_avail_actions_qmix(self):
        Env    = QMixNMMO
        config = Config()

        env       = Env({'config': config})
        idxs      = range(1, config.NENT+1)

        obs_space = Tuple([env.observation_space(i) for i in idxs])
        act_space = Tuple([env.action_space(i) for i in idxs])

        grouping = lambda e: 'group1'
        ray.tune.registry.register_env("Neural_MMO",
            lambda config: Env(config).with_agent_groups(grouping,
            obs_space=obs_space, act_space=act_space))

        ray.rllib.models.ModelCatalog.register_custom_model(
            'nmmo_policy', rllib_wrapper.RLlibPolicy)

        policies = {}
        for i in range(1):
            params = {
                "agent_id": i,
                "obs_space_dict": obs_space,
                "act_space_dict": act_space}

            policies[f'policy_{i}'] = (None, obs_space, act_space, params) 

        agent = QMixTrainer(
            env="Neural_MMO",
            config={
                'num_envs_per_worker': 1,  # test with vectorization on
                'env_config': {
                    'config': config
                },
                'framework': 'torch',
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': lambda i: 'policy_0',
                },
               'model': {
                  'custom_model': 'nmmo_policy',
                  'custom_model_config': {'config': config},
                  'max_seq_len': 16,
               },
               'simple_optimizer': False,
            })

        agent.train()  # OK if it doesn't trip the action assertion error

        agent.stop()
        ray.shutdown()


if __name__ == "__main__":
    import pytest
    import sys
    #sys.exit(pytest.main(["-v", __file__]))

    TestQMix.setUpClass()
    qmix = TestQMix()
    qmix.test_avail_actions_qmix()
    TestQMix.tearDownClass()

