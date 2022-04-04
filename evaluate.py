from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch.distributions.categorical import Categorical

import nmmo

from scripted import baselines

class Policy:
    def __init__(self, config, torch_model,
            flat_obs=True, flat_atn=True, device='cuda:0'):
        self.model  = torch_model
        self.config = config
        self.device = device

        self.flat_obs = flat_obs
        self.flat_atn = flat_atn

    def sample_logits(self, logits):
        return Categorical(logits=logits).sample()

    def policy_logits(self, ob):
        return self.model(ob)

    def policy_value(self, ob):
        return self.model.value(ob)

    def compute_action(self, obs):
        config = self.config

        obs_keys = obs.keys()
        obs = np.stack(obs.values())
        obs = torch.tensor(obs).float()

        logits = self.policy_logits(obs)
    
        #if self.flat_atn:
        #    return self.sample_logits(logits)

        action = {} 
        for atnKey, atn in sorted(logits.items()):                              
            action[atnKey] = {}
            for argKey, arg in sorted(atn.items()):
                action[atnKey][argKey] = self.sample_logits(arg)

        unpack_action = {}
        for idx, ob_key in enumerate(obs_keys):
            unpack_action[ob_key] = {}
            for atnKey, atn in sorted(logits.items()):                              
               unpack_action[ob_key][atnKey] = {}
               for argKey, arg in sorted(atn.items()):
                   unpack_action[ob_key][atnKey][argKey] = action[atnKey][argKey][idx]

        return unpack_action

class Evaluator:
    def __init__(self, config, torch_model=None):
        self.config = config

        config.EMULATE_FLAT_OBS   = True

        # Generate maps once at the start
        if config.FORCE_MAP_GENERATION:
            nmmo.MapGenerator(self.config).generate_all_maps()
            config.FORCE_MAP_GENERATION = False

        self.ratings = nmmo.OpenSkillRating(config.AGENTS, baselines.Combat)

        if torch_model:
            self.policy  = Policy(config, torch_model)

    def render(self):
        self.config.RENDER = True
        self.rollout()

    def evaluate(self, rollouts=10):
        for i in range(rollouts):
            stats = self.rollout()

            ratings = self.ratings.update(
                    policy_ids=stats['PolicyID'],
                    scores=stats['Task_Reward']) 

            for policy, rating in ratings.items():
                print(f'{policy.__name__}: {rating.mu}')

    def rollout(self):
        config = self.config
        env    = nmmo.Env(config)
        obs    = env.reset()

        t = 0
        while True:
            if config.RENDER:
                env.render()
            elif t == config.HORIZON:
                break

            t += 1

            # TODO: Should we pad here or not? Makes handling LSTMs easier to pad...
            actions = {}
            if obs:
                actions = self.policy.compute_action(obs)

            obs, atns, dones, infos = env.step(actions)

        return env.terminal()['Stats']
        
