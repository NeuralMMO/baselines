from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
from torch.distributions.categorical import Categorical

import nmmo

from scripted import baselines
from neural import overlays

class Policy:
    def __init__(self, config, torch_model, device='cuda:0'):
        self.model  = torch_model
        self.config = config
        self.device = device

        batch = config.NENT

        # Set initial state for recurrent models
        self.state = None
        if hasattr(self.model, 'get_initial_state'):
            self.state = self.model.get_initial_state(batch)

    def sample_logits(self, logits):
        return Categorical(logits=logits).sample()

    def compute_action(self, obs):
        config = self.config

        obs_keys = obs.keys()
        obs = np.stack(obs.values())
        obs = torch.tensor(obs).float()
        obs = nmmo.emulation.unpack_obs(self.config, obs)

        if self.state:
            logits, self.state = self.model(obs, self.state)
        else:
            logits = self.model(obs)
    
        if self.config.EMULATE_FLAT_ATN:
            return self.sample_logits(logits)

        # Big mess for unpacking observations
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
        config.EMULATE_CONST_NENT = True

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

        # Init env with extra overlays
        env          = nmmo.Env(config)
        env.registry = overlays.NeuralOverlayRegistry(env).init(self.policy)

        t   = 0  
        obs = env.reset()
        while True:
            with torch.no_grad():
                actions = self.policy.compute_action(obs)

            if config.RENDER:
                env.render()
            elif t == config.HORIZON:
                break

            obs, rewards, dones, infos = env.step(actions)
            t += 1

        return env.terminal()['Stats']
        
