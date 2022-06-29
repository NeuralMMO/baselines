from pdb import set_trace as T
import numpy as np

import sys
import os

from tqdm import tqdm
from collections import defaultdict

import torch
from torch.distributions.categorical import Categorical

import nmmo

from scripted import baselines
from neural import overlays

class Policy:
    def __init__(self, config, torch_model, device):
        self.model  = torch_model
        self.config = config
        self.device = device

        batch = config.PLAYER_N

        # Set initial state for recurrent models
        self.state = None
        if hasattr(self.model, 'get_initial_state'):
            self.state = self.model.get_initial_state(batch, device)

    def sample_logits(self, logits):
        return Categorical(logits=logits).sample()

    def compute_action(self, obs):
        obs = torch.Tensor(obs).float().to(self.device)

        if self.state:
            done = torch.zeros(len(obs)).to(self.device)
            atns, _, _, _, self.state = self.model(obs, self.state, done)
        else:
            atns, _, _, _ = self.model(obs)

        return atns.cpu().numpy()
    
class Evaluator:
    def __init__(self, config_cls, torch_policy_cls=None, rating_stats=None, num_cpus=8, device='cuda:0', *args):
        self.envs   = nmmo.integrations.cleanrl_vec_envs(config_cls)
        config      = config_cls()
        self.config = config

        # Generate maps once at the start
        if config.MAP_FORCE_GENERATION:
            nmmo.MapGenerator(self.config).generate_all_maps()
            config.MAP_FORCE_GENERATION = False

        self.ratings = nmmo.OpenSkillRating(config.PLAYERS, baselines.Combat)

        # Load ratings
        if rating_stats:
            for (mu, sigma), r in zip(rating_stats, self.ratings.ratings.values()):
                r.sigma = sigma
                r.mu    = mu

        if torch_policy_cls:
            self.device  = device
            torch_policy = torch_policy_cls(config, *args).to(device)
            self.policy  = Policy(config, torch_policy, device)

    def load_model(self, state_dict):
        state_dict = {key.lstrip('module')[1:]: val for key, val in state_dict.items()}
        self.policy.model.load_state_dict(state_dict)

    def __str__(self):
        return ', '.join(f'{p.__name__}: {int(r.mu)}'
                for p, r in self.ratings.ratings.items())

    @property
    def stats(self):
        return {p.__name__: int(r.mu) for p, r in self.ratings.ratings.items()}

    def evaluate(self, print_ratings=True):
        config = self.config
        obs    = self.envs.reset()

        from tqdm import tqdm
        for i in tqdm(range(config.HORIZON)):
            with torch.no_grad():
                actions = self.policy.compute_action(obs)
            obs, _, _, infos = self.envs.step(actions)

            for e in infos:
                if 'logs' not in e:
                    continue

                stats = e['logs']
                ratings = self.ratings.update(
                        policy_ids=stats['PolicyID'],
                        scores=stats['Task_Reward']) 

                if print_ratings:
                    print(self)

        return self.stats

    def render(self):
        env          = nmmo.integrations.CleanRLEnv(self.config)

        # Extra overlays -- have to fix these for CleanRL models
        #env.registry = overlays.NeuralOverlayRegistry(env).init(self.policy)

        obs = env.reset()
        while True:
            with torch.no_grad():
                obs_keys = obs.keys()
                obs = torch.Tensor(list(obs.values()))
                actions = self.policy.compute_action(obs)
                actions = {k: v for k, v in zip(obs_keys, actions)}

            env.render()
            obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    from config.cleanrl import Eval as Config
    from main import Agent

    model  = 'models/model_randnent_642m.pt'
    device = 'cpu'

    # Most GPUs should be able to handle 16 parallel envs
    Config.NUM_CPUS = min(16, os.cpu_count())

    # Training currently sets a lower horizon for mem constraints
    Config.HORIZON = 1024
    Config.MAP_FORCE_GENERATION = True

    #state_dict = torch.load(model, map_location=device)
    evaluator  = Evaluator(Config, Agent, num_cpus=Config.NUM_CPUS, device=device)
    #evaluator.load_model(state_dict)

    # Config.RENDER = True # Uncomment to render -- don't delete the check
    # The param is required by the env to generate packets
    # Open the Unity client separately (this just starts the render server)
    if sys.argv[1] == 'render':
        Config.RENDER=True
        evaluator.render()

    # Runs evaluations forever
    # Less accurate at 100% win rate (slowly pushes SR apart forever)
    # Best used to determine if one policy is worse, better, or significantly better than another
    while True:
        evaluator.evaluate()
        print(evaluator)

        np.save('ratings.npy', evaluator.stats, allow_pickle=True)

