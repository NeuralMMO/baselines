from pdb import set_trace as T
import numpy as np

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

        batch = config.NENT

        # Set initial state for recurrent models
        self.state = None
        if hasattr(self.model, 'get_initial_state'):
            self.state = self.model.get_initial_state(batch, device)

    def sample_logits(self, logits):
        return Categorical(logits=logits).sample()

    def compute_action(self, obs):
        config = self.config
        obs = torch.tensor(obs).float()
        obs = obs.to(self.device)
        #obs = nmmo.emulation.unpack_obs(self.config, obs)

        if self.state:
            logits, _, _, _, self.state = self.model(obs, self.state)
        else:
            logits, _, _, _ = self.model(obs)

        actions = logits.cpu().numpy()
        return actions
    
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
    def __init__(self, config_cls, torch_policy_cls=None, rating_stats=None, num_cpus=8, device='cuda:0', *args):
        self.envs   = nmmo.integrations.cleanrl_vec_envs(config_cls, num_cpus, num_cpus)
        config      = config_cls()
        self.config = config

        config.EMULATE_FLAT_OBS   = True
        config.EMULATE_FLAT_ATN   = True
        config.EMULATE_CONST_NENT = True

        # Generate maps once at the start
        if config.FORCE_MAP_GENERATION:
            nmmo.MapGenerator(self.config).generate_all_maps()
            config.FORCE_MAP_GENERATION = False

        self.ratings = nmmo.OpenSkillRating(config.AGENTS, baselines.Combat)

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
        state_dict = {key.lstrip('module')[1:]: val.to(self.device) for key, val in state_dict.items()}
        self.policy.model.load_state_dict(state_dict)

    def __str__(self):
        return ', '.join(f'{p.__name__}: {int(r.mu)}'
                for p, r in self.ratings.ratings.items())

    @property
    def stats(self):
        return {p.__name__: int(r.mu) for p, r in self.ratings.ratings.items()}

    def render(self):
        self.config.RENDER = True
        self.rollout()

    def evaluate(self):
        config = self.config
        obs    = self.envs.reset()
        for i in range(config.HORIZON):
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

        return self.stats

    def render(self):
        # Init env with extra overlays
        env          = nmmo.Env(self.config)
        env.registry = overlays.NeuralOverlayRegistry(env).init(self.policy)

        obs = env.reset()
        while True:
            with torch.no_grad():
                actions = self.policy.compute_action(obs)

            env.render()
            obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    import cleanrl_lstm_wrapper
    from scripted import baselines
    import time

    class EvalConfig(cleanrl_lstm_wrapper.Config):
        AGENTS = [baselines.Forage, baselines.Combat, nmmo.Agent]

    evaluator = Evaluator(EvalConfig, cleanrl_lstm_wrapper.Agent)
    device = 'cuda:1'
    while True:
        try:
            model = torch.load('model.pt', map_location=device)
        except:
            time.sleep(1)
            continue

        evaluator.load_model(model)
        evaluator.evaluate()

        np.save('ratings.npy', evaluator.stats, allow_pickle=True)

