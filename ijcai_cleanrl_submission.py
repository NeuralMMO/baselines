'''Wrapper file for CleanRL models + IJCAI 2022 NMMO Challenge

You need to add the following dependencies to the runtime specification
in the starter kit: supersuit, tensorboard

Simply copy baselines/* into ijcai-starter-kit/my-submission and rename this
file to submission.py'''

from pdb import set_trace as T
import torch
import numpy as np

import nmmo

from ijcai2022nmmo import Team

from main import Agent
from config.cleanrl import Train as Config
from evaluate import Policy


class CleanRLTeam(Team):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = Config()
        self.device = 'cpu'

        model = Agent(config)
        state_dict = torch.load('my-submission/model_4xt4_96vcpu_1b.pt', map_location=self.device)
        state_dict = {k.lstrip('module')[1:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        self.policy = Policy(config, model, device=self.device)

        self.state = False

        self.keys  = None
        self.dummy = None
    
    def act(self, observations):
        '''Wrapper for CleanRL action function

        This is actually very simple -- all this logic is just to handle
        wrappers missing from the competition version of the environment'''

        # Get initial state for the number of agents used
        if not self.state:
            n = len(observations)
            self.state = True
            self.policy.state = self.policy.model.get_initial_state(n, self.device)

        # Flatten observations
        observations = nmmo.emulation.pack_obs(observations)
        obs_keys = list(observations.keys())

        # Store keys of agents used
        if self.keys is None:
            self.keys = obs_keys
        else: #And pad with 0 obs if needed
            for key in self.keys:
                if key not in observations:
                    observations[key] = self.dummy

        # Stack obs to tensor
        observations = np.stack(observations.values())

        # Store a dummy ob to use for padding
        if self.dummy is None:
            self.dummy = observations[0]

        # Actually run the policy
        with torch.no_grad():
            observations = torch.Tensor(observations)
            atns = self.policy.compute_action(observations)

        # Undo pad observation keys
        atns = [atns[k] for k in obs_keys]

        # Format actions as a structured dict instead of a flat list
        actions = {}
        for key, atn in zip(obs_keys, atns):
            idx = 0
            ent_action = {}
            for a in nmmo.Action.edges:
                ent_action[a] = {}
                for arg in a.edges:
                    ent_action[a][arg] = atn[idx]
                    idx += 1
            actions[key] = ent_action
                
        return actions

class Submission:
    team_klass = CleanRLTeam
    init_params = {}

if __name__ == '__main__':
    '''Some random tests -- you can ignore these, use the submission tool''' 
    config = Config()
    config.EMULATE_FLAT_OBS = True
    config.EMULATE_CONST_NENT = True

    team = CleanRLTeam(0, config)

    env = nmmo.Env(config)
    obs = env.reset()

    for i in range(10):
        actions = team.act(obs)
        obs, rewards, dones, infos = env.step(actions)
        
