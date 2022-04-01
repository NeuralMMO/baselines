from pdb import set_trace as T
from torch.distributions.categorical import Categorical

import nmmo

from scripted import baselines

class Agent:
    def __init__(self, config, policy, scripted=False, flat_obs=True, flat_atn=True):
        self.policy   = policy
        self.config   = config

        self.scripted = scripted
        self.flat_obs = flat_obs
        self.flat_atn = flat_atn

    def sample_logits(self, logits):
        return Categorical(logits=logits).sample()

    def policy_logits(self, ob):
        return self.policy(ob)

    def policy_value(self, ob):
        return self.policy.value(ob)

    def compute_action(self, ob):
        #if self.policy.scripted or not self.flat_obs:
        #    ob = ob.reshape(1, -1)
        #    ob = nmmo.emulation.unpack_obs(self.config, ob) 

        if self.policy.scripted:
            return self.policy(ob)

        logits = self.policy_logits(ob)
    
        if self.flat_atn:
            return self.sample_logits(logits)

        action = {} 
        for atnKey, atn in sorted(output.items()):                              
            action[atnKey] = {}
            for argKey, arg in sorted(atn.items()):
                action[atnKey][argKey] = sample(logits)

        return action

def population_fn(idx):
    return idx // 8

class Evaluator:
    def __init__(self, config, agents):
        self.config = config

        self.ratings = nmmo.OpenSkillRating(agents, baselines.Combat)

    def evaluate(self, agents):
        config  = self.config
        config.AGENTS = agents
        config.HORIZON = 32

        agents = {i: agent for i, agent in enumerate(agents)}

        env     = nmmo.Env(config)
        obs     = env.reset()
        actions = {}

        for _ in range(config.HORIZON):
            for k, ob in obs.items():
                agent_idx  = config.population_mapping_fn(k)
                agent      = agents[agent_idx]
                actions[k] = agent(ob)

            obs, atns, dones, infos = env.step(actions)

        stats = env.terminal()['Stats']
        
         
