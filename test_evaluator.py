from pdb import set_trace as T

import nmmo

import evaluate
from scripted import baselines

from neural import policy

class NeuralAgent(nmmo.Agent):
    def __init__(self, config, idx):
        super().__init__(config, idx)
        self.policy = policy.Baseline(config)

    def __call__(self, obs):
        return self.policy(obs) 

if __name__ == '__main__':
    config    = nmmo.config.Default()
    config.FORCE_MAP_GENERATION = True
    config.HIDDEN = 16
    config.EMBED  = 16
    
    forage   = baselines.Forage
    combat   = baselines.Combat
    neural   = NeuralAgent
    agents   = [forage, combat, neural]

    evaluator = evaluate.Evaluator(config, agents)
    evaluator.evaluate(agents)
