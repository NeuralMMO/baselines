from pdb import set_trace as T

import os
from tqdm import tqdm

import nmmo
from nmmo import config

import tasks

from scripted import baselines

class MediumAllSystems(config.Medium, config.AllGameSystems):
    PATH_MAPS = os.path.join(config.Medium.PATH_MAPS, 'evaluation')
    AGENTS    = [baselines.Meander, baselines.Forage, baselines.Combat]
    TASKS     = tasks.All

def test_scripted():
    conf = MediumAllSystems()
    env  = nmmo.Env(conf)
    env.reset()

    for i in tqdm(range(128)):
        env.step({})

    logs = env.terminal()
    
    max_lifetime = max(logs['Stats']['Lifetime'])
    assert max_lifetime > 100, 'Best scripted model should live > 100 steps'

if __name__ == '__main__':
    test_scripted()
