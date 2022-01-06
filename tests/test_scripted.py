from pdb import set_trace as T

import os

import nmmo
from nmmo import config

import tasks

from scripted import baselines

def test_scripted():
    conf           = config.Small()
    conf.TASKS     = [tasks.All]
    conf.AGENTS    = [baselines.Meander, baselines.Forage, baselines.Combat]
    conf.PATH_MAPS = os.path.join(conf.PATH_MAPS, 'evaluation')

    env = nmmo.Env(conf)
    env.reset()

    for i in range(128):
        env.step({})

    logs = env.terminal()
    
    max_lifetime = max(logs['Stats']['Lifetime'])

    assert max_lifetime > 100, 'Best scripted model should live > 100 steps'
