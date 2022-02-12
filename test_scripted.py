from pdb import set_trace as T
import numpy as np

import os
from tqdm import tqdm

import nmmo
from nmmo import config

import tasks

from scripted import baselines

class MediumAllSystems(config.Small, config.AllGameSystems):
    PATH_MAPS = os.path.join(config.Small.PATH_MAPS, 'evaluation')
    MAP_FORCE_GENERATION = True

    TASKS     = tasks.All
    PLAYERS   = [
            baselines.Fisher, baselines.Herbalist,
            baselines.Prospector, baselines.Carver, baselines.Alchemist,
            baselines.Melee, baselines.Range, baselines.Mage]


def test_scripted():
    conf = MediumAllSystems()
    env  = nmmo.Env(conf)
    env.reset()

    for i in tqdm(range(128)):
        #env.render()
        env.step({})

    logs  = env.terminal()
    stats = logs['Stats']
    
    for key, vals in stats.items():
        print(f'{key}: {min(vals)}, {np.mean(vals)}, {max(vals)}')

    max_lifetime = max(stats['Lifetime'])
    print(max_lifetime)

    assert max_lifetime > 100, 'Best scripted model should live > 100 steps'

if __name__ == '__main__':
    test_scripted()
