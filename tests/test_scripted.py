from pdb import set_trace as T

import nmmo

from nmmo import config, achievement
from scripted import baselines

def test_scripted():
    conf              = config.Small()
    conf.ACHIEVEMENTS = [achievement.PlayerKills, achievement.Equipment, achievement.Exploration, achievement.Foraging]
    conf.AGENTS       = [baselines.Meander, baselines.Forage, baselines.Combat]

    env = nmmo.Env(conf)
    env.reset()

    for i in range(128):
        env.step({})

    logs = env.terminal()
    
    max_lifetime = max(logs['Stats']['Lifetime'])

    assert max_lifetime > 100, 'Best scripted model should live > 100 steps'
