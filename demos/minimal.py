'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo

# Scripted models included with the baselines repository
from scripted import baselines


def simulate(env, config, render=False, horizon=float('inf')):
    '''Simulate an environment for a fixed horizon'''

    # Environment accepts a config object
    env = env(config())
    env.reset()

    t = 0
    while True:
        if render:
            env.render()

        # Scripted API computes actions
        obs, rewards, dones, infos = env.step(actions={})

        # Later examples will use a fixed horizon
        t += 1
        if t >= horizon:
            break

    # Called at the end of simulation to obtain logs
    return env.terminal()


class Config(nmmo.config.Small, nmmo.config.AllGameSystems):
    '''Config objects subclass a nmmo.config.{Small, Medium, Large} template

    Can also specify config game systems to enable various features'''

    # Agents will be instantiated using templates included with the baselines
    # Meander: randomly wanders around
    # Forage: explicitly searches for food and water
    # Combat: forages and actively fights other agents
    AGENTS    = [baselines.Meander, baselines.Forage, baselines.Combat]

    #Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'


if __name__ == '__main__':
    simulate(nmmo.Env, Config, render=True)
