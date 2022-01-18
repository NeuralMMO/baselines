'''Documented at neuralmmo.github.io'''

from pdb import set_trace as T

import nmmo

from scripted import baselines


def simulate(env, config, render=False, horizon=float('inf')):
    '''Simulate an environment for a fixed horizon'''
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
    # Agents will be instantiated using these templates
    AGENTS    = [baselines.Meander, baselines.Forage, baselines.Combat]

    #Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'


if __name__ == '__main__':
    simulate(nmmo.Env, Config, render=True)
