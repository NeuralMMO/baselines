'''Documented at neuralmmo.github.io

Google Fire CLI for this configuration demo

python -m fire demos.configuration {evaluate, render} <CONFIG_CLASS>'''


from pdb import set_trace as T
import numpy as np
import sys

from fire import Fire

import nmmo

# Scripted models included with the baselines repository
from scripted import baselines
from demos import minimal


class Base(nmmo.config.Medium):
    '''Base environment: empty map with agents'''
    PLAYERS = [baselines.Explore]

class Forage(Base, nmmo.config.Resource, nmmo.config.Terrain):
    '''Forage for food and water'''
    PLAYERS = [baselines.Forage]

class Fight(Forage, nmmo.config.Combat):
    '''Fight for control of resources'''
    PLAYERS = [baselines.Range]
    PROGRESSION_BASE_XP_SCALE = 1000

class Specialize(Fight, nmmo.config.Progression):
    '''Specialize training to gain an upper hand'''
    PLAYERS = [baselines.Melee, baselines.Range, baselines.Mage]

class Equip(Specialize, nmmo.config.NPC, nmmo.config.Item, nmmo.config.Equipment):
    '''Equip armor and weapons'''
    pass

class Exchange(Equip, nmmo.config.Exchange):
    '''Exchange goods on a central market'''
    pass

class Skill(Exchange, nmmo.config.Profession):
    '''Skill++: lots of additional profesions for added complexity'''
    PLAYERS = [baselines.Melee, baselines.Range, baselines.Mage,
              baselines.Fisher, baselines.Herbalist,
              baselines.Prospector, baselines.Carver, baselines.Alchemist]

def get_config(name):
    try:
        return getattr(sys.modules[__name__], name)
    except AttributeError as e:
        print('Specify a valid config')
        raise

def summary(logs):
    stats = logs['Player']

    for key, vals in stats.items():
        print(f'{key}: {min(vals)}, {np.mean(vals)}, {max(vals)}')

def evaluate(config):
    config = get_config(config)
    logs = minimal.simulate(nmmo.Env, config, horizon=128)
    summary(logs)

def render(config):
    config = get_config(config)
    logs = minimal.simulate(nmmo.Env, config, render=True)
    summary(logs)
