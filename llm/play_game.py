from pdb import set_trace as T
from tqdm import tqdm

import nmmo

from scripted import baselines
from generated_agent import Agent


SEED = 42

config = nmmo.config.Default()
config.PLAYERS = [Agent]
config.LOG_ENV = True
config.LOG_EVENTS = True
config.LOG_MILESTONES = True
config.LOG_VERBOSE = True
#config.LOG_FILE = 'log.txt'

env = nmmo.Env(config, seed=SEED)

obs = env.reset(seed=SEED)
for t in tqdm(range(128)):
    obs = env.step({})

logs = env.realm.log_helper


for key, vals in logs.packet['Milestone'].items():
    print(f'{key}: Maximum value of {max(vals)}')