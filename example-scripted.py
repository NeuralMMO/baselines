'''Scripted-only example not dependent upon RLlib and WanDB'''

from pdb import set_trace as T
from tqdm import tqdm
import numpy as np

from nmmo import Env, config

from scripted import baselines
import tasks


HORIZON = 1024

class Config(config.Default):
    AGENTS  = [baselines.Combat]
    TASKS   = tasks.All

conf = Config()
env  = Env(conf)

env.reset()
for t in tqdm(range(HORIZON)):
    actions = {} #Scripted API computes actions
    obs, rewards, dones, infos = env.step(actions=actions)

stats = env.terminal()['Stats']

for key, vals in stats.items():
    print('{}: {}'.format(key, np.mean(vals)))

print('{}: {}'.format('Max Achievements', max(stats['Achievements_Completed'])))
