from pdb import set_trace as T
import numpy as np

import os
from tqdm import tqdm

import nmmo
from nmmo import config

import tasks

from scripted import baselines

from matplotlib import pyplot as plt
import matplotlib as mpl

params = {
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.fontsize': 10,
        'lines.linewidth': 3,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': [5.5, 8],
        'figure.dpi': 200
        }

for key, val in params.items():
    mpl.rcParams[key] = val



class MediumAllSystems(config.Medium, config.AllGameSystems):
    PATH_MAPS = os.path.join(config.Small.PATH_MAPS, 'evaluation')
    MAP_FORCE_GENERATION = True

    TASKS     = tasks.All
    PLAYERS   = [
            baselines.Fisher, baselines.Herbalist,
            baselines.Prospector, baselines.Carver, baselines.Alchemist,
            baselines.Melee, baselines.Range, baselines.Mage]

class Env(nmmo.Env):
    def max(self, fn):
        return max(fn(player) for player in self.realm.players.values())

    def max_held(self, policy):
        lvls = [player.equipment.held.level.val for player in self.realm.players.values()
                if player.equipment.held is not None and player.policy == policy]

        if len(lvls) == 0:
            return 0

        return max(lvls)

    def max_item(self, policy):
        lvls = [player.equipment.item_level for player in self.realm.players.values() if player.policy == policy]

        if len(lvls) == 0:
            return 0

        return max(lvls)

    def log_env(self):
        quill  = self.realm.quill

        quill.log_env('Skill_Mage', self.max(lambda e: e.skills.mage.exp))
        quill.log_env('Skill_Range', self.max(lambda e: e.skills.range.exp))
        quill.log_env('Skill_Melee', self.max(lambda e: e.skills.melee.exp))
        quill.log_env('Skill_Fisher', self.max(lambda e: e.skills.fishing.exp))
        quill.log_env('Skill_Herbalist', self.max(lambda e: e.skills.herbalism.exp))
        quill.log_env('Skill_Prospector', self.max(lambda e: e.skills.prospecting.exp))
        quill.log_env('Skill_Carver', self.max(lambda e: e.skills.carving.exp))
        quill.log_env('Skill_Alchemist', self.max(lambda e: e.skills.alchemy.exp))

        for key in 'Mage Range Melee Fisher Herbalist Prospector Carver Alchemist'.split():
            quill.log_env(f'Held_{key}', self.max_held(key))
            quill.log_env(f'Item_{key}', self.max_item(key))

        #quill.log_env(f'Wealth', self.max(lambda e: e.inventory.gold.quantity.val))
        #quill.log_env(f'{policy} Held Item Level', self.max(lambda e: e.equipment.held.level.val if e.equipment.held else 0))

def preprocess(all_stats):
    #Preprocess
    stats = {}
    keys  = list(all_stats[0].keys())
    for key in keys:
        key_stats = [e[key] for e in all_stats]
        stats[key] = np.mean(key_stats, axis=0).tolist()

    return stats


def plots(all_stats):
    stats = preprocess(all_stats)

    fig, axs = plt.subplots(3)

    x = np.arange(HORIZON + 1)
    for key, vals in stats.items():
        idx = None
        if key.startswith('Skill_'):
            idx = 0
        elif key.startswith('Held_'):
            idx = 1
        elif key.startswith('Item_'):
            idx = 2
        else:
            continue
        
        key = key.split('_')[-1]

        if idx == 0:
            vals = np.array(vals) / 65000
            vals = vals.tolist()

        axs[idx].plot(x, vals, label=key)

    for i in range(3):
        axs[i].set_xticks(np.arange(0, HORIZON+1, HORIZON//10))

    axs[0].legend(loc='upper left', ncol=2)
    axs[0].set_ylabel('Normalized Experience')
    axs[0].set_yticks(np.arange(0, 1.01, 0.1))
    axs[0].set_title(f'{SAVE_TO.capitalize()} Progress by Profession')

    axs[1].set_ylabel('Tool or Weapon Level')
    axs[1].set_yticks(np.arange(0, 6))

    axs[2].set_ylabel('Equipment Score')
    axs[2].set_yticks(np.arange(0, 21, 2))
    axs[2].set_xlabel('Simulation Time Step')


    fig.tight_layout()
    fig.align_ylabels()
    plt.savefig(f'{SAVE_TO}.pdf')

def load_combined(log_file):
    log_file = np.load(log_file, allow_pickle=True)[0]

    stats = {}
    start_keys = 'Skill_ Held_ Item_'.split()

    for start in start_keys:
        stats[start] = np.mean([log_file[e] for e in log_file if e.startswith(start)], 0)

    return stats


def plot_combined():
    generalist = load_combined('generalist.npy')
    specialist = load_combined('specialist.npy')
        
    fig, axs = plt.subplots(3)

    x = np.arange(HORIZON + 1)

    g, s = 'Generalist', 'Specialist'

    axs[0].plot(x, generalist['Skill_']/65000, label=g)
    axs[0].plot(x, specialist['Skill_']/65000, label=s)
    axs[0].legend(loc='upper left', ncol=2)
    axs[0].set_ylabel('Normalized Experience')
    axs[0].set_yticks(np.arange(0, 1.01, 0.1))
    axs[0].set_title('Aggregated Progress Over Simulation')

    axs[1].plot(x, generalist['Held_'], label=g)
    axs[1].plot(x, specialist['Held_'], label=s)
    axs[1].set_ylabel('Tool or Weapon Level')
    axs[1].set_yticks(np.arange(0, 6))

    axs[2].plot(x, generalist['Item_'], label=g)
    axs[2].plot(x, specialist['Item_'], label=s)
    axs[2].set_ylabel('Equipment Score')
    axs[2].set_yticks(np.arange(0, 21, 2))
    axs[2].set_xlabel('Simulation Time Step')

    for i in range(3):
        axs[i].set_xticks(np.arange(0, HORIZON+1, HORIZON//10))

    fig.tight_layout()
    fig.align_ylabels()
    plt.savefig(f'{SAVE_TO}.pdf')
    
def test_scripted():
    all_stats = []
    for trial in range(TRIALS):
        conf = MediumAllSystems()
        conf.SPECIALIZE = SAVE_TO == 'specialist'
        conf.LOG_FILE   = f'{SAVE_TO}.txt'

        env  = Env(conf)
        env.reset()

        for i in tqdm(range(HORIZON)):
            #env.render()
            env.step({})

        logs  = env.terminal()
        stats = logs['Player']

        for key, vals in stats.items():
            print(f'{key}: {min(vals)}, {np.mean(vals)}, {max(vals)}')

        all_stats.append(logs['Env'])

    np.save(f'{SAVE_TO}.npy', all_stats)

    plots(all_stats)
    

if __name__ == '__main__':
    TRIALS  = 10
    HORIZON = 1000

    SAVE_TO = 'combined'

    plot_combined()

    #test_scripted()
