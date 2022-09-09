from pdb import set_trace as T
import numpy as np
import os

import ray
from tqdm import tqdm

import nmmo

from config import cleanrl
from scripted import baselines


def plot_wandb(data):
    aggregated = {}

    # Mean over data
    for category, log_data in data[0].items():
        if category != 'Env':
            continue

        for group in log_data:
            dat = [d[category][group] for d in data]
            #TODO: Thes eshould be same len
            llen = min([len(e) for e in dat])
            dat = [d[:llen] for d in dat]
            vals = np.mean(dat, axis=0).tolist()
            lens = np.arange(len(vals)).tolist()

            split = group.split('_')
            prefix = split[0]
            suffix = '_'.join(split[1:])

            if prefix not in aggregated:
                aggregated[prefix] = {'keys': [], 'ys': [], 'xs': []}

            aggregated[prefix]['keys'].append(suffix)
            aggregated[prefix]['xs'].append(lens)
            aggregated[prefix]['ys'].append(vals)

    # WanDB integration
    with open('wandb_api_key') as key:
        os.environ['WANDB_API_KEY'] = key.read().rstrip('\n')
        run_name = f'test_wandb_plotting'

        import wandb
        wandb.init(
            project='NeuralMMO-Dynamics',
            entity=None,
            name='simulation',
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True)

    for group, data in tqdm(aggregated.items()):
        plot = wandb.plot.line_series(
            xs = data['xs'],
            ys = data['ys'],
            keys = data['keys'],
            title = group,
            xname = 'ticks',
            )
        wandb.log({group: plot})


@ray.remote
def simulate_env(conf, idx):
    env = nmmo.Env(conf)
    env.reset(idx+1)

    for i in range(HORIZON):
        env.step({})

    logs  = env.terminal()
    stats = logs['Player']

    for key, vals in stats.items():
        print(f'{key}: {min(vals)}, {np.mean(vals)}, {max(vals)}')

    return logs

def render(conf):
    conf.RENDER = True
    env = nmmo.Env(conf)
    env.reset()
    env.render()

    while True:
        env.step({})
        env.render()


if __name__ == '__main__':
    TRIALS  = 1
    HORIZON = 250

    conf = cleanrl.Eval()
    conf.PLAYERS = [
            baselines.Fisher, baselines.Herbalist, baselines.Prospector, baselines.Carver, baselines.Alchemist,
            baselines.Melee, baselines.Range, baselines.Mage]
    conf.MAP_N = TRIALS
    conf.LOG_FILE = 'env_logs.txt'
    conf.LOG_ENV = True
    conf.LOG_EVENTS = True
    conf.LOG_VERBOSE = True

    conf.MAP_FORCE_GENERATION = True
    nmmo.MapGenerator(conf).generate_all_maps()
    conf.MAP_FORCE_GENERATION = False

    render(conf)

    all_stats = []
    for i in range(TRIALS):
        remote = simulate_env.remote(conf, i)
        all_stats.append(remote)

    all_stats = ray.get(all_stats)
    np.save(f'logs.npy', all_stats)

    #plot_wandb(all_stats)
