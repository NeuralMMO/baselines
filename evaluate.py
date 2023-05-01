# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import os
import time
import colorsys
import json
import lzma
import pickle

import random
import numpy as np

import torch

import nmmo
import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.registry.nmmo
import pufferlib.vectorization.serial

import cleanrl_ppo_lstm
from model.policy import BaselinePolicy
from model.simple.simple_policy import SimplePolicy
from team_helper import TeamHelper


def replay_config(num_teams, team_size):
  class ReplayConfig(
    nmmo.config.Medium,
    nmmo.config.Terrain,
    nmmo.config.Resource,
    nmmo.config.Progression,
    # nmmo.config.Profession
    nmmo.config.Combat
  ):
    SAVE_REPLAY = True
    PROVIDE_ACTION_TARGETS = True
    PLAYER_N = num_teams * team_size

  return ReplayConfig()


##################################################
# save_replay helper functions
def rainbow_colormap(n):
  colormap = []
  for i in range(n):
    hue = i / float(n)
    r, g, b = tuple(int(255 * x) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
    hexcode = f'#{r:02x}{g:02x}{b:02x}'
    colormap.append(hexcode)
  return colormap

def apply_team_color(replay, team_helper):
  colormap = rainbow_colormap(team_helper.num_teams)
  for packet in replay['packets']:
    for ent_id, player in packet['player'].items():
      team_id, pos = team_helper.team_and_position_for_agent[ent_id]
      # assign team info
      player['base']['name'] = f'Team{team_id:02d}-{pos}'
      player['base']['population'] = team_id
      player['base']['color'] = colormap[team_id]
  return replay

# use to fix json.dumps() cannot serialize numpy objects
# pylint: disable=inconsistent-return-statements
def np_encoder(obj):
  if isinstance(obj, np.generic):
    return obj.item()

def save_replay(
      model_arch,
      model_checkpoint,
      seed=1,
      num_teams=16,
      team_size=8,
      save_dir=None):

  config = replay_config(num_teams, team_size)
  team_helper = TeamHelper({
    i: [i*team_size+j+1 for j in range(team_size)]
    for i in range(num_teams)}
  )

  policy_cls = BaselinePolicy # realikun
  if model_arch == "simple":
    policy_cls = SimplePolicy

  binding = pufferlib.emulation.Binding(
    env_creator=policy_cls.env_creator(config, team_helper),
    env_name="Neural MMO",
    suppress_env_prints=False,
  )
  agent = policy_cls.create_policy()(binding)

  print(f"Initializing model from {model_checkpoint}...")
  cleanrl_ppo_lstm.load_matching_state_dict(
    agent,
    torch.load(model_checkpoint)["agent_state_dict"]
  )

  # TRY NOT TO MODIFY: seeding
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  ####################################################
  # some constants, keeping the same as train()
  # NOT using buffers, using only single env to make the replay
  num_cores = 1
  envs_per_worker = 1
  num_envs = num_cores * envs_per_worker
  num_agents = policy_cls.num_agents(team_helper)

  envs = pufferlib.vectorization.serial.VecEnv(
    binding,
    num_workers=num_cores,
    envs_per_worker=int(envs_per_worker),
  )
  agent = agent.to(device)

  # ALGO Logic: Storage setup, to cross-examine with the replay. for one env only
  #   CHECK ME: num_agents is 16 .. is this for one team only?
  actions = [] # dim: (steps, num_agents, action_space_dim)
  logprobs = [] # dim: (steps, num_agents)
  rewards = [] #torch.zeros(num_agents).to(device)
  dones = [] #torch.zeros(num_agents).to(device)
  values = [] #torch.zeros(num_agents).to(device)

  next_obs, next_done, next_lstm_state = [], [], []
  envs.async_reset()
  o, _, _, _ = envs.recv()
  next_obs = torch.Tensor(o).to(device)
  next_done = torch.zeros((num_envs * num_agents,)).to(device)
  next_lstm_state = (
      torch.zeros(agent.lstm.num_layers, num_envs * num_agents, agent.lstm.hidden_size).to(device),
      torch.zeros(agent.lstm.num_layers, num_envs * num_agents, agent.lstm.hidden_size).to(device),
  )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

  epoch_lengths = []
  epoch_returns = []

  puffer_env = envs.envs_lists[0].envs[0]
  step = 0
  while puffer_env.done is False:
    # TRY NOT TO MODIFY: Receive from game and log data
    if step == 0:
      dones.append(next_done.cpu())
    else:
      o, r, d, i = envs.recv()

      next_obs = torch.Tensor(o).to(device)
      next_done = torch.Tensor(d).to(device)

      dones.append(next_done.cpu())
      rewards.append(torch.tensor(r).float().view(-1).cpu()) # CHECK if this is correct

      for item in i:
        if "episode" in item.keys():
          epoch_lengths.append(item["episode"]["l"])
          epoch_returns.append(item["episode"]["r"])

    # ALGO LOGIC: action logic
    with torch.no_grad():
      action, logprob, _, value, next_lstm_state = \
        agent.get_action_and_value(next_obs, next_lstm_state, next_done)
      values.append(value.flatten().cpu())

    actions.append(action.cpu())
    logprobs.append(logprob.cpu())

    # TRY NOT TO MODIFY: execute the game
    envs.send(action.cpu().numpy(), None)

    step += 1

  print('Seed', seed, 'roll-out complete after', step-1, 'steps.')

  # CHECK ME: other way to get to the env realm?
  #   puffer_env -> team_env -> nmmo_env -> realm
  replay = apply_team_color(puffer_env.env._env.realm.get_replay(),
                            team_helper)

  # check save_dir, create file name
  if save_dir is None:
    save_dir = f"replay_{model_arch}"
  os.makedirs(save_dir, exist_ok=True)
  checkpoint_name = os.path.basename(model_checkpoint).split('.')[0]
  filename_body = f"{checkpoint_name}_{seed:04d}_{int(time.time())}"

  # save replay
  data = json.dumps(replay, default=np_encoder).encode('utf8')
  data = lzma.compress(data, format=lzma.FORMAT_ALONE)
  save_file = os.path.join(save_dir, 'replay_' + filename_body + '.7z')
  with open(save_file, 'wb') as out:
    out.write(data)
    print(f'Saved the replay {seed:04d} to {save_file}...')

  # save additional info: teams, event_log
  replay_info = {}
  replay_info['teams'] = team_helper.teams
  replay_info['event_log'] = puffer_env.env._env.realm.event_log.get_data()
  with open(os.path.join(save_dir, 'supplement_' + filename_body + '.pkl'), 'wb') as out:
    pickle.dump(replay_info, out)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--model.arch",
    dest="model_arch", choices=["realikun", "simple"],
    default="realikun",
    help="model architecture (default: realikun)")

  parser.add_argument(
    "--model.checkpoint",
    dest="model_checkpoint", type=str,
    default="model_weights/realikun.001470.pt", # "model_weights/simple.049500.pt",
    help="path to model checkpoint to load")

  parser.add_argument(
    "--env.seed", dest="seed", type=int, default=1,
    help="random seed to initialize the env (default: 1)")
  parser.add_argument(
    "--env.num_teams", dest="num_teams", type=int, default=16,
    help="number of teams to use for training (default: 16)")
  parser.add_argument(
    "--env.team_size", dest="team_size", type=int, default=8,
    help="number of agents per team to use for training (default: 8)")

  parser.add_argument(
    "--eval.num_rounds", dest="num_rounds", type=int, default=1,
    help="number of rounds to use for evaluation (default: 1)")
  parser.add_argument(
    "--eval.save_dir", dest="save_dir", type=str, default=None,
    help="path to save replay files (default: auto-generated)")

  args = parser.parse_args()

  for ri in range(args.num_rounds):
    print('Generating the replay for round', ri+1, 'with seed', args.seed+ri)
    save_replay(
      model_arch=args.model_arch,
      model_checkpoint=args.model_checkpoint,
      seed=args.seed+ri,
      num_teams=args.num_teams,
      team_size=args.team_size,
      save_dir=args.save_dir,
    )
