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

from env.nmmo_team_env import NMMOTeamEnv
from lib.team.team_helper import TeamHelper
from model.realikun.baseline_agent import BaselineAgent


def replay_config(num_teams, team_size, num_npcs):
  class ReplayConfig(
    nmmo.config.Medium,
    nmmo.config.Terrain,
    nmmo.config.Resource,
    nmmo.config.NPC,
    nmmo.config.Progression,
    nmmo.config.Equipment,
    nmmo.config.Item,
    nmmo.config.Exchange,
    nmmo.config.Profession,
    nmmo.config.Combat,
  ):
    SAVE_REPLAY = True
    PROVIDE_ACTION_TARGETS = True
    PLAYER_N = num_teams * team_size
    NPC_N = num_npcs

    MAP_PREVIEW_DOWNSCALE        = 8
    MAP_CENTER                   = 64

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
      model_checkpoint,
      seed=1,
      num_teams=16,
      team_size=8,
      num_npcs=0,
      save_dir=None):

  config = replay_config(num_teams, team_size, num_npcs)
  team_helper = TeamHelper({
    i: [i*team_size+j+1 for j in range(team_size)]
    for i in range(num_teams)}
  )

  binding = pufferlib.emulation.Binding(
    env_creator=lambda: NMMOTeamEnv(config, team_helper),
    env_name="Neural Team MMO",
    suppress_env_prints=False,
  )

  agent_list = []
  for _ in range(num_teams):
    agent_list.append(BaselineAgent(model_checkpoint, binding))

  # TRY NOT TO MODIFY: seeding
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

  team_env = binding.raw_env_creator()  # NMMOTeamEnv

  step = 0
  actions = {}
  while True:
    if step == 0:
      team_obs = team_env.reset(seed=seed)
    else:
      team_obs, _, _, _ = team_env.step(actions)

    if len(team_obs):
      # get actions for the next tick
      actions = { team_id: agent_list[team_id].act(obs)
                    for team_id, obs in team_obs.items() }
      step += 1
    else:
      break

  print('Seed', seed, 'roll-out complete after', step-1, 'steps.')

  # CHECK ME: other way to get to the env realm?
  #   puffer_env -> team_env -> nmmo_env -> realm
  replay = apply_team_color(team_env._env.realm.get_replay(),
                            team_helper)

  # check save_dir, create file name
  if save_dir is None:
    save_dir = "replays"
  os.makedirs(save_dir, exist_ok=True)
  checkpoint_name = os.path.basename(model_checkpoint).split('.')[0]
  filename_body = f"{checkpoint_name}_{seed:04d}_{int(time.time())}"

  # save replay
  data = json.dumps(replay, default=np_encoder).encode('utf8')
  data = lzma.compress(data, format=lzma.FORMAT_ALONE)
  save_file = os.path.join(save_dir, 'replay_' + filename_body + '.lz')
  with open(save_file, 'wb') as out:
    out.write(data)
    print(f'Saved the replay {seed:04d} to {save_file}...')

  # save additional info: teams, event_log
  replay_info = {}
  replay_info['teams'] = team_helper.teams
  replay_info['event_log'] = team_env._env.realm.event_log.get_data()
  replay_info['event_attr_col'] = team_env._env.realm.event_log.attr_to_col

  with open(os.path.join(save_dir, 'supplement_' + filename_body + '.pkl'), 'wb') as out:
    pickle.dump(replay_info, out)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--model.checkpoint",
    dest="model_checkpoint", type=str,
    default="model_weights/realikun.001470.pt",
    help="path to model checkpoint to load")

  parser.add_argument(
    "--env.seed", dest="seed", type=int, default=1,
    help="random seed to initialize the env (default: 1)")
  parser.add_argument(
    "--env.num_teams", dest="num_teams", type=int, default=16,
    help="number of teams to use for replay (default: 16)")
  parser.add_argument(
    "--env.team_size", dest="team_size", type=int, default=8,
    help="number of agents per team to use for replay (default: 8)")
  parser.add_argument(
    "--env.num_npcs", dest="num_npcs", type=int, default=0,
    help="number of NPCs to use for replay (default: 0)")


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
      model_checkpoint=args.model_checkpoint,
      seed=args.seed+ri,
      num_teams=args.num_teams,
      team_size=args.team_size,
      num_npcs=args.num_npcs,
      save_dir=args.save_dir,
    )
