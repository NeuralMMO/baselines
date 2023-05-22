'''Manual test for deterministic rollout with baseline agents'''
from tqdm import tqdm
import numpy as np
import random

import nmmo
from pufferlib.emulation import Binding

from env.nmmo_env import RewardsConfig
from env.nmmo_team_env import NMMOTeamEnv
from lib.team.team_helper import TeamHelper
from model.realikun.model import ModelArchitecture
from model.realikun.baseline_agent import BaselineAgent

HORIZON = 30
MODEL_WEIGHTS = '../model_weights/achievements_4x10_new.200.pt'

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
  PLAYER_N = ModelArchitecture.NUM_TEAMS * ModelArchitecture.NUM_PLAYERS_PER_TEAM
  NPC_N = 64

def get_rollout_actions(team_env, agent_list, seed):
  team_obs = team_env.reset(seed=seed)

  saved_actions = {}
  for step in tqdm(range(HORIZON)):
    # generate actions
    saved_actions[step] = { team_id: agent_list[team_id].act(obs)
                              for team_id, obs in team_obs.items() }
    team_obs, _, _, _ = team_env.step(saved_actions[step])

  event_log = team_env._env.realm.event_log.get_data()

  return saved_actions, event_log

def rollout_with_saved_actions(team_env, seed, saved_actions):
  team_env.reset(seed=seed)
  for step in tqdm(range(HORIZON)):
    team_env.step(saved_actions[step])

  # return the event log for comparison
  return team_env._env.realm.event_log.get_data()

def test_determinism(seed, model_weights=MODEL_WEIGHTS):
  config = ReplayConfig()
  reward_config = RewardsConfig()

  # NOTE: using hardcoded values
  team_size = ModelArchitecture.NUM_PLAYERS_PER_TEAM
  num_teams = ModelArchitecture.NUM_TEAMS
  team_helper = TeamHelper({
    i: [i*team_size+j+1 for j in range(team_size)]
    for i in range(num_teams)}
  )

  binding = Binding(
    env_creator=lambda: NMMOTeamEnv(config, team_helper, reward_config),
    env_name="Neural MMO",
    suppress_env_prints=False,
  )

  team_env = binding.raw_env_creator()  # NMMOTeamEnv

  agent_list = []
  for _ in range(num_teams):
    agent_list.append(BaselineAgent(model_weights, binding))

  saved_actions, src_log = get_rollout_actions(team_env, agent_list, seed)
  rep_log = rollout_with_saved_actions(team_env, seed, saved_actions)

  assert np.array_equal(src_log, rep_log), "Two rollouts are not the same."


if __name__ == '__main__':
  seed = random.randint(0, 100000)
  test_determinism(seed)
  print(f"Test passed with seed {seed}")