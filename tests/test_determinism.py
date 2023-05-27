# pylint: disable=protected-access

'''Manual test for deterministic rollout with baseline agents'''
import random
import unittest
import numpy as np
from tqdm import tqdm

from pufferlib.emulation import Binding

from env.nmmo_env import RewardsConfig
from env.nmmo_config import NmmoConfig
from env.nmmo_team_env import NMMOTeamEnv
from lib.team.team_helper import TeamHelper
from model.realikun.model import ModelArchitecture
from lib.agent.baseline_agent import BaselineAgent

HORIZON = 30
RANDOM_SEED = random.randint(0, 100000)
MODEL_WEIGHTS = '../model_weights/achievements_4x10_new.200.pt'

def init_team_env(model_weights):
  num_teams = ModelArchitecture.NUM_TEAMS
  team_size = ModelArchitecture.NUM_PLAYERS_PER_TEAM

  config = NmmoConfig(num_teams=num_teams, team_size=team_size)
  reward_config = RewardsConfig()

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
    agent_list.append(BaselineAgent(binding, weights_path=model_weights))

  return team_env, agent_list

def get_rollout_actions(team_env, agent_list, seed):
  team_obs = team_env.reset(seed=seed, map_id=0)

  saved_actions = {}
  for step in tqdm(range(HORIZON)):
    # generate actions
    saved_actions[step] = { team_id: agent_list[team_id].act(obs)
                              for team_id, obs in team_obs.items() }
    team_obs, _, _, _ = team_env.step(saved_actions[step])

  event_log = team_env._env.realm.event_log.get_data()

  return saved_actions, event_log

def rollout_with_saved_actions(team_env, seed, saved_actions):
  team_env.reset(seed=seed, map_id=0)
  for step in tqdm(range(HORIZON)):
    team_env.step(saved_actions[step])

  # return the event log for comparison
  return team_env._env.realm.event_log.get_data()

def run_rollouts(model_weights, seed):
  team_env, agent_list = init_team_env(model_weights)
  saved_actions, src_log = get_rollout_actions(team_env, agent_list, seed)
  rep_log = rollout_with_saved_actions(team_env, seed, saved_actions)

  return src_log, rep_log, saved_actions

class TestDeterminism(unittest.TestCase):
  def test_determinism(self):
    src_log, rep_log, _ = run_rollouts(model_weights=MODEL_WEIGHTS, seed=RANDOM_SEED)

    # import pickle
    # with open('actions.pickle', 'wb') as f:
    #   pickle.dump(saved_actions, f)

    assert np.array_equal(src_log, rep_log),\
      f"The determinism test failed with the seed: {RANDOM_SEED}."


if __name__ == '__main__':
  unittest.main()

  # team_env, agent_list = init_team_env(MODEL_WEIGHTS)
  # with open('actions.pickle', 'rb') as f:
  #   saved_actions = pickle.load(f)
  # rep_log = rollout_with_saved_actions(team_env, RANDOM_SEED, saved_actions)
  # rep2_log = rollout_with_saved_actions(team_env, RANDOM_SEED, saved_actions)

  # np.savetxt("rep_log1.csv", rep_log, delimiter=',', fmt="%d")
  # np.savetxt("rep_log2.csv", rep2_log, delimiter=',', fmt="%d")

  # assert np.array_equal(rep_log, rep2_log)
