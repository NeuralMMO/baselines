import unittest
from unittest.mock import Mock
import nmmo
from typing import Any, Dict, List
from model.TeamEnv import TeamEnv
from scripted import baselines
import random

RANDOM_SEED = random.randint(0, 10000)

# Define configuration for the test
class Config(nmmo.config.Small, nmmo.config.AllGameSystems):
    RENDER = False
    SPECIALIZE = True
    PLAYERS = [
    baselines.Fisher, baselines.Herbalist, baselines.Prospector,
    baselines.Carver, baselines.Alchemist,
    baselines.Melee, baselines.Range, baselines.Mage]

# Define a test case for TeamEnv
class TestTeamEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up the environment and map players to teams
        cls.config = Config()
        env = nmmo.Env(cls.config, RANDOM_SEED)
        team = 0 
        cls.player_to_team_map = {}
        for agent in range(1, 65):
            cls.player_to_team_map[agent] = team
            if team%8 == 0:
                team +=1
        cls.team_env = TeamEnv(env, cls.player_to_team_map)

    def test_grouped_observation_space(self):
        # Test that the grouped observation space is as expected
        expected_obs_space = {}
        for agent_id, team_id in self.player_to_team_map.items():
            if team_id not in expected_obs_space:
                expected_obs_space[team_id] = {}
            expected_obs_space[team_id][agent_id] = self.team_env.env.observation_space(agent_id)

        self.assertEqual(self.team_env.grouped_observation_space(), expected_obs_space)

    def test_grouped_action_space(self):
        # Test that the grouped action space is as expected
        expected_atn_space = {}
        for agent_id, team_id in self.player_to_team_map.items():
            if team_id not in expected_atn_space:
                expected_atn_space[team_id] = {}
            expected_atn_space[team_id][agent_id] = self.team_env.env.action_space(agent_id)

        self.assertEqual(self.team_env.grouped_action_space(), expected_atn_space)

    def test_reset(self):
        # get the expected merged observation space and the actual merged observation space
        expected_merged_obs = self.team_env._merge_obs(self.team_env.env.reset())
        merged_obs = self.team_env.reset()
        # validate that the two merged observation spaces are equal
        self._validate_merged_dicts(merged_obs, expected_merged_obs)

    # a helper method to validate two merged dictionaries
    def _validate_merged_dicts(self, expected_merged_obs, merged_obs):
        # check if same number of teams
        assert expected_merged_obs.keys() == merged_obs.keys()
        # check if same number of agents in each team
        for team in merged_obs.keys():
            assert expected_merged_obs[team].keys() == merged_obs[team].keys()

    def test_step(self):
        # take a step with an empty action and get the expected and actual merged observations, rewards, dones and infos
        obs, rewards, dones, infos  = self.team_env.env.step({})
        expected_merged_obs = self.team_env._merge_obs(obs)
        expected_merged_rewards, expected_merged_infos = self.team_env._merge_rewards_infos(rewards, infos)
        expected_merged_dones = self.team_env._merge_dones(dones)
        merged_obs, merged_rewards, merged_dones, merged_infos = self.team_env.step({})

        # validate that the merged observation spaces and infos are equal
        self._validate_merged_dicts(merged_obs, expected_merged_obs)
        self._validate_merged_dicts(merged_infos, expected_merged_infos)
        
        # validate that the merged rewards and dones are equal
        self.assertEqual(merged_dones, expected_merged_dones)
        self.assertDictEqual(merged_rewards, expected_merged_rewards)

if __name__ == '__main__':
    unittest.main()
