import nmmo
from typing import Any, Dict, List

class TeamEnv():

    def __init__(self, env: nmmo.Env, player_to_team_map):
        """
        Initializes TeamEnv class.

        Args:
            env (nmmo.Env): The gym environment to wrap.
            player_to_team_map (Dict): A dictionary that maps player IDs to their team IDs.
        """
        self.env = env
        self.player_to_team_map = player_to_team_map

    def grouped_observation_space(self):
        """
        Returns the observation space for each team.

        Returns:
            Dict: A dictionary that maps team IDs to their observation spaces.
        """
        obs_space = {}
        for agent_id, team_id in self.player_to_team_map.items():
            if team_id not in obs_space:
                obs_space[team_id] = {}
            obs_space[team_id][agent_id] = self.env.observation_space(agent_id)
        
        return obs_space

    def grouped_action_space(self):
        """
        Returns the action space for each team.

        Returns:
            Dict: A dictionary that maps team IDs to their action spaces.
        """
        atn_space = {}
        for agent_id, team_id in self.player_to_team_map.items():
            if team_id not in atn_space:
                atn_space[team_id] = {}
            atn_space[team_id][agent_id] = self.env.action_space(agent_id)
        
        return atn_space

    def reset(self, map_id=None, seed=None, options=None):
        """
        Resets the environment and returns the initial observation.

        Args:
            map_id (int, optional): ID of the map to use. Defaults to None.
            seed (int, optional): Seed to use for the environment's RNG. Defaults to None.
            options (Dict, optional): Additional options to pass to the environment. Defaults to None.

        Returns:
            Dict: The initial observation of the wrapped environment, grouped by team.
        """
        gym_obs = self.env.reset(map_id, seed, options)
        merged_obs = self._merge_obs(gym_obs)

        return merged_obs 

    def step(self, actions: Dict[int, Dict[str, Dict[str, Any]]]):
        """
        Executes one time step of the environment with the given actions.

        Args:
            actions (Dict): A dictionary that maps agent IDs to their action dictionaries.

        Returns:
            Tuple: A tuple of merged observations, merged rewards, merged dones, and merged infos, all grouped by team.
        """
        gym_obs, rewards, dones, infos = self.env.step(actions)

        merged_obs = self._merge_obs(gym_obs)
        merged_rewards, merged_infos = self._merge_rewards_infos(rewards, infos)
        merged_dones = self._merge_dones(dones)

        return merged_obs, merged_rewards, merged_dones, merged_infos
    
    def _merge_obs(self, obs):
        """
        Merges the observations for each agent into a single dictionary grouped by team.

        Args:
            obs (Dict): A dictionary that maps agent IDs to their observations.

        Returns:
            Dict: The merged observations, grouped by team.
        """
        merged_obs = {}
     
        for agent, obs in obs.items():
            team_id = self.player_to_team_map[agent]
            if team_id not in merged_obs:
                merged_obs[team_id] = {}
            merged_obs[team_id][agent] = obs

        return merged_obs
    
    def _merge_rewards_infos(self, rewards, infos):
        """
        Merge rewards and infos dictionaries by team.

        Args:
            rewards (dict): A dictionary where keys are agent IDs and values are rewards.
            infos (dict): A dictionary where keys are agent IDs and values are info objects.

        Returns:
            A tuple containing two dictionaries:
            - merged_rewards: A dictionary where keys are team IDs and values are the sum of rewards for the team.
            - merged_infos: A nested dictionary where keys are team IDs, and values are dictionaries containing
                info objects for each agent in the team.
        """
        merged_rewards = {}
        merged_infos = {}
        
        for agent, reward in rewards.items():
            team_id = self.player_to_team_map[agent]
            prev_reward = merged_rewards.get(team_id, 0)
            merged_rewards[team_id] = prev_reward + reward

        for agent, info in infos.items():
            team_id = self.player_to_team_map[agent]
            if team_id not in merged_infos:
                merged_infos[team_id] = {}
            merged_infos[team_id][agent] = info

        return merged_rewards, merged_infos

    def _merge_dones(self, dones):
        """
        Merge dones dictionary by team.

        Args:
            dones (dict): A dictionary where keys are agent IDs and values are done flags.

        Returns:
            A dictionary where keys are team IDs, and values are dictionaries containing done flags for each agent
            in the team.
        """
        merged_dones = {}
        for agent, done in dones.items():
            team_id = self.player_to_team_map[agent]
            if team_id not in merged_dones:
                merged_dones[team_id] = {}
            merged_dones[team_id][agent] = done
        
        return merged_dones