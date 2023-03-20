import nmmo
from typing import Any, Dict, List

class TeamEnv():
    def __init__(self, env: nmmo.Env, player_to_team_map: Dict[int, int]):
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
        Groups the observation space by team.

        Returns:
            Dict: The observation space grouped by team.
        """
        return self._group_by_team(self.env.observation_space)
    
    def grouped_action_space(self):
        """
        Groups the action space by team.

        Returns:
            Dict: The action space grouped by team.
        """
        return self._group_by_team(self.env.action_space)

    def _group_by_team(self, func):
        """
        Groups the function output by team.

        Args:
            func (function): The function to group.

        Returns:
            Dict: The output of the function grouped by team.
        """
        group = {}
        for agent_id, team_id in self.player_to_team_map.items():
            if team_id not in group:
                group[team_id] = {}
            group[team_id][agent_id] = func(agent_id)
        return group

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
        return self._group_by_team(gym_obs.get)

    def step(self, actions: Dict[int, Any]):
        """
        Executes one time step of the environment with the given actions.

        Args:
            actions (Dict): A dictionary that maps agent IDs to their action dictionaries.

        Returns:
            Tuple: A tuple of merged observations, merged rewards, merged dones, and merged infos, all grouped by team.
        """
        gym_obs, rewards, dones, infos = self.env.step(actions)

        merged_obs = self._group_by_team(gym_obs.get)
        merged_rewards = self._merge_rewards(rewards)
        merged_infos = self._group_by_team(infos.get)
        merged_dones = self._group_by_team(dones.get)

        return merged_obs, merged_rewards, merged_dones, merged_infos
    
    def _merge_rewards(self, rewards):
        """
        Merge rewards by team.

        Args:
            rewards (dict): A dictionary where keys are agent IDs and values are rewards.

        Returns:
            Dict: A dictionary where keys are team IDs and values are the sum of rewards for the team.
        """
        merged_rewards = {}
       
        for agent_id, reward in rewards.items():
            team_id = self.player_to_team_map[agent_id]
            prev_reward = merged_rewards.get(team_id, 0)
            merged_rewards[team_id] = prev_reward + reward

        return merged_rewards
