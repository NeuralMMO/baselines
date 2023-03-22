from collections import defaultdict
from typing import Any, Dict, List

import gym
from pettingzoo.utils.env import AgentID, ParallelEnv

from team_helper import TeamHelper

class TeamEnv(ParallelEnv):
  def __init__(self, env: ParallelEnv, team_helper: TeamHelper):
    self._env = env
    self._team_helper = team_helper

    self.possible_agents = list(range(team_helper.num_teams))

  def action_space(self, team: int) -> gym.Space:
    return gym.spaces.Dict({
      pos: self._env.action_space(self._team_helper.agent_for_team_and_position[team, pos]) \
        for pos in range(self._team_helper.team_size[team])
    })

  def observation_space(self, team: int) -> gym.Space:
    return gym.spaces.Dict({
      pos: self._env.observation_space(self._team_helper.agent_for_team_and_position[team, pos]) \
        for pos in range(self._team_helper.team_size[team])
    })

  def _group_by_team(self, data: Dict[int, Any]) -> Dict[int, Dict[int, Any]]:
    grouped_data = defaultdict(dict)
    for agent_id, value in data.items():
      team_id, pos = self._team_helper.team_and_position_for_agent[agent_id]
      grouped_data[team_id][pos] = value
    return dict(grouped_data)

  def _team_actions_to_agent_actions(self, team_actions: Dict[int, Dict[int, Any]]) -> Dict[int, Any]:
    agent_actions = {}
    for team_id, team_action in team_actions.items():
      for pos, action in team_action.items():
        agent_id = self._team_helper.agent_for_team_and_position[team_id, pos]
        agent_actions[agent_id] = action
    return agent_actions

  def reset(self, **kwargs) -> Dict[int, Any]:
    gym_obs = self._env.reset(**kwargs)
    return self._group_by_team(gym_obs)

  def step(self, actions: Dict[int, Dict[str, Any]]):
    agent_actions = self._team_actions_to_agent_actions(actions)
    gym_obs, rewards, dones, infos = self._env.step(agent_actions)
    merged_obs = self._group_by_team(gym_obs)
    merged_rewards = {
      tid: sum(vals.values()) for tid, vals in self._group_by_team(rewards).items()
    }
    merged_infos = self._group_by_team(infos)
    merged_dones = {
      tid: all(vals.values()) for tid, vals in self._group_by_team(dones).items()
    }
    return merged_obs, merged_rewards, merged_dones, merged_infos

  ############################################################################
  # PettingZoo API
  ############################################################################

  def render(self, mode='human'):
    return self._env.render(mode)

  @property
  def agents(self) -> List[AgentID]:
    return self.possible_agents

  def close(self):
    return self._env.close()

  def seed(self, seed=None):
    return self._env.seed(seed)

  def state(self):
    return self._env.state()

  @property
  def metadata(self) -> Dict:
    return self._env.metadata
