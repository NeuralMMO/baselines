from typing import List


class TeamHelper():
  def __init__(self, teams: List[List[int]]):
    self.teams = teams
    self.num_teams = len(teams)
    self.team_size = {}
    self.team_and_position_for_agent = {}
    self.position_for_agent = {}
    self.agent_for_team_and_position = {}

    for team_id, team in enumerate(teams):
      self.team_size[team_id] = len(team)
      for position, agent_id in enumerate(team):
        self.team_and_position_for_agent[agent_id] = (team_id, position)
        self.agent_for_team_and_position[team_id, position] = agent_id

