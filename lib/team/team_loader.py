
from nmmo.core.agent import Agent
from nmmo.lib.team_helper import TeamHelper
from nmmo.lib import spawn

class TeamLoader(spawn.SequentialLoader):
  def __init__(self, config, team_helper: TeamHelper, np_random):
    assert config.PLAYERS == [Agent], \
      "TeamLoader only supports config.PLAYERS == [Agent]"
    super().__init__(config, np_random)
    self.team_helper = team_helper
    self.config = config
    self.np_random = np_random

    self.candidate_spawn_pos = \
      self.get_team_spawn_positions()
    # print("TeamLoader: candidate_spawn_pos", self.candidate_spawn_pos)

  def get_spawn_position(self, agent_id):
    team_id, _ = self.team_helper.team_and_position_for_agent[agent_id]
    return self.candidate_spawn_pos[team_id]

  def get_team_spawn_positions(self):
    '''Generates spawn positions for new teams
    Agents in the same team spawn together in the same tile
    Evenly spaces teams around the square map borders

    Returns:
        list of tuple(int, int):

    position:
        The position (row, col) to spawn the given teams
    '''
    num_teams = len(self.team_helper.teams)
    teams_per_sides = (num_teams + 3) // 4 # 1-4 -> 1, 5-8 -> 2, etc.

    sides = spawn.get_edge_tiles(self.config)
    self.np_random.shuffle(sides)
    for s in sides:
      self.np_random.shuffle(s)

    assert len(sides[0]) >= 4*teams_per_sides, 'Map too small for teams'

    team_spawn_positions = []
    for side in sides:
      for i in range(teams_per_sides):
        idx = int(len(side)*(i+1)/(teams_per_sides + 1))
        team_spawn_positions.append(side[idx])

    return team_spawn_positions
