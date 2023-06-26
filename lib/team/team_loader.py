
from nmmo.core.agent import Agent
from nmmo.lib.team_helper import TeamHelper
from nmmo.lib import spawn

class TeamLoader(spawn.SequentialLoader):
  def __init__(self, config, team_helper: TeamHelper):
    assert config.PLAYERS == [Agent], \
      "TeamLoader only supports config.PLAYERS == [Agent]"
    super().__init__(config)
    self.team_helper = team_helper

    self.candidate_spawn_pos = \
      spawn.get_team_spawn_positions(config, team_helper.num_teams)
    # print("TeamLoader: candidate_spawn_pos", self.candidate_spawn_pos)

  def get_spawn_position(self, agent_id):
    team_id, _ = self.team_helper.team_and_position_for_agent[agent_id]
    return self.candidate_spawn_pos[team_id]
