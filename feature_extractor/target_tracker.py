class TargetTracker():
  def __init__(self, team_size: int) -> None:
    self.team_size = team_size

    self.target_entity_id = None
    self.target_entity_pop = None

  # pylint: disable=unused-argument
  def reset(self, init_obs):
    self.target_entity_id = [None] * self.team_size
    self.target_entity_pop = [None] * self.team_size

  def update(self, obs):
    pass
