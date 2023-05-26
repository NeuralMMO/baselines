import nmmo

class NmmoConfig(
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
  def __init__(
      self,
      num_teams=16, team_size=8, num_npcs=256,
      num_maps=5,
      maps_path=None,
      max_episode_length=1024,
      death_fog_tick=None,
      ):

    super().__init__()

    self.PROVIDE_ACTION_TARGETS = True
    self.MAP_FORCE_GENERATION = False
    self.PLAYER_N = num_teams * team_size
    self.NPC_N = num_npcs
    self.HORIZON = max_episode_length
    self.MAP_N = num_maps
    self.PLAYER_DEATH_FOG = death_fog_tick
    if maps_path is not None:
      self.MAP_PATH = maps_path
