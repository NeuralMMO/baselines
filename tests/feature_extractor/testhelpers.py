import unittest

import nmmo
from nmmo.systems import item as Item
from nmmo.core.realm import Realm

from scripted import baselines

from team_helper import TeamHelper

class FeatureTestConfig(nmmo.config.Medium, nmmo.config.AllGameSystems):
  RENDER = False
  SPECIALIZE = True
  # config.Medium's PLAYER_N = 128
  # make 16 teams, so that there are 8 agents per team
  PLAYERS = [
    baselines.Fisher, baselines.Herbalist, baselines.Prospector, baselines.Carver,
    baselines.Alchemist, baselines.Melee, baselines.Range, baselines.Mage,
    baselines.Fisher, baselines.Herbalist, baselines.Prospector, baselines.Carver,
    baselines.Alchemist, baselines.Melee, baselines.Range, baselines.Mage]


class FeaturizerTestTemplate(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = FeatureTestConfig()

    # pylint: disable=invalid-name
    # CHECK ME: this provides action targets, at the cost of performance
    #   We may want to revisit this
    cls.config.PROVIDE_ACTION_TARGETS = True

    # default: 16 teams x 8 players
    cls.num_team = len(cls.config.PLAYERS)
    cls.team_size = int(cls.config.PLAYER_N / cls.num_team)

    # match the team definition to the default nmmo
    teams = {team_id: [cls.num_team*j+team_id+1 for j in range(cls.team_size)]
              for team_id in range(cls.num_team)}
    cls.team_helper = TeamHelper(teams)

  def _filter_obs(self, obs, team_id):
    flt_obs = {}
    for ent_id, ent_obs in obs.items():
      if ent_id in self.team_helper.teams[team_id]:
        flt_obs[ent_id] = ent_obs

    return flt_obs


def provide_item(realm: Realm, ent_id: int,
                 item: Item, level: int,
                 quantity: int=1,
                 list_price=0):
  for _ in range(quantity):
    item_inst = item(realm, level=level)
    realm.players[ent_id].inventory.receive(item_inst)
    if list_price > 0:
      item_inst.listed_price.update(list_price)
