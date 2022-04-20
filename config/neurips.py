from pdb import set_trace as T
import os

import nmmo

from config import bases, scale
import tasks


# Legacy naming -- medium config intended
class SmallMultimodalSkills(scale.Baseline, bases.Medium, nmmo.config.AllGameSystems):
   TASKS                   = tasks.All

class LargeMultimodalSkills(scale.Baseline, bases.Large, nmmo.config.AllGameSystems):
   TASKS                   = tasks.All

class DomainRandomization(scale.Baseline, bases.Medium, nmmo.config.AllGameSystems): pass
class DomainRandomization16384(DomainRandomization):
   MAP_N_TRAIN             = 16384
class DomainRandomization256(DomainRandomization):
   MAP_N_TRAIN             = 256
class DomainRandomization32(DomainRandomization):
   MAP_N_TRAIN             = 32
class DomainRandomization1(DomainRandomization):
   MAP_N_TRAIN             = 1

class MagnifyExploration(scale.Baseline, bases.Medium, nmmo.config.Resource, nmmo.config.Progression): pass
class Population4(MagnifyExploration):
   PLAYER_N                = 4
class Population32(MagnifyExploration):
   PLAYER_N                = 32
class Population256(MagnifyExploration):
   PLAYER_N                = 256

class TeamBased(MagnifyExploration, nmmo.config.Combat):
   PLAYER_N                = 128
   NPOP                    = 32
   COOPERATIVE             = True
   TEAM_SPIRIT             = 0.5

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT
