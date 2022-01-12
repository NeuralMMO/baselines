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
   N_TRAIN_MAPS            = 16384
class DomainRandomization256(DomainRandomization):
   N_TRAIN_MAPS            = 256
class DomainRandomization32(DomainRandomization):
   N_TRAIN_MAPS            = 32
class DomainRandomization1(DomainRandomization):
   N_TRAIN_MAPS            = 1

class MagnifyExploration(scale.Baseline, bases.Medium, nmmo.config.Resource, nmmo.config.Progression): pass
class Population4(MagnifyExploration):
   NENT                    = 4
class Population32(MagnifyExploration):
   NENT                    = 32
class Population256(MagnifyExploration):
   NENT                    = 256

class TeamBased(MagnifyExploration, nmmo.config.Combat):
   NENT                    = 128
   NPOP                    = 32
   COOPERATIVE             = True
   TEAM_SPIRIT             = 0.5

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT
