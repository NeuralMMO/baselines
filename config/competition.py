'''Configs used during the 2021 AICrowd Neural MMO Challenge'''

from pdb import set_trace as T

import nmmo

import tasks
from config import bases, scale
from scripted import baselines


class CompetitionRound1(scale.Baseline, bases.Medium, nmmo.config.AllGameSystems):

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   TASKS                   = tasks.All
   PLAYER_N                = 128
   NPOP                    = 1


class CompetitionRound2(scale.Baseline, bases.Medium, nmmo.config.AllGameSystems):

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   @property
   def PLAYER_N(self):
      return 8 * len(self.PLAYERS)

   NPOP                    = 16
   EVAL_PLAYERS            = 8*[baselines.Meander, baselines.Forage, baselines.Range, nmmo.Agent]
   AGENTS                  = NPOP*[nmmo.Agent]
   TASKS                   = tasks.All

   PLAYER_LOADER           = nmmo.spawn.TeamLoader
   COOPERATIVE             = True
   TEAM_SPIRIT             = 1.0


class CompetitionRound3(scale.Baseline, bases.Large, nmmo.config.AllGameSystems):

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   PLAYER_N                = 1024
   NPOP                    = 32
   COOPERATIVE             = True
   TEAM_SPIRIT             = 1.0
   PLAYER_LOADER           = nmmo.spawn.TeamLoader
   TASKS                   = tasks.All
