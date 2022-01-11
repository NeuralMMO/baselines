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
   NENT                    = 128
   NPOP                    = 1


class CompetitionRound2(scale.Baseline, bases.Medium, nmmo.config.AllGameSystems):

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   @property
   def NENT(self):
      return 8 * len(self.AGENTS)

   NPOP                    = 16
   EVAL_AGENTS             = 8*[baselines.Meander, baselines.Forage, baselines.Combat, nmmo.Agent]
   AGENTS                  = NPOP*[nmmo.Agent]
   TASKS                   = tasks.All

   AGENT_LOADER            = nmmo.config.TeamLoader
   COOPERATIVE             = True
   TEAM_SPIRIT             = 1.0


class CompetitionRound3(scale.Baseline, bases.Large, nmmo.config.AllGameSystems):

   @property
   def SPAWN(self):
      return self.SPAWN_CONCURRENT

   NENT                    = 1024
   NPOP                    = 32
   COOPERATIVE             = True
   TEAM_SPIRIT             = 1.0
   AGENT_LOADER            = nmmo.config.TeamLoader
   TASKS                   = tasks.All
