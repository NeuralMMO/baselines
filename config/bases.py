from pdb import set_trace as T

import os

import nmmo

from scripted import baselines
import tasks


class Base:
   @property
   def PATH_MAPS(self):
      return os.path.join(super().PATH_MAPS, 'training')

   @property
   def SPAWN(self):
       return self.SPAWN_CONCURRENT

   PLAYERS  = [nmmo.Agent]
   TASKS    = tasks.All

   MAP_N    = 256
   PLAYER_N = 128

   HIDDEN   = 64
   EMBED    = 64


def make_eval_config(config_cls):
    class Eval(config_cls):
        PLAYERS = [baselines.Meander,
                   baselines.Fisher, baselines.Herbalist, baselines.Prospector,
                   baselines.Carver, baselines.Alchemist,
                   baselines.Melee, baselines.Range, baselines.Mage]

        NUM_CPUS          = 1
        MAP_N             = 32

        TERRAIN_FLIP_SEED = True
        SPECIALIZE        = True

        @property
        def PATH_MAPS(self):
          return os.path.join(super().PATH_MAPS, 'evaluation')


    return Eval
