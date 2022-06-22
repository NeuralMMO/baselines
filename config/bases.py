from pdb import set_trace as T

import os

import nmmo

from scripted import baselines
import tasks


def make_eval_config(config_cls):
    class Eval(config_cls):
        SPECIALIZE = True

        PLAYERS = [
           baselines.Meander, 
           baselines.Fisher, baselines.Herbalist, baselines.Prospector, baselines.Carver, baselines.Alchemist,
           baselines.Melee, baselines.Range, baselines.Mage] + [nmmo.Agent] * 7
  
        NUM_CPUS = 4

        TERRAIN_FLIP_SEED = True
        RESPAWN = False

        @property
        def PATH_MAPS(self):
          return os.path.join(super().PATH_MAPS, 'evaluation')

        MAP_N = 32

    return Eval


class Base:
   @property
   def PATH_MAPS(self):
      return os.path.join(super().PATH_MAPS, 'training')

   RESPAWN  = True

   PLAYERS = [nmmo.Agent]
   TASKS    = tasks.All


   MAP_N = 256

   # TODO: Check this param name
   PLAYER_N = 128

   HIDDEN   = 64
   EMBED    = 64

   MAP_FORCE_GENERATION = False
