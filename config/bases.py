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

   AGENTS   = [nmmo.Agent]
   TASKS    = tasks.All

   NMAPS    = 256
   NENT     = 128

   HIDDEN   = 64
   EMBED    = 64


def make_eval_config(config_cls):
    class Eval(config_cls):
        AGENTS = [baselines.Meander, baselines.Forage, baselines.Combat, nmmo.Agent]

        NUM_CPUS = 2

        TERRAIN_FLIP_SEED = True

        @property
        def PATH_MAPS(self):
          return os.path.join(super().PATH_MAPS, 'evaluation')

        NMAPS = 32

    return Eval
