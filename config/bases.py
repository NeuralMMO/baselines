from pdb import set_trace as T

import os

import nmmo

from scripted import baselines
import tasks

class Base:
   RESPAWN  = True

   PLAYERS = [nmmo.Agent]
   TASKS    = tasks.All


   MAP_N = 256

   # TODO: Check this param name
   PLAYER_N = 128

   HIDDEN   = 64
   EMBED    = 64

   MAP_FORCE_GENERATION = False
