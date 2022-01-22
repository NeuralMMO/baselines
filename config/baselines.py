from pdb import set_trace as T

import nmmo

from config import bases, scale
import tasks


class Medium(scale.Baseline, bases.Medium, nmmo.config.AllGameSystems):
    '''Config for NMMO default environment with concurrent spawns'''
    TASKS                   = tasks.All

    # Load 1000 epoch pretrained model
    RESTORE                 = True
    RESTORE_ID              = '870d'

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT
    pass


class Debug(scale.Debug, bases.Small, nmmo.config.AllGameSystems):
   '''Debug Neural MMO training setting

   A version of the Smallsetting with greatly reduced batch parameters.
   Only intended as a tool for identifying bugs in the model or environment'''

   TASKS                   = tasks.All

   RESTORE                 = False

   TRAINING_ITERATIONS     = 2

   SGD_MINIBATCH_SIZE      = 100
   TRAIN_BATCH_SIZE        = 400
   TRAIN_HORIZON           = 200
   EVALUATION_HORIZON      = 50

   HIDDEN                  = 2
   EMBED                   = 2

