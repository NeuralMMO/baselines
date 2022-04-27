from pdb import set_trace as T

import os

import nmmo

from scripted import baselines

class Base:
   N_TRAIN_MAPS = 256
   N_EVAL_MAPS  = 32

   @property
   def PATH_MAPS(self):
      maps = super().PATH_MAPS
      if self.EVALUATE:
          self.TERRAIN_FLIP_SEED = True
          return os.path.join(maps, 'evaluation')
      return os.path.join(maps, 'training')

   @property
   def NMAPS(self):
      if not self.EVALUATE:
          return self.N_TRAIN_MAPS
      return super().NMAPS

   #Policy specification
   EVAL_AGENTS = [baselines.Meander, baselines.Forage, baselines.Combat, nmmo.Agent]
   AGENTS      = [nmmo.Agent]

   LOG_LEVEL               = 1

class CleanRL(Base, nmmo.config.Medium, nmmo.config.AllGameSystems):
    EVALUATE = False

    EXP_NAME = 'CleanRL'
    SEED     = 1
    TORCH_DETERMINISTIC = True
    CUDA = [0]
    WANDB_PROJECT_NAME = 'cleanRL'
    WANDB_ENTITY = None
    ENV_ID = 'nmmo'
    TOTAL_TIMESTEPS = 500_000_000
    LEARNING_RATE = 5e-5

    @property
    def NUM_ENVS(self):
        return 1 * self.NENT

    @property
    def NUM_EVAL_ENVS(self):
        return 1 * self.NENT

    NUM_CPUS = 2
    NUM_STEPS = 32
    ANNEAL_LR = False
    GAE = True
    GAMMA = 0.99
    GAE_LAMBDA = 1.0
    NUM_MINIBATCHES = 32
    UPDATE_EPOCHS = 1
    NORM_ADV = True
    CLIP_COEF = 0.3
    CLIP_VLOSS = True
    ENT_COEF = 0.0
    VF_COEF = 1.0
    MAX_GRAD_NORM = 0.5
    TARGET_KL = None

    @property
    def BATCH_SIZE(self):
        return int(self.NUM_ENVS * self.NUM_STEPS)

    HIDDEN = 64
    EMBED  = 64

    NENT   = 128

    import tasks
    tasks = tasks.All

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

    FORCE_MAP_GENERATION = False

    NUM_ARGUMENTS = 3


class RLlib:
   '''Base config for RLlib Models

   Extends core Config, which contains environment, evaluation,
   and non-RLlib-specific learning parameters

   Configure NUM_GPUS and NUM_WORKERS for your hardware
   Note that EVALUATION_NUM_WORKERS cores are reserved for evaluation
   and one additional core is reserved for the driver process.
   Therefore set NUM_WORKERS <= cores - EVALUATION_NUM_WORKERS - 1
   '''

   #Run in train/evaluation mode
   EVALUATE     = False
   N_TRAIN_MAPS = 256

   @property
   def MODEL(self):
      return self.__class__.__name__

   @property
   def PATH_MAPS(self):
      maps = super().PATH_MAPS
      if self.EVALUATE:
          self.TERRAIN_FLIP_SEED = True
          return os.path.join(maps, 'evaluation')
      return os.path.join(maps, 'training')

   @property
   def NMAPS(self):
      if not self.EVALUATE:
          return self.N_TRAIN_MAPS
      return super().NMAPS

   @property
   def TRAIN_BATCH_SIZE(self):
      return 64 * 256 * self.NUM_WORKERS

   #Checkpointing. Resume will load the latest trial, e.g. to continue training
   #Restore (overrides resume) will force load a specific checkpoint (e.g. for rendering)
   EXPERIMENT_DIR          = 'experiments'
   RESUME                  = False

   RESTORE                 = True
   RESTORE_ID              = 'Baseline' #Experiment name suffix
   RESTORE_CHECKPOINT      = 1000

   #Policy specification
   EVAL_AGENTS             = [baselines.Meander, baselines.Forage, baselines.Combat, nmmo.Agent]
   AGENTS                  = [nmmo.Agent]
   TASKS                   = []

   #Hardware and debug
   NUM_GPUS_PER_WORKER     = 0
   LOCAL_MODE              = False
   LOG_LEVEL               = 1

   #Training and evaluation settings
   EVALUATION_INTERVAL     = 1
   EVALUATION_PARALLEL     = True
   TRAINING_ITERATIONS     = 1000
   KEEP_CHECKPOINTS_NUM    = 3
   CHECKPOINT_FREQ         = 1
   LSTM_BPTT_HORIZON       = 16
   NUM_SGD_ITER            = 1

   #Model
   SCRIPTED                = None
   N_AGENT_OBS             = 100
   NPOLICIES               = 1
   HIDDEN                  = 64
   EMBED                   = 64

   #Reward
   COOPERATIVE             = False
   TEAM_SPIRIT             = 0.0
