from pdb import set_trace as T

import nmmo

from config.bases import Base, make_eval_config


class Train(Base, nmmo.config.Medium, nmmo.config.AllGameSystems):
    @property
    def NUM_ENVS(self):
        return self.NUM_CPUS * self.NENT

    @property
    def BATCH_SIZE(self):
        return int(self.NUM_ENVS * self.NUM_STEPS)

    # Hardcoded for now: number of args to predict
    NUM_ARGUMENTS = 3
    COMBAT = True

    EXP_NAME                = 'CleanRL'
    ENV_ID                  = 'nmmo'

    WANDB_PROJECT_NAME      = 'cleanRL'
    WANDB_ENTITY            = None

    TORCH_DETERMINISTIC     = True
    SEED                    = 1

    TOTAL_TIMESTEPS         = 1_000_000_000
    CUDA                    = [0]
    NUM_CPUS                = 32

    HORIZON                 = 512
    NUM_STEPS               = 512
    NUM_MINIBATCHES         = 528
    LEARNING_RATE           = 5e-5
    UPDATE_EPOCHS           = 1

    ANNEAL_LR               = False
    GAE                     = True
    NORM_ADV                = True
    CLIP_VLOSS              = True
    TARGET_KL               = None

    GAE_LAMBDA              = 1.0
    GAMMA                   = 0.99
    CLIP_COEF               = 0.3
    ENT_COEF                = 0.0
    VF_COEF                 = 1.0
    MAX_GRAD_NORM           = 0.5

class TrainForage(Train):
    COMBAT = False

Eval = make_eval_config(Train)


class Debug(Train):
    HORIZON                 = 6

    NUM_CPUS                = 2
    NUM_STEPS               = 128
    NUM_MINIBATCHES         = 128
    NUM_STEPS               = 1
    CUDA                    = {}

DebugEval = make_eval_config(Debug)
