class Baseline:
   '''Scale arguments for reproducing baselines

   Requires a 32 core 1 GPU machine
   '''
   NUM_GPUS                = 1
   NUM_WORKERS             = 28
   EVALUATION_NUM_WORKERS  = 3
   EVALUATION_NUM_EPISODES = 3

class Debug:
   '''Scale arguments for debugging

   Requires a 32 core 1 GPU machine
   '''

   NUM_GPUS                = 0
   NUM_WORKERS             = 2
   EVALUATION_NUM_WORKERS  = 1
   EVALUATION_NUM_EPISODES = 1
