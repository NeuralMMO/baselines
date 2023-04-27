#!/bin/bash

source /fsx/home-daveey/miniconda3/etc/profile.d/conda.sh
conda activate nmmo

python -O -m main \
--rollout.num_envs=4 \
--rollout.num_buffers=4  \
--rollout.num_steps=32 \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.experiments_dir=/fsx/home-daveey/experiments \
--train.num_steps=100000000 \
"$@"
