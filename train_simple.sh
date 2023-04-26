#!/bin/bash

source /fsx/home-daveey/miniconda3/etc/profile.d/conda.sh
conda activate nmmo

python -O -m main \
--model.arch=simple \
--env.num_teams=8 \
--env.team_size=1 \
--rollout.num_envs=1 \
--rollout.num_buffers=1  \
--rollout.num_steps=128 \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.experiments_dir=/fsx/home-daveey/experiments \
--train.num_steps=100000000 \
"$@"

