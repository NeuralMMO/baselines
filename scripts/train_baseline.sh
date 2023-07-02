#!/bin/bash

python -u -O -m train \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.experiments_dir=/fsx/home-daveey/experiments \
--train.num_steps=10000000000 \
--env.num_maps=100 \
"${@}"
