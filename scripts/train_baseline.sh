#!/bin/bash

python -u -O -m train \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.runs_dir=/fsx/home-daveey/runs/ \
--train.policy_store_dir=/fsx/home-daveey/policies/ \
--train.num_steps=10000000000 \
--env.num_maps=100 \
"${@}"
