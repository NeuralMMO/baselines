#!/bin/bash

python -u -O -m train \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.runs_dir=/fsx/home-daveey/runs/ \
--train.num_steps=10000000000 \
--env.num_maps=100 \
--rollout.num_buffers=2 \
--rollout.num_envs=12 \
--env.team_size=1 \
--env.num_npcs=266 \
--ppo.training_batch_size=128 \
--rollout.batch_size=131072 \
--env.combat_enabled \
"${@}"
