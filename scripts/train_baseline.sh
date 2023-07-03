#!/bin/bash

python -u -O -m train \
--rollout.num_envs=12 \
--rollout.num_buffers=2  \
--ppo.bptt_horizon=8 \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.experiments_dir=/fsx/home-daveey/experiments \
--train.num_steps=10000000000 \
--env.num_maps=100 \
--env.team_size=1 \
--env.num_teams=128 \
--env.num_npcs=256 \
--reward.environment \
--ppo.training_batch_size=128 \
"${@}"
