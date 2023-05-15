#!/bin/bash

python -u -O -m tools.train \
--rollout.num_envs=4 \
--rollout.num_buffers=4  \
--rollout.num_steps=32 \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.experiments_dir=/fsx/home-daveey/experiments \
--train.num_steps=10000000000 \
--reward.hunger \
--reward.thirst \
--reward.health \
--env.team_size=8 \
--env.num_teams=16 \
--env.num_npcs=256 \
--env.num_learners=16 \
"${@}"
