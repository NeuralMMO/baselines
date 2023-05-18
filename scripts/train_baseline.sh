#!/bin/bash

python -u -O -m tools.train \
--rollout.num_envs=8 \
--rollout.num_buffers=2  \
--rollout.num_steps=128 \
--ppo.bptt_horizon=8 \
--ppo.num_minibatches=16 \
--wandb.entity=daveey \
--wandb.project=nmmo \
--train.opponent_pool=/fsx/home-daveey/experiments/pool.json \
--train.experiments_dir=/fsx/home-daveey/experiments \
--train.num_steps=10000000000 \
--env.num_maps=100 \
--env.team_size=8 \
--env.num_teams=16 \
--env.num_npcs=256 \
--env.num_learners=16 \
"${@}"
