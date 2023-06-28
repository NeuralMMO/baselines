#!/bin/bash

python -u -O -m tools.evaluate \
--model.policy_pool=/fsx/home-daveey/experiments/pool.json \
--env.num_npcs=256 \
--eval.num_rounds=1000000 \
--eval.num_policies=8 \
"${@}"
