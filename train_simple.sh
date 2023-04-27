#!/bin/bash

# Usage: sbatch train_simple.sh experiment_name --arg1=value1 --arg2=value2 ...

job_name=$2
echo $job_name

#SBATCH --comment=carperai
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=40G
#SBATCH --chdir=/fsx/home-daveey/nmmo-baselines/
#SBATCH --output=sbatch/%j.out
#SBATCH --error=sbatch/%j.error
#SBATCH --requeue
#SBATCH --job-name="$job_name"

source /fsx/home-daveey/miniconda3/etc/profile.d/conda.sh && \
conda activate nmmo && \
stdbuf -oL -eL python -O -m main \
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
--experiment_name="$2" \
"${@:3}"
