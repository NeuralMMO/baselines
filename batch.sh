#!/bin/bash

#SBATCH --comment=carperai
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --chdir=/fsx/home-daveey/nmmo-baselines/
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --requeue

stdbuf -oL -eL "$@"
