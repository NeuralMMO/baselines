#!/bin/bash

#SBATCH --comment=carperai
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --chdir=/fsx/home-daveey/nmmo-baselines/
#SBATCH --output=sbatch/%j.out
#SBATCH --error=sbatch/%j.error
#SBATCH --requeue

stdbuf -oL -eL "$@"
