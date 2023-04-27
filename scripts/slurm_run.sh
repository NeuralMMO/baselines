#!/bin/bash

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
#SBATCH --export=PYTHONUNBUFFERED=1

source /fsx/home-daveey/miniconda3/etc/profile.d/conda.sh && \
conda activate nmmo && \
ulimit -c unlimited && \
ulimit -s unlimited && \
ulimit -a

while true; do
  stdbuf -oL -eL $@

  exit_status=$?

  if [ $exit_status -eq 0 ]; then
    echo "Job completed successfully."
    break
  else
    echo "Job failed with exit status $exit_status. Retrying..."
  fi
done
