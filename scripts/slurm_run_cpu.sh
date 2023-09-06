#!/bin/bash

# Example ussage:
#
# sbatch ./scripts/slurm_run.sh scripts/train_baseline.sh \
#   --train.experiment_name=realikun_16x8_0001

#SBATCH --comment=carperai
#SBATCH --partition=cpu128
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --chdir=/fsx/home-daveey/nmmo-baselines/
#SBATCH --output=sbatch/%j.log
#SBATCH --error=sbatch/%j.log
#SBATCH --requeue
#SBATCH --export=PYTHONUNBUFFERED=1

source /fsx/home-daveey/miniconda3/etc/profile.d/conda.sh && \
conda activate nmmo && \
ulimit -c unlimited && \
ulimit -s unlimited && \
ulimit -a

# Extract experiment_name from the arguments
experiment_name=""
args=()
for i in "$@"
do
  case $i in
    --train.experiment_name=*)
    experiment_name="${i#*=}"
    args+=("$i")
    shift
    ;;
    *)
    args+=("$i")
    shift
    ;;
  esac
done

# Create symlink to the log file
if [ ! -z "$experiment_name" ]; then
  logfile="$SLURM_JOB_ID.log"
  symlink="sbatch/${experiment_name}.log"
  if [ -L "$symlink" ]; then
    rm "$symlink"
  fi
  ln -s "$logfile" "$symlink"
fi

while true; do
  stdbuf -oL -eL "${args[@]}"

  exit_status=$?
  echo "Job exited with status $exit_status."

  if [ $exit_status -eq 0 ]; then
    echo "Job completed successfully."
    break
  elif [ $exit_status -eq 101 ]; then
    echo "Job failed due to torch.cuda.OutOfMemoryError."
    break
  else
    echo "Job failed with exit status $exit_status. Retrying..."
  fi
done

echo "Slurm Job completed."
