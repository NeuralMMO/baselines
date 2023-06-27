#!/bin/bash

# Example ussage:
#
# sbatch ./scripts/slurm_run.sh scripts/train_baseline.sh \
#   --train.experiment_name=realikun_16x8_0001

#SBATCH --comment=carperai
#SBATCH --partition=g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=80G
#SBATCH --chdir=/fsx/home-daveey/nmmo-baselines/
#SBATCH --output=sbatch/%j.log
#SBATCH --error=sbatch/%j.log
#SBATCH --requeue
#SBATCH --export=PYTHONUNBUFFERED=1,WANDB_DIR=/fsx/home-daveey/tmp/wandb,WANDB_CONFIG_DIR=/fsx/home-daveey/tmp/wandb

source /fsx/home-daveey/miniconda/etc/profile.d/conda.sh && \
conda activate nmmo && \
ulimit -c unlimited && \
ulimit -s unlimited && \
ulimit -a

wandb login --host=https://stability.wandb.io

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

max_retries=5
retry_count=0

while true; do
  stdbuf -oL -eL "${args[@]}"

  exit_status=$?
  echo "Job exited with status $exit_status."

  if [ $exit_status -eq 0 ]; then
    echo "Job completed successfully."
    break
  elif [ $exit_status -eq 101 ]; then
    echo "Job failed due to torch.cuda.OutOfMemoryError."
  elif [ $exit_status -eq 137 ]; then
    echo "Job failed due to OOM. Killing child processes..."

    # Killing child processes
    child_pids=$(pgrep -P $$)  # This fetches all child processes of the current process
    if [ "$child_pids" != "" ]; then
      echo "The following child processes will be killed:"
      for pid in $child_pids; do
        echo "Child PID $pid: $(ps -p $pid -o cmd=)"
      done
      kill $child_pids  # This kills the child processes
    fi

    # Killing processes that have the experiment name in their command line
    experiment_pids=$(pgrep -f "$experiment_name")
    if [ "$experiment_pids" != "" ]; then
      echo "The following processes with '$experiment_name' will be killed:"
      for pid in $experiment_pids; do
        echo "Experiment PID $pid: $(ps -p $pid -o cmd=)"
      done
      kill $experiment_pids  # This kills the processes
    fi
  elif [ $exit_status -eq 143 ]; then
    echo "Killing Zombie processes..."
    pids=$(pgrep -P $$)
    for pid in $pids; do
      if [ $(ps -o stat= -p $pid) == "Z" ]; then
        kill -9 $pid
        echo "Killed zombie process $pid"
      fi
    done
  fi
  retry_count=$((retry_count + 1))
  if [ $retry_count -gt $max_retries ]; then
    echo "Job failed with exit status $exit_status. Maximum retries exceeded. Exiting..."
    break
  fi
  echo "Job failed with exit status $exit_status. Retrying in 10 seconds..."
  sleep 10
done

echo "Slurm Job completed."
