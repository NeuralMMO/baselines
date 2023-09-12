#!/bin/bash

# Example ussage:
#
# sbatch ./scripts/slurm_run.sh scripts/train_baseline.sh \
#   --run-name=test --wandb-project=nmmo --wandb-entity=kywch

#SBATCH --account=carperai
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#__SBATCH --mem=80G
#SBATCH --chdir=/fsx/proj-nmmo/nmmo-baselines/
#SBATCH --output=sbatch/%j.log
#SBATCH --error=sbatch/%j.log
#SBATCH --requeue
#SBATCH --export=PYTHONUNBUFFERED=1,WANDB_BASE_URL="https://stability.wandb.io",WANDB_DIR=/fsx/proj-nmmo/tmp/wandb,WANDB_CONFIG_DIR=/fsx/proj-nmmo/tmp/wandb

source /fsx/proj-nmmo/venv/bin/activate && \
ulimit -c unlimited && \
ulimit -s unlimited && \
ulimit -a

wandb login --host=https://stability.wandb.io

# Extract run_name from the arguments
run_name=""
args=()
for i in "$@"
do
  case $i in
    --train.run_name=*)
    run_name="${i#*=}"
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
if [ ! -z "$run_name" ]; then
  logfile="$SLURM_JOB_ID.log"
  symlink="sbatch/${run_name}.log"
  if [ -L "$symlink" ]; then
    rm "$symlink"
  fi
  ln -s "$logfile" "$symlink"
fi

max_retries=50
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

    # Killing processes that have the run name in their command line
    run_pids=$(pgrep -f "python.*$run_name")
    if [ "$run_pids" != "" ]; then
      echo "The following processes with '$run_name' will be killed:"
      for pid in $run_pids; do
        echo "Experiment PID $pid: $(ps -p $pid -o cmd=)"
      done
      kill $run_pids  # This kills the processes
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
