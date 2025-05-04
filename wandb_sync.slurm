#!/bin/bash
#SBATCH --job-name=grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=../sink_wandb_logs/%x-%j.out
#SBATCH --err=../sink_wandb_logs/%x-%j.err
#SBATCH --environment=vllm071
#SBATCH -A a-a05
#SBATCH --partition=debug

set -x -e
ulimit -c 0  # Avoid core dumps

source ~/.bashrc
cd /Documents/sink
echo "START TIME: $(date)"

# Export wandb API key
export WANDB_API_KEY=$(cat wandb_api_key.txt)

# Read SLURM JOB ID to process (argument 1)
SLURM_JOB_ID_TO_SYNC=$1
SLURM_LOGS_FILEPATH="../sink_logs/grpo-chem-${SLURM_JOB_ID_TO_SYNC}.out"

# Setup wandb entity
LOGGING_WANDB_ENTITY="liac"

# Read wandb project
#LOGGING_WANDB_PROJECT=$(cat ${SLURM_LOGS_FILEPATH} | grep 'View project at https://wandb.ai/liac/' | awk '{print $6}' | head -n1 | sed 's|.*/||' | xargs)
LOGGING_WANDB_PROJECT="r1-wandb_test"

# Read all wandb local folders
LOGGING_WANDB_FOLDERS=$(cat ${SLURM_LOGS_FILEPATH} | grep 'saved locally' | awk '{print $8}')
# Convert LOGGING_WANDB_FOLDERS string to an array (split by \n)
readarray -t LOGGING_WANDB_FOLDERS_ARRAY <<< "$LOGGING_WANDB_FOLDERS"
# Iterate over the array
for line in "${LOGGING_WANDB_FOLDERS_ARRAY[@]}"; do
  LOGGING_WANDB_FOLDER=$(echo $line | xargs) # xargs strips the string
  # If LOGGING_WANDB_FOLDER is not an empty string, log to wandb
  if [ -n "$LOGGING_WANDB_FOLDER" ]; then
    LOGGING_WANDB_RUN_ID=$(echo $LOGGING_WANDB_FOLDER | sed 's|.*-||' | xargs)
    # Logging to wandb
    echo "Logging ${LOGGING_WANDB_ENTITY}/${LOGGING_WANDB_PROJECT}/${LOGGING_WANDB_RUN_ID} to wandb (from SLURM job ${SLURM_JOB_ID_TO_SYNC})..."
    wandb sync \
      --include-online \
      --include-synced \
      --append \
      --entity $LOGGING_WANDB_ENTITY \
      --project $LOGGING_WANDB_PROJECT \
      --id $LOGGING_WANDB_RUN_ID \
      ${LOGGING_WANDB_FOLDER}
  fi
done

echo "END TIME: $(date)"