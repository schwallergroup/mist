#!/bin/bash
#SBATCH --job-name=logprobs
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=../sink_logs/%x-%j.out
#SBATCH --err=../sink_logs/%x-%j.err
#SBATCH -A a-a131

set -x -e

source ~/.bashrc
echo "START TIME: $(date)"

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NUM_GPUS_FOR_TRAINING=$(($WORLD_SIZE))


MODEL_ID=$1

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

MODEL=$(grep "^${MODEL_ID}:" model_paths.txt | cut -d':' -f2- | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
MODEL_NAME=$(echo $MODEL | sed -E 's/.*models--(.*)--(.*)\/snapshots.*/\1\/\2/' | sed 's/--/\//')
CONFIG_FILE=recipes/$TASK.yaml

export HF_HOME=/cache/huggingface
export WANDB_PROJECT="r1-${TASK%%.*}"
export WANDB_API_KEY=$(cat wandb_api_key.txt)

export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_TIMEOUT=3600
export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error.json"
export NCCL_P2P_DISABLE=1  # Try this if you're having network issues
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_TIMEOUT=23
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TREE_THRESHOLD=0
export CUDA_LAUNCH_BLOCKING=1

export CMD=" \
    conda deactivate;
    cd /Documents/sink;
    python src/open_r1/diagnostic/smiles_competence.py
"

export NCCL_ASYNC_ERROR_HANDLING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
"

clear; srun --environment=vllm071 bash -c "$CMD" 2>&1

echo "END TIME: $(date)"