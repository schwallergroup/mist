#!/bin/bash
#SBATCH --job-name=logprobs
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --err=slurm_logs/%x-%j.err
#SBATCH -A a-a131

set -x -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/cluster/common.sh"

source ~/.bashrc
cd "${MIST_REPO_ROOT}"
echo "START TIME: $(date)"

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))

MODEL_ID=$1
DATA_PATH=${2:-"${MIST_DATA_DIR}/CRLLM-PubChem-compounds1M.csv"}
NUM_ROWS=${3:-10000}
OUTPUT_DIR=${4:-"${MIST_OUTPUT_DIR}/scs/${MODEL_ID}-${SLURM_JOB_ID}"}
TP_SIZE=${5:-${WORLD_SIZE}}

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

MODEL=$(mist_resolve_model_path "${MODEL_ID}")
export HF_HOME="${MIST_HF_HOME}"

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
    cd ${MIST_REPO_ROOT};
    PYTHONPATH=src python src/open_r1/diagnostic/smiles_competence.py \
      --model '${MODEL}' \
      --data-path '${DATA_PATH}' \
      --output-dir '${OUTPUT_DIR}' \
      --num-rows '${NUM_ROWS}' \
      --tensor-parallel-size '${TP_SIZE}'
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
