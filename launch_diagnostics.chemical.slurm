#!/bin/bash
#SBATCH --job-name=logprobs
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=./diagnostic_logs/%x-%j.out
#SBATCH --err=./diagnostic_logs/%x-%j.err
#SBATCH -A a-a131

set -x -e

source ~/.bashrc
echo "START TIME: $(date)"

MODEL_PATH=/capstor/store/cscs/swissai/a05/LIAC/checkpoints/iupacsm_441486/checkpoint-75
DATA_PATH=/data/sink_diagnostic/smol_instruct_mol_caption.test.corrupted.v2.csv
OUTDIR=./diagnostic_results/Qwen2.5-3B_iupacsm_441486

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NUM_GPUS_FOR_TRAINING=$(($WORLD_SIZE))

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


export CMD="conda deactivate;
cd /Documents/sink;
pip install -U pip;
pip install hf_transfer gdown levenshtein rdkit wandb exmol;
pip install -e . --no-deps;
pip show torch vllm;
python src/open_r1/diagnostic/chemical_competence.py \
    --model $MODEL_PATH \
    --data_path $DATA_PATH \
    --outdir $OUTDIR
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
