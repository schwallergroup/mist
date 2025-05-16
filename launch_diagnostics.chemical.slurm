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

MODEL_PATH=/LLM_models/Qwen2.5-3B_pretrained-v6-1
DATA_PATH=/data/sink_diagnostic/smol_instruct_mol_caption.test.corrupted.csv
OUTDIR=./diagnostic_results/Qwen2.5-3B_pretrained-v6-1

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NUM_GPUS_FOR_TRAINING=$(($WORLD_SIZE))


export CMD="conda deactivate;
cd /Documents/sink;
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