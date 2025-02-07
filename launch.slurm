#!/bin/bash
#SBATCH --job-name=grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --environment=vllm071
#SBATCH -A a-a05

# Run like:
# sbatch launch.slurm recipes/grpo.yaml

set -x -e

source ~/.bashrc
echo "START TIME: $(date)"

CONFIG_FILE=$1

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NUM_GPUS_FOR_TRAINING=$(($WORLD_SIZE - 1))


# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

MODEL=/cache/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
MODEL=/cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
MODEL=/cache/huggingface/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
MODEL=/cache/huggingface/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
#MODEL=/cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987

MODEL_NAME=$(echo $MODEL | sed -E 's/.*models--(.*)--(.*)\/snapshots.*/\1\/\2/' | sed 's/--/\//')
export HF_HOME=/cache/huggingface
export WANDB_PROJECT=r1
export WANDB_API_KEY=a271772bd2d02dce487419cb4448c81195dda55c

export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_TIMEOUT=3600
export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error.json"
export NCCL_P2P_DISABLE=1  # Try this if you're having network issues
export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_TIMEOUT=23
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TREE_THRESHOLD=0
export CUDA_LAUNCH_BLOCKING=1

export CMD=" \
    src/open_r1/run_r1_grpo.py --config $CONFIG_FILE \
    --model_name_or_path=$MODEL \
    --output_dir=/cache/checkpoints/$MODEL_NAME/${SLURM_JOB_ID} \
    --dataset_id_or_path=/cache/data/ \
    --run_name grpo-${SLURM_JOB_ID} \
    "

export LAUNCHER="conda deactivate;
pip install hf_transfer;
cd /Documents/sink/;
HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
   --config_file configs/deepspeed_zero3.yaml \
   --num_machines $NUM_NODES \
   --num_processes $NUM_GPUS_FOR_TRAINING \
   --main_process_ip $MASTER_ADDR \
   --main_process_port $MASTER_PORT \
   --machine_rank \$SLURM_NODEID \
   --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
   --max_restarts 1 \
   --role \$(hostname -s): \
   --tee 3 \
   "

export NCCL_ASYNC_ERROR_HANDLING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"
