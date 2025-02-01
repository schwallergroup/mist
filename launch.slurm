#!/bin/bash
#SBATCH --job-name=chem-grpo
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --environment=sink_vllm

set -x -e

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

# Training setup
GPUS_PER_NODE=4
NUM_NODES=$SLURM_NNODES
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

ACCELERATE_CONFIG_FILE=/users/amarulan/Documents/sink/configs/recipe-ds.yaml

cd $HOME/Documents/sink/

    # --model_name_or_path /cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \

export CMD=" \
    /Documents/sink/src/open_r1/grpo.py \
    --model_name_or_path /cache/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 \
    --dataset_name forward_rxn/USPTO \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --max_prompt_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir data/Qwen2.5-7B-GRPO \
    --optim 'adamw_torch_fused' \
    --max_grad_norm 1.0 \
    --lr_scheduler_type 'cosine' \
    --warmup_ratio 0.03 \
    --report_to wandb \
    --run_name grpo-${SLURM_JOB_ID}
    "

export LAUNCHER="conda deactivate;
cd /Documents/sink/;
WANDB_API_KEY=a271772bd2d02dce487419cb4448c81195dda55c \
HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file ${ACCELERATE_CONFIG_FILE}  \
    --num_processes $(($WORLD_SIZE)) \
    --gradient_accumulation_steps 16 \
    --num_machines $NUM_NODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"
