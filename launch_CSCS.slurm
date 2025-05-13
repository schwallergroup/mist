#!/bin/bash
#SBATCH --job-name=grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --output=../sink_logs/%x-%j.out
#SBATCH --err=../sink_logs/%x-%j.err
#SBATCH --environment=vllm071
#SBATCH -A a-a131

# Run like:
# Example 1: sbatch launch.slurm Qwen2.5-0.5B canonmc
# Example 2: sbatch launch.slurm Qwen2.5-0.5B canonmc base
# Argument 1: model id (from model_paths.txt)
# Argument 2: task name (in folder recipes)
# Argument 3 (optional): job_id to continue training from (if none, the default is 0)
# Argument 4 (optional): task mode
# Argument 5 (optional): sampling_params config
# Note: if you would like to use "task mode" without using "job_id" for example, you still need to pass the job_id parameter with the value 0
# Note: task_mode should not be defined in the recipe! It should be used in the sbatch command

set -x -e

source ~/.bashrc
cd /Documents/sink
echo "START TIME: $(date)"

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NUM_GPUS_FOR_TRAINING=$(($WORLD_SIZE - 1))


# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

MODEL_ID=$1
TASK=$2
if [[ "$TASK" == *.* ]]; then
    TASK_SUFFIX="_${TASK#*.}"
else
    TASK_SUFFIX=""
fi
RESUME_JOB_ID="${3:-0}"
if [ $RESUME_JOB_ID = 0 ]; then
    RESUME_JOB_ID=$SLURM_JOB_ID
fi
TASK_MODE="${4:-base}"
if [ "$TASK_MODE" != "base" ]; then
    TASK_MODE_SUFFIX="_$TASK_MODE"
else
    TASK_MODE_SUFFIX=""
fi
SAMPLING_PARAMS_CONFIG_NAME="${5:-default}"
echo "MODEL ID: ${MODEL_ID}"
echo "TASK: ${TASK}"
echo "RESUME_JOB_ID: ${RESUME_JOB_ID}"
echo "TASK_MODE: ${TASK_MODE}"
echo "SAMPLING_PARAMS_CONFIG_NAME: ${SAMPLING_PARAMS_CONFIG_NAME}"


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
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_TIMEOUT=23
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TREE_THRESHOLD=0
export CUDA_LAUNCH_BLOCKING=1


export CMD=" \
    src/open_r1/run_r1_grpo.py --config $CONFIG_FILE \
    --model_name_or_path=$MODEL \
    --output_dir=/cache/checkpoints/${MODEL_ID}/${RESUME_JOB_ID} \
    --run_name grpo-${MODEL_ID}${TASK_SUFFIX}${TASK_MODE_SUFFIX}-${SLURM_JOB_ID}_from_${RESUME_JOB_ID} \
    --base_model_name=$MODEL_NAME \
    --base_model_id=$MODEL_ID \
    --task_recipe=${TASK}.yaml \
    --task_recipe_suffix=$TASK_SUFFIX \
    --task_mode=$TASK_MODE \
    --slurm_job_id=$SLURM_JOB_ID \
    --slurm_resume_job_id=$RESUME_JOB_ID \
    --sampling_params_config_name=$SAMPLING_PARAMS_CONFIG_NAME \
"

export LAUNCHER="conda deactivate;
cd /Documents/sink;
pip install hf_transfer gdown levenshtein smact pymatgen datasets;
pip install -e . --no-deps;
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

export WANDBLOG="wandb sync \$(cat ../sink_logs/grpo-chem-${SLURM_JOB_ID}.out | grep 'saved locally' | awk '{print \$8}')"

export NCCL_ASYNC_ERROR_HANDLING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
"

clear; srun $SRUN_ARGS bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD; $WANDBLOG" 2>&1

echo "END TIME: $(date)"