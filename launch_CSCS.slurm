#!/bin/bash
#SBATCH --job-name=grpo-chem
#SBATCH --ntasks-per-node=1
<<<<<<< HEAD
#SBATCH --time=72:00:00
=======
#SBATCH --time=08:00:00
>>>>>>> 796e7588de6a0a92d1c01e810ab00efe6835c1ec
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
<<<<<<< HEAD:launch.slurm
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --partition=h100
#SBATCH -A liac

export WANDB_API_KEY=cf870797fb856fe5fd2bbfb69114d65c364c59d9
<<<<<<< HEAD
export CACHE_DIR=/scratch/kuroki/logs
export CONTAINER_PATH=/work/liac/sink/containers/vllm-openai_v0.7.1.sif
export LLM_MODEL_DIR=/work/liac/LLM_models
# コンテナ内にCA証明書がないため、ホストのCA証明書をマウント
# export PROXY_CA=/etc/pki/tls/certs/ca-bundle.crt
export PROXY_CA=/home/kuroki/sink/cacert.pem
export PIP_CACHE_DIR=/tmp/pip-cache
=======
export CACHE_DIR=./logs
export CONTAINER_PATH=/work/liac/sink/containers/vllm-openai_v0.7.1.sif
export LLM_MODEL_DIR=/work/liac/LLM_models
>>>>>>> 796e7588de6a0a92d1c01e810ab00efe6835c1ec
=======
#SBATCH --output=../sink_logs/%x-%j.out
#SBATCH --err=../sink_logs/%x-%j.err
#SBATCH --environment=vllm071
#SBATCH -A a-a05

# Run like:
# Example 1: sbatch launch.slurm Qwen2.5-0.5B canonmc
# Example 2: sbatch launch.slurm Qwen2.5-0.5B canonmc base
# Argument 1: model id (from model_paths.txt)
# Argument 2: task name (in folder recipes)
# Argument 3 (optional): job_id to continue training from (if none, the default is 0)
# Argument 4 (optional): task mode
# Note: if you would like to use "task mode" without using "job_id", you still need to pass the job_id parameter with the value 0
# Note: task_mode should not be defined in the recipe! It should be used in the sbatch command
>>>>>>> origin/main:launch_CSCS.slurm

set -x -e

source ~/.bashrc
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
<<<<<<< HEAD:launch.slurm
RESUME_JOB_ID=$3
=======
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
echo "MODEL ID: ${MODEL_ID}"
echo "TASK: ${TASK}"
echo "RESUME_JOB_ID: ${RESUME_JOB_ID}"
echo "TASK_MODE: ${TASK_MODE}"

>>>>>>> origin/main:launch_CSCS.slurm

MODEL=$(grep "^${MODEL_ID}:" model_paths.txt | cut -d':' -f2- | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
MODEL_NAME=$(echo $MODEL | sed -E 's/.*models--(.*)--(.*)\/snapshots.*/\1\/\2/' | sed 's/--/\//')

if [[ $RESUME_JOB_ID = 0 ]]
then
	RESUME_JOB_ID=$SLURM_JOB_ID
fi

CONFIG_FILE=recipes/$TASK.yaml

echo "MODEL ID: ${MODEL_ID}"
echo "MODEL PATH: ${MODEL}"
echo "TASK: ${TASK}"
echo "TRAINING ON ${NUM_GPUS_FOR_TRAINING} GPU(S)"

export HF_HOME=/cache/huggingface
<<<<<<< HEAD:launch.slurm
export WANDB_PROJECT="r1-$TASK"
=======
export WANDB_PROJECT="r1-${TASK%%.*}"
export WANDB_API_KEY=$(cat wandb_api_key.txt)
>>>>>>> origin/main:launch_CSCS.slurm

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
<<<<<<< HEAD:launch.slurm
    --output_dir=/cache/checkpoints/$MODEL_NAME/${RESUME_JOB_ID} \
    --run_name grpo-${SLURM_JOB_ID}-from_${RESUME_JOB_ID}-${MODEL_ID} \
=======
    --output_dir=/cache/checkpoints/${MODEL_ID}/${RESUME_JOB_ID} \
    --run_name grpo-${MODEL_ID}${TASK_SUFFIX}${TASK_MODE_SUFFIX}-${SLURM_JOB_ID}_from_${RESUME_JOB_ID} \
>>>>>>> origin/main:launch_CSCS.slurm
    --base_model_name=$MODEL_NAME \
    "

<<<<<<< HEAD:launch.slurm
<<<<<<< HEAD
export REQUESTS_CA_BUNDLE=/certs/cacert.pem
export SSL_CERT_FILE=/certs/cacert.pem

export LAUNCHER="cd /Documents/sink;
apt-get update && apt-get install -y ca-certificates;
update-ca-certificates;

export PIP_CACHE_DIR=/tmp/pip-cache

pip install --no-cache-dir numpy==1.24.4 pandas==2.0.0;
pip install --no-cache-dir hf_transfer rdkit levenshtein wandb;
pip install --no-cache-dir -e .;
pip install --no-cache-dir trl==0.14.0 transformers==4.48.2 gdown==4.7.1;
=======
export LAUNCHER="cd /Documents/sink;
pip install --no-cache-dir hf_transfer rdkit levenshtein wandb;
pip install --no-cache-dir -e .;
pip install --no-cache-dir trl==0.14.0 transformers==4.48.2 pandas==2.0.0 gdown==4.7.1;
>>>>>>> 796e7588de6a0a92d1c01e810ab00efe6835c1ec
=======
export LAUNCHER="conda deactivate;
cd /Documents/sink;
pip install hf_transfer gdown levenshtein;
pip install -e . --no-deps;
>>>>>>> origin/main:launch_CSCS.slurm
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

<<<<<<< HEAD
export WANDBLOG="SSL_CERT_FILE=/certs/cacert.pem wandb sync \$(cat logs/grpo-chem-${SLURM_JOB_ID}.out | grep 'saved locally' | awk '{print \$8}')"
=======
export WANDBLOG="wandb sync \$(cat logs/grpo-chem-${SLURM_JOB_ID}.out | grep 'saved locally' | awk '{print \$8}')"
>>>>>>> 796e7588de6a0a92d1c01e810ab00efe6835c1ec

export NCCL_ASYNC_ERROR_HANDLING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
"

<<<<<<< HEAD:launch.slurm
clear;
# srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD; $WANDBLOG" 2>&1
srun apptainer exec --nv \
	--bind /scratch \
    --bind /work \
	--mount type=bind,src=${LLM_MODEL_DIR},dst=/LLM_models \
	--mount type=bind,src="$(dirname "$(pwd)")",dst=/Documents \
	--mount type=bind,src=${CACHE_DIR},dst=/cache \
<<<<<<< HEAD
    --mount type=bind,src=${PROXY_CA},dst=/certs/cacert.pem \
	${CONTAINER_PATH} \
	bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD; $WANDBLOG" 2>&1

echo "END TIME: $(date)"
=======
	${CONTAINER_PATH} \
	bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD; $WANDBLOG" 2>&1

echo "END TIME: $(date)"
>>>>>>> 796e7588de6a0a92d1c01e810ab00efe6835c1ec
=======
clear; srun $SRUN_ARGS bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD; $WANDBLOG" 2>&1

echo "END TIME: $(date)"
>>>>>>> origin/main:launch_CSCS.slurm
