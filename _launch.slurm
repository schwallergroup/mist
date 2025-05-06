#!/bin/bash
#SBATCH --job-name=grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --partition=h100
#SBATCH -A liac

export WANDB_API_KEY=cf870797fb856fe5fd2bbfb69114d65c364c59d9
export CACHE_DIR=/scratch/kuroki/logs
export CONTAINER_PATH=/work/liac/sink/containers/vllm-openai_v0.7.1.sif
export LLM_MODEL_DIR=/work/liac/LLM_models
# コンテナ内にCA証明書がないため、ホストのCA証明書をマウント
# export PROXY_CA=/etc/pki/tls/certs/ca-bundle.crt
export PROXY_CA=/home/kuroki/sink/cacert.pem
export PIP_CACHE_DIR=/tmp/pip-cache

set -x -e

source ~/.bashrc
echo "START TIME: $(date)"

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NUM_GPUS_FOR_TRAINING=$(($WORLD_SIZE - 1))


# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)fz
MASTER_PORT=6000


MODEL_ID=$1
TASK=$2
RESUME_JOB_ID=$3

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
export WANDB_PROJECT="r1-$TASK"

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
# export CUDA_VISIBLE_DEVICES=0,1,2


export CMD=" \
    src/open_r1/run_r1_grpo.py --config $CONFIG_FILE \
    --model_name_or_path=$MODEL \
    --output_dir=/cache/checkpoints/$MODEL_NAME/${RESUME_JOB_ID} \
    --run_name grpo-${SLURM_JOB_ID}-from_${RESUME_JOB_ID}-${MODEL_ID} \
    --base_model_name=$MODEL_NAME \
    "

export REQUESTS_CA_BUNDLE=/certs/cacert.pem
export SSL_CERT_FILE=/certs/cacert.pem

export LAUNCHER="cd /Documents/sink;
apt-get update && apt-get install -y ca-certificates;
update-ca-certificates;
export PIP_CACHE_DIR=/tmp/pip-cache;
pip install --no-cache-dir numpy==1.24.4 pandas==2.0.0;
pip install --no-cache-dir hf_transfer rdkit levenshtein wandb scipy;
pip install --no-cache-dir -e .;
pip install --no-cache-dir trl==0.14.0 transformers==4.48.2 gdown==4.7.1;
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

export WANDBLOG="wandb sync \$(cat logs/grpo-chem-${SLURM_JOB_ID}.out | grep 'saved locally' | awk '{print \$8}')"

export NCCL_ASYNC_ERROR_HANDLING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear;
# srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD; $WANDBLOG" 2>&1
srun apptainer exec --nv \
    --bind /work \
	--mount type=bind,src=${LLM_MODEL_DIR},dst=/LLM_models \
	--mount type=bind,src="$(dirname "$(pwd)")",dst=/Documents \
	--mount type=bind,src=${CACHE_DIR},dst=/cache \
    --mount type=bind,src=${PROXY_CA},dst=/certs/cacert.pem \
	${CONTAINER_PATH} \
	bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD; $WANDBLOG" 2>&1

echo "END TIME: $(date)"
