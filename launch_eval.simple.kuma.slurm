#!/bin/bash
#SBATCH --job-name=eval-grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --output=./eval_logs/%x-%j.out
#SBATCH --err=./eval_logs/%x-%j.err
#SBATCH --partition=h100
#SBATCH -A liac

set -x -e

# Load kuma environment variables
source kuma.env
CONTAINER_PATH=/work/liac/sink/containers/vllm-openai_v0.7.1.sif

source ~/.bashrc
cd ${KUMA_FOLDER_DOCUMENTS}/sink
echo "START TIME: $(date)"

# export NUM_GPUS=1 # Set to 8 for 32B and 70B models
# export MODEL_ID='Qwen2.5-3B_pretrained-v6-1'
# export JOB_ID=0

# if [ $JOB_ID = 0 ]; then
#     MODEL_PATH=$(grep "^${MODEL_ID}:" model_paths.txt | cut -d':' -f2- | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
# else
#     MODEL_PATH=/cache/checkpoints/${MODEL_ID}/${JOB_ID}
# fi

# MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
# # ,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}

# OUTPUT_DIR=eval_results/${MODEL_ID}_${JOB_ID}


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
export SSL_CERT_FILE="cacert.pem"

# Refer to file kuma_requirements.txt to see what's inside the environment env/
export LAUNCHER="cd /Documents/sink;
. env/bin/activate;
pip show torch vllm;
nvcc --version;
python evaluate.rxnpred.py \
    --model /LLM_models/Qwen2.5-3B_pretrained-v6-1 \
    --task_mode tagged_no_thinking \
    --think  \
    --save_dir eval_results/rxnpred_ShaiV6 \
"
# /scratch/nxnguyen/sink/models/rxnpred/431710/checkpoint-84/

APPTAINER_ARGS=" \
    --nv \
    --bind /scratch \
    --bind /home \
    --bind /work \
    --mount type=bind,src=${KUMA_FOLDER_LLM_MODELS},dst=/LLM_models \
    --mount type=bind,src=${KUMA_FOLDER_DATA},dst=/data \
    --mount type=bind,src=${KUMA_FOLDER_DOCUMENTS},dst=/Documents \
    --mount type=bind,src=${KUMA_FOLDER_CACHE},dst=/cache \
    ${CONTAINER_PATH} \
"


echo "Running lighteval script ..."
echo "Eval results will be saved to $OUTPUT_DIR"

clear; apptainer exec $APPTAINER_ARGS bash -c "$LAUNCHER" 2>&1

# cd ${KUMA_FOLDER_DOCUMENTS}/sink
# . env/bin/activate;
# pip show torch vllm;
# apptainer exec $APPTAINER_ARGS lighteval vllm "$MODEL_ARGS" "custom|iupacsm|0|0" \
#     --custom-tasks src/open_r1/chem_evaluate/iupacsm.py \
#     --output-dir $OUTPUT_DIR \
#     --save-details


echo "END TIME: $(date)"
