#!/bin/bash
#SBATCH --job-name=eval-grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./eval_logs/%x-%j.out
#SBATCH --err=./eval_logs/%x-%j.err
#SBATCH --environment=vllm071
#SBATCH -A a-a131

set -x -e
ulimit -c 0

source ~/.bashrc
cd /Documents/sink
echo "START TIME: $(date)"

# Load kuma environment variables
# source kuma.env
# CONTAINER_PATH=/work/liac/sink/containers/vllm-openai_v0.7.1.sif



export NUM_GPUS=1 # Set to 8 for 32B and 70B models
export MODEL_ID='Qwen2.5-3B_pretrained-v6-1'
export JOB_ID=0

if [ $JOB_ID = 0 ]; then
    MODEL_PATH=$(grep "^${MODEL_ID}:" model_paths.txt | cut -d':' -f2- | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
else
    MODEL_PATH=/cache/checkpoints/${MODEL_ID}/${JOB_ID}
fi

MODEL_ARGS="model_name=$MODEL_PATH,dtype=bfloat16,max_num_batched_tokens=16384,max_model_length=4096,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=evals/${MODEL_ID}_${JOB_ID}


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
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# export LAUNCHER="conda deactivate;
# cd /Documents/sink;
# pip install hf_transfer gdown levenshtein rdkit wandb exmol;
# pip install -U git+https://github.com/XuanVuNguyen/lighteval;
# pip install lighteval[math];
# pip install -e . --no-deps;
# pip install -U trl==0.14.0 transformers==4.48.2 pandas==2.0.0 vllm==0.7.1;
# pip install -U torch==2.7.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126;
# pip show torch vllm;
# lighteval vllm \"$MODEL_ARGS\" \"custom|iupacsm|0|0\" \
#     --custom-tasks src/open_r1/chem_evaluate/iupacsm.py \
#     --output-dir $OUTPUT_DIR \
# "

# # echo $LAUNCHER
# # echo $OUTPUT_DIR

# # SRUN_ARGS=" \
# #     --wait=60 \
# #     --kill-on-bad-exit=1 \
# #     --jobid $SLURM_JOB_ID \
# # "


# # clear; srun $SRUN_ARGS bash -c "$LAUNCHER" 2>&1

conda deactivate;
cd /Documents/sink;
pip install hf_transfer gdown levenshtein rdkit wandb exmol;
pip install -U git+https://github.com/XuanVuNguyen/lighteval;
pip install lighteval[math];
pip install -e . --no-deps;
pip install -U trl==0.14.0 transformers==4.48.2 pandas==2.0.0 vllm==0.7.1;
pip install -U torch==2.7.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126;
pip show torch vllm;
lighteval vllm \"$MODEL_ARGS\" \"custom|iupacsm|0|0\" \
    --custom-tasks src/open_r1/chem_evaluate/iupacsm.py \
    --output-dir $OUTPUT_DIR \

echo "END TIME: $(date)"
