#!/bin/bash
#SBATCH --job-name=eval-grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./eval_logs/%x-%j.out
#SBATCH --err=./eval_logs/%x-%j.err
#SBATCH --environment=vllm071
#SBATCH -A a131

set -x -e
ulimit -c 0

source ~/.bashrc

echo "START TIME: $(date)"

conda deactivate;
cd /Documents/sink;
pip install -U pip;
pip install hf_transfer gdown levenshtein rdkit wandb exmol;
pip install -e . --no-deps;
pip show torch vllm;

vllm serve /capstor/store/cscs/swissai/a131/jmeng/models/pretrained_sft_qwen2.5_7B/pretrained_sft_qwen2.5_7B_epoch8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000


echo "END TIME: $(date)"
