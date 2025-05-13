#!/bin/bash
#SBATCH --job-name=eval-grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./eval_logs/%x-%j.out
#SBATCH --err=./eval_logs/%x-%j.err
#SBATCH --environment=vllm071
#SBATCH -A a-a131

set -x -e
ulimit -c 0

source ~/.bashrc

echo "START TIME: $(date)"

conda deactivate;
cd /Documents/sink;
pip install hf_transfer gdown levenshtein rdkit wandb exmol;
pip install -e . --no-deps;
pip install -U trl==0.14.0 transformers==4.48.2 pandas==2.0.0 vllm==0.7.1;
pip install -U torch==2.5.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126;
pip show torch vllm;
python evaluate.rxnpred.py \
    --model_path "/iopsstor/scratch/cscs/nnguyenx/.cache/checkpoints/Qwen2.5-3B_pretrained-v6-1/431710/checkpoint-84/" \
    --task_mode "tagged" \
    --save_dir "eval_results/rxnpred_431710"

echo "END TIME: $(date)"
