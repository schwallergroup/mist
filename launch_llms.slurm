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

vllm serve /iopsstor/store/cscs/swissai/a05/LIAC/LLM_models/models--AI4Chem--ChemLLM-7B-Chat-1_5-DPO/snapshots/766f19ab06c0237aa3a1643c3152cecea83e8fca/ \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000


echo "END TIME: $(date)"
