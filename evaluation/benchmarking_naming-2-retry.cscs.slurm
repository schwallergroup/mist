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

MODEL=/capstor/store/cscs/swissai/a05/LIAC/checkpoints/iupacsm_441486/checkpoint-75/ # Path to your model checkpoint
DATAPATH=/data/david/final_tasks_prompts/reaction_class_prompts_600k.csv # likely you don't need to change this
OUTDIR=eval_results_500/qwen_pt_sft_rl_iupacsm_441486 # Where do you want to save the evaluation results

conda deactivate;
cd /Documents/sink;
pip install -U pip;
pip install hf_transfer gdown levenshtein rdkit wandb exmol;
pip install -e . --no-deps;
pip show torch vllm;

python evaluation/benchmarking_naming-2-retry.py \
    --model /cache/checkpoints/Qwen2.5-3B_pretrained-v6-1/431710/checkpoint-88 \
    --data_path $DATAPATH \
    --out_dir $OUTDIR \

echo "END TIME: $(date)"
