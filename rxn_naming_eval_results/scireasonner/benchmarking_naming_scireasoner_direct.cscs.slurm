#!/bin/bash
#SBATCH --job-name=i2s-7b-retry
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

MODEL=/capstor/store/cscs/swissai/a131/GRPO_models/iupacsm_7B_613353/checkpoint-125  # Path to your model checkpoint
DATAPATH=/capstor/store/cscs/swissai/a131/dsegura/reaction_class_prompts_600k.csv # likely you don't need to change this
OUTDIR=/iopsstor/scratch/cscs/davidsegura/evaluation-mist/sink/rxn_naming_eval_results/vu_7B_GRPO_I2S # Where do you want to save the evaluation results

conda deactivate;
cd /Documents/sink;
pip install -U pip;
pip install hf_transfer gdown levenshtein rdkit wandb exmol datasets;
pip install --upgrade transformers
pip install -e . --no-deps;
pip show torch vllm;

python /iopsstor/scratch/cscs/davidsegura/git/sink/rxn_naming_eval_results/scireasonner/eval_scireasonner_direct.py \

echo "END TIME: $(date)"
