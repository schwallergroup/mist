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

MODEL=/capstor/store/cscs/swissai/a131/jmeng/models/pretrained_sft_qwen2.5_3B/pretrained_sft_qwen2.5_3B_epoch8 # Path to your model checkpoint
DATAPATH=/capstor/store/cscs/swissai/a131/nnguyenx/benchmark_i2s.pkl # likely you don't need to change this
OUTDIR=eval_results/iupac_to_smiles/jeffrey_3B_sft # Where do you want to save the evaluation results

conda deactivate;
cd /Documents/sink;
pip install -U pip;
pip install hf_transfer gdown levenshtein rdkit wandb exmol;
pip install -e . --no-deps;
pip show torch vllm;

python evaluation_scripts/iupac_to_smiles/iupac_to_smiles_benchmark.py \
    --model $MODEL \
    --data_path $DATAPATH \
    --out_dir $OUTDIR \

echo "END TIME: $(date)"
