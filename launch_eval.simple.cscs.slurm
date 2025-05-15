#!/bin/bash
#SBATCH --job-name=eval-grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
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

MODEL=/cache/checkpoints/Qwen2.5-3B/441900/checkpoint-48/ # Path to your model checkpoint
THINK_MODE=qwen_base # "qwen_base" for post-RL and base Qwen models. "tagged" for any model produced by Shai. This is equivalent to 'task_mode` params in `src/open_r1/tasks/reactions/forward.py`, which affects the prompt the model is given. Used to evaluate the model with reasoning, i.e system 2
NO_THINK_MODE=qwen_base_no_thinking # "qwen_base_no_thinking" for post-RL and base Qwen models. "tagged_no_thinking" for any model produced by Shai. This is equivalent to 'task_mode` params in `src/open_r1/tasks/reactions/forward.py`, which affects the prompt the model is given. Used to evaluate the model without thinking, i.e system 1
DATAPATH=/data/USPTO/USPTO_480k_clean_no_sft # likely you don't need to change this
OUTDIR=eval_results_500/Qwen2.5-3B_rf # Where do you want to save the evaluation results

conda deactivate;
cd /Documents/sink;
pip install -U pip;
pip install hf_transfer gdown levenshtein rdkit wandb exmol;
pip install -e . --no-deps;
pip show torch vllm;

python evaluate.rxnpred.py \
    --model $MODEL \
    --task_mode $THINK_MODE \
    --datapath $DATAPATH \
    --maxlength 8192 \
    --n_samples 1 \
    --temperature 0.01 \
    --save_dir ${OUTDIR}/think_at1

python analyze_rxnpred_output.py \
    --input_txt ${OUTDIR}/think_at1/output.txt \
    --datapath $DATAPATH \
    --task_mode $THINK_MODE

python evaluate.rxnpred.py \
    --model $MODEL \
    --task_mode $THINK_MODE \
    --datapath $DATAPATH \
    --maxlength 8192 \
    --n_samples 5 \
    --temperature 0.01 \
    --save_dir ${OUTDIR}/think_at5

python analyze_rxnpred_output.py \
    --input_txt ${OUTDIR}/think_at5/output.txt \
    --datapath $DATAPATH \
    --task_mode $THINK_MODE

python evaluate.rxnpred.py \
    --model $MODEL \
    --task_mode $NO_THINK_MODE \
    --datapath $DATAPATH \
    --maxlength 8192 \
    --n_samples 1 \
    --temperature 0.01 \
    --save_dir ${OUTDIR}/no_think_at1

python analyze_rxnpred_output.py \
    --input_txt ${OUTDIR}/no_think_at1/output.txt \
    --datapath $DATAPATH \
    --task_mode $NO_THINK_MODE

python evaluate.rxnpred.py \
    --model $MODEL \
    --task_mode $NO_THINK_MODE \
    --datapath $DATAPATH \
    --maxlength 8192 \
    --n_samples 5 \
    --temperature 0.01 \
    --save_dir ${OUTDIR}/no_think_at5

python analyze_rxnpred_output.py \
    --input_txt ${OUTDIR}/no_think_at5/output.txt \
    --datapath $DATAPATH \
    --task_mode $NO_THINK_MODE

echo "END TIME: $(date)"
