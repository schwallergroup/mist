sbatch \
    --nodes=2 \
    --time=06:00:00 \
    launch_kuma.slurm Qwen2.5-3B_pretrained-v4-cot rxnpred_with_tags 456945
