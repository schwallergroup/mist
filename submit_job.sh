sbatch \
    --nodes=2 \
    --time=01:00:00 \
    launch_kuma.slurm Qwen2.5-3B_pretrained-v5 rxnpred 0 fg_tagged
