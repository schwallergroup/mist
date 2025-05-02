sbatch \
    --nodes=4 \
    --time=08:00:00 \
    launch_kuma.slurm Qwen2.5-3B_pretrained-v6-1 rxnpred 0 tagged
