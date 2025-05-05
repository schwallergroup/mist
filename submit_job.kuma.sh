sbatch \
    --nodes=4 \
    --time=00:30:00 \
    launch_kuma.slurm Qwen2.5-3B_pretrained-v6-1 rxnpred.long_fast 0 tagged
