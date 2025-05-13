sbatch \
    --account='a-a131' \
    --nodes=4 \
    --time=10:00:00 \
    --partition=normal \
    launch_CSCS.slurm Qwen2.5-3B_pretrained-v6-1 rxnpred 431710 tagged
