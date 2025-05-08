sbatch \
    --nodes=4 \
    --time=10:00:00 \
    --partition=normal \
    launch_CSCS.slurm Qwen2.5-3B_pretrained-v6-1 iupacsm.long 0 tagged
