sbatch \
    --account='a131' \
    --nodes=4 \
    --time=10:00:00 \
    --partition=normal \
    launch_CSCS.slurm Qwen2.5-7B_pretrained-vj1 rxnpred.7B 0 tagged
