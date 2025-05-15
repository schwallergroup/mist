sbatch \
    --account='a-a131' \
    --nodes=4 \
    --time=05:00:00 \
    --partition=normal \
    launch_CSCS.slurm Qwen2.5-3B rxnpred 0 qwen_base
