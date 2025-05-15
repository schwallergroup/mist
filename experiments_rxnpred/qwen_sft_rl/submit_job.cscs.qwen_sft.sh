sbatch \
    --account='a-a131' \
    --nodes=4 \
    --time=10:00:00 \
    --partition=normal \
    launch_CSCS.slurm qwen_sft_base_hf rxnpred 0 tagged
