sbatch \
    --nodes=4 \
    --time=08:00:00 \
    launch_CSCS.slurm Qwen2.5-3B_pretrained-v5 rxnpred_with_tags.4nodes_cscs
