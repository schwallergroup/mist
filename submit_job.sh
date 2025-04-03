sbatch \
    --nodes=1 \
    --time=08:00:00 \
    launch.slurm \
    Qwen2.5-0.5B \
    rxnpred_with_tags.small \  # basename of task yaml file
    0 \  # Checkpoint to resume from. If 0 then start from scratch