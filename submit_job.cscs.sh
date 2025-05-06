sbatch \
    --nodes=2 \
    --time=00:40:00 \
    --partition=debug \
    launch_CSCS.slurm Qwen2.5-3B_pretrained-v6-1 iupacsm.long.vllm15 0 tagged
