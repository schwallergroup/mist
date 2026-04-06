# Cluster Support

MiST supports multiple execution environments.

- `launch_CSCS.slurm` is the SwissAI / CSCS launcher example.
- `launch_kuma.slurm` is the Kuma launcher example.

Both launchers source the same repo-level environment variables from
`.env.local` when present. Copy one of the example files below, adjust the
paths for your cluster, and keep the local file out of version control.

- `cluster/cscs.env.example`
- `cluster/kuma.env.example`

Common variables:

- `MIST_MODELS_DIR`: root directory containing base and MiST checkpoints
- `MIST_DATA_DIR`: root directory containing released datasets and task inputs
- `MIST_CACHE_DIR`: scratch/cache directory used for Hugging Face and checkpoints
- `MIST_WANDB_API_KEY_FILE`: path to a file containing the WANDB API key
- `MIST_CONTAINER_PATH`: container image path when needed by the launcher

The Python code resolves repo-local assets automatically and expands
environment variables in recipe dataset paths such as
`${MIST_DATA_DIR}/rxnpred/...`.
