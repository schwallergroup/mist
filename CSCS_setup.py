import os
from pathlib import Path

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent

    # Get username
    username = os.getlogin()

    # Create .cache directory in scratch
    path_scratch = f"/iopsstor/scratch/cscs/{username}"
    assert os.path.isdir(path_scratch), f"The user's scratch directory does not exist: {path_scratch}"
    if os.path.isdir(f"{path_scratch}/.cache") is False:
        os.mkdir(f"{path_scratch}/.cache")

    # Create environment in .edf folder
    path_home = f"/users/{username}"
    assert os.path.isdir(path_home), f"The user's home directory does not exist: {path_home}"
    if os.path.isdir(f"{path_home}/.edf") is False:
        os.mkdir(f"{path_home}/.edf")
    path_environment = f"{path_home}/.edf/vllm071.toml"
    if os.path.isfile(path_environment) is False:
        with open(path_environment, "w") as f:
            path_documents = str(repo_root.parent)
            content = (
                f'image = "/iopsstor/store/cscs/swissai/a05/containers/vllm_trl_sink.sqsh"\n'
                f"mounts = [\n"
                f'  "/capstor",\n'
                f'  "/iopsstor",\n'
                f'  "/users",\n'
                f'  "/iopsstor/store/cscs/swissai/a05/LIAC/LLM_models/:/LLM_models",\n'
                f'  "{path_documents}/:/Documents",\n'
                f'  "/iopsstor/scratch/cscs/{username}/.cache/:/cache",\n'
                f"]\n"
                f"[annotations]\n"
                f'com.hooks.aws_ofi_nccl.enabled = "true"\n'
                f'com.hooks.aws_ofi_nccl.variant = "cuda12"'
            )
            f.write(content)

    # WANDB setup
    wandb_api_key_needed = True
    wandb_api_key_path = repo_root / "wandb_api_key.txt"
    if wandb_api_key_path.is_file():
        with open(wandb_api_key_path, "r") as f:
            wandb_api_key = f.read().strip()
            if len(wandb_api_key) == 0:
                wandb_api_key_needed = True
    if wandb_api_key_needed:
        wandb_api_key = input("Please enter your WANDB API key (found in https://wandb.ai/settings): ").strip()
        with open(wandb_api_key_path, "w") as f:
            f.write(wandb_api_key)
        input(
            'Ensure that the "Default team" in https://wandb.ai/settings is set to "liac" and press Enter to continue.'
        )

    env_file = repo_root / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                f'export MIST_REPO_ROOT="{repo_root}"',
                'export MIST_MODELS_DIR="/iopsstor/store/cscs/swissai/a05/LIAC/LLM_models"',
                'export MIST_DATA_DIR="/iopsstor/store/cscs/swissai/a05/chem"',
                f'export MIST_CACHE_DIR="/iopsstor/scratch/cscs/{username}/.cache/mist"',
                f'export MIST_WANDB_API_KEY_FILE="{wandb_api_key_path}"',
            ]
        )
        + "\n"
    )

    # End
    print(
        "Well done! The setup is complete.\n"
        "A repo-local .env.local file was written for this cluster.\n"
        "You can now run experiments in the MiST directory such as the next example:\n"
        "\tsbatch launch_CSCS.slurm Qwen2.5-0.5B rxnpred\n\n"
        "The general SLURM command is: sbatch launch_CSCS.slurm <model_id> <task> <resume_job_id> <task_mode>\n"
        "\t- <model_id>: The model id to use (can be found in the file model_paths.txt)\n"
        "\t- <task>: The task to run (can be found in the folder recipes; for example for rxnpred.yaml, the task is rxnpred)"
        "\t- <resume_job_id>: Used when you want to resume a job from a checkpoint (if run from scratch, set it to 0)"
        "\t- <task_mode>: The task mode you would like to use"
        "You can find more information in the README file."
    )
