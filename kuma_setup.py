
LAUNCH_TEMPLATE_PATH = 'launch.template.slurm'

def main():
    print("Please enter the following information:")
    
    config = {
        'wandb_api_key': input("WandB API key: ").strip(),
        'cache_dir': input("Cache directory path, where the checkpoints would be saved: ").strip(),
    }
    
    with open(LAUNCH_TEMPLATE_PATH, 'r') as file:
        template = file.read()
    
    for k, v in config.items():
        key_in_bash = k.upper()
        template = template.replace(f"export {key_in_bash}=", (f"export {key_in_bash}={v}"))
    
    with open('launch.slurm', 'w') as file:
        file.write(template)
    
    print("")
    print("A `launch.slurm` file has been created with the provided information.")
    print("You can directly use this file to launch the job as follows:")
    print("\t`sbatch launch.slurm <model_id> <task_id> <resume_job_id>`")
    print("- <model_id> is the model you want to use, can be found in `model_paths.txt`.")
    print("- <task_id> is the id of the task you want to run, which is the basename of the yaml files in `recipes`.")
    print("- <resume_job_id> is used when you want to resume a job from a checkpoint. If run from scratch, set it to 0.")
    print("If you want more control over `sbatch` paramters such as `--nodes`, `--time`, `--mem`,... you can also edit these information in the `submit_job.sh` file and use this file to submit job instead, which is more recommended rather than directly editing the `launch.slurm` file.")

if __name__ == "__main__":
    main()