# Sink

Chemical reasoning emerges from RL in simple chemical tasks.
This repo is heavily based on [Open-R1](https://github.com/huggingface/open-r1), an open reproduction of DeepSeek-R1, from the HF Team.


## How to

> [!IMPORTANT]  
> Clone this repo into $HOME/Documents/.
> The repository needs to be accessible from this location, and your environment file should mount /Documents

Sample environment file:

```toml
# env.toml
image = "/iopsstor/scratch/cscs/amarulan/vllm_trl_sink.sqsh"
mounts = [
    "/capstor",
    "/iopsstor",
    "/users",
    "/users/amarulan/Documents/:/Documents",
    "/iopsstor/scratch/cscs/amarulan/.cache/:/cache",
]
workdir = "/workspace"
[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```

Run the following script to initialize a `launch.slurm`, which will set you up for running jobs on the CSCS cluster.
```bash
python CSCS_setup.py
```

After this, you'll be able to run jobs like this

```
sbatch launch.slurm [MODEL] [TASK]
```

Where  `[MODEL]` is any model specified in `model_paths.txt` (e.g. Qwen2.5-3B) and `[TASK]` is the short-name for task as specified under `recipes/`.

```bash
# Launch a job for training Qwen2.5-3B as specified in recipes/rxnpred.yaml
sbatch launch.slurm Qwen2.5-3B rxnpred
```



## Documentation

The documentation is built using Sphinx. To build and view the documentation locally:

```bash
cd docs
make html
python -m http.server -d build/html
```

Then open `http://localhost:8000` in your browser.

## Contributing New Tasks

**Sink** is designed to be easily extensible with new chemsitry tasks suitable for reasoning. Each task inherits from the base `RLTask` class and implements specific logic for data handling and reward calculation.

### Creating a New Task

1. Create a new file in `src/open_r1/tasks/` for your task
2. Inherit from the base `RLTask` class
3. Implement required methods
4. Add documentation

Here's a template for creating a new task:

```python
from open_r1.tasks.base import RLTask
from datasets import DatasetDict

class NewTask(RLTask):
    """
    Description of your new task.
    
    This task should [describe what the task does and its purpose].
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = "Your task-specific question format: {}"
        
    def load(self) -> DatasetDict:
        """
        Load and prepare the dataset for the task.
        
        Returns:
            DatasetDict: Dataset with 'train' and 'test' splits
        """
        # Implement dataset loading logic
        pass
        
    def accuracy_reward(self, completions, solution, **kwargs):
        """
        Calculate rewards for model completions.
        
        Args:
            completions (List[str]): Model generated responses
            solution (List[str]): Ground truth solutions
            
        Returns:
            List[float]: Rewards for each completion
        """
        # Implement reward calculation
        pass
```

### Example: Forward Reaction Task

The Forward Reaction task demonstrates how to implement a chemical reaction prediction task:

```python
from open_r1.tasks.base import RLTask
from rdkit import Chem

class ForwardReaction(RLTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            f"What is the product of the following reaction? "
            f"Reactants: {self.begin_smiles_tag} {{}} {self.end_smiles_tag}"
        )
    
    def accuracy_reward(self, completions, solution, **kwargs):
        rewards = []
        for content, sol in zip(completions, solution):
            ans = self.preprocess_response(content)
            try:
                if Chem.MolToSmiles(Chem.MolFromSmiles(ans)) == \
                   Chem.MolToSmiles(Chem.MolFromSmiles(sol)):
                    rewards.append(1)
                else:
                    rewards.append(-0.5)
            except:
                rewards.append(-1)
        return rewards
```

### Task Requirements

When creating a new task, ensure:

1. **Base Class Inheritance**: Inherit from `RLTask`
2. **Required Methods**: Implement at minimum:
   - `load()`: Dataset loading
   - `accuracy_reward()`: Reward calculation
3. **Documentation**:
   - Class docstring explaining the task
   - Method docstrings
   - Example usage
4. **Testing**: Add tests for your task in `tests/`

### Adding Documentation

1. Create a new RST file in `docs/source/api/` for your task
2. Add your task to `docs/source/modules.rst`
3. Include examples and usage instructions
4. Build and verify the documentation

### Current Tasks

- **Forward Reaction**: Chemical reaction product prediction
- [Add your task here]

For detailed examples and API reference, please check the [documentation](link-to-docs).



---

## Installation

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n openr1 python=3.11 && conda activate openr1
```

Next, install vLLM:

```shell
pip install vllm==0.6.6.post1

# For HF (cluster only has CUDA 12.1)
pip install vllm==0.6.6.post1 --extra-index-url https://download.pytorch.org/whl/cu121
```

This will also install PyTorch `v2.5.1` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

```shell
pip install -e ".[dev]"
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
```

## Training models

We support training models with either DDP or DeepSpeed ZeRO-2 and ZeRO-3. To switch between methods, simply change the path to the `accelerate` YAML config in `configs`.

> [!NOTE]
> The training commands below are configured for a node of 8 x H100s (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

### SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k), run:

```
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
```

To launch a Slurm job, run:

```shell
sbatch --output=/path/to/logs/%x-%j.out --err=/path/to/logs/%x-%j.err slurm/sft.slurm {model} {dataset} {accelerator}
```

Here `{model}` and `{dataset}` refer to the model and dataset IDs on the Hugging Face Hub, while `{accelerator}` refers to the choice of 🤗 Accelerate config in `configs`. 

### GRPO

```
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --bf16
```

## Evaluating models

We use `lighteval` to evaluate models, with custom tasks defined in `src/open_r1/evaluate.py`. For models which fit on a single GPU, run:

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=float16,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
    --output-dir $OUTPUT_DIR 
```

To increase throughput across multiple GPUs, use _data parallel_ as follows:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=float16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
    --output-dir $OUTPUT_DIR 
```

For large models which require sharding across GPUs, use _tensor parallel_ and run:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="pretrained=$MODEL,dtype=float16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
    --output-dir $OUTPUT_DIR 
```

You can also launch an evaluation with `make evaluate`, specifying the model, task, and optionally the parallelism technique and number of GPUs.

To evaluate on a single GPU:
```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

To use Data Parallelism:
```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

To use Tensor Parallelism:
```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## Data generation

### Generate data from a smol distilled R1 model

The following example can be run in 1xH100. 
First install the following dependencies:

```shell
pip install "distilabel[vllm]>=1.5.2"
```

Now save the following snippet into a file named `pipeline.py` and run with `python pipeline.py`. It will generate for each of the 10 examples 4 generations (change the username for the repository to your org/user name):

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

Take a look at the sample dataset at [HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b).


### Generate data from DeepSeek-R1

To run the bigger DeepSeek-R1, we used 2 nodes, each with 8×H100 GPUs using the slurm file present in this repo at `slurm/generate.slurm`. First, install the dependencies:

(for now we need to install the vllm dev wheel that [fixes the R1 cuda graph capture](https://github.com/vllm-project/vllm/commits/221d388cc5a836fa189305785ed7e887cea8b510/csrc/moe/moe_align_sum_kernels.cu))
```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

And then run the following command:

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]  
> While the job is running, you can setup an SSH tunnel through the cluster login node to access the Ray dashboard from your computer running `ssh -L 8265:ray_ip_head_node:8265 <login_node>`, then browsing `http://localhost:8265`

## Contributing

Contributions are welcome. Please refer to https://github.com/huggingface/open-r1/issues/23.
