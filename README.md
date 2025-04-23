# Sink

Chemical reasoning emerges from RL in simple chemical tasks.

This repo is heavily based on [Open-R1](https://github.com/huggingface/open-r1), an open reproduction of DeepSeek-R1, from the HF Team.


## 🔥 How to

> [!IMPORTANT]  
> Clone this repo into `$HOME/`. This partition has no cleaning policy and your code should be stored there. You can clone it into `$HOME/`, `$HOME/Documents/` or into any other folder of your choice.

Run the following script to setup all the necessary files/environments and follow the instructions. It will set you up for running jobs on the CSCS or Kuma cluster.
```bash
python3 CSCS_setup.py
# or
python3 kuma_setup.py
```

**Cluster-specific launch files (used in the example below):**
- **If you are using CSCS cluster, you need to use `launch_CSCS.slurm` instead of `launch.slurm`.**
- **If you are using kuma cluster, you need to use `launch_kuma.slurm` instead of `launch.slurm`.**

After this, you'll be able to run jobs as shown below. `[MODEL]` is any model specified in `model_paths.txt` (e.g. Qwen2.5-3B) and `[TASK]` is the short-name for task as specified under `recipes/` (without the suffix `.yaml`).
```bash
sbatch launch.slurm [MODEL] [TASK]

# Example: launch a job for training Qwen2.5-3B as specified in recipes/rxnpred.yaml
sbatch launch.slurm Qwen2.5-3B rxnpred
```

A third optional parameter is `[RESUME_JOB_ID]`. It is used if you would like to continue the training of a previous job. `[RESUME_JOB_ID]` should contain the job ID of the previous job you want to continue from. If you want to start a run from scratch (without using a previous run checkpoint), then you can set this parameter to 0 (however it is not necessary since the default value is 0 if the parameter is omitted).
```bash
sbatch launch.slurm [MODEL] [TASK] [RESUME_JOB_ID]

# Example: launch a job for training Qwen2.5-3B as specified in recipes/rxnpred.yaml, continuing from job ID 123456
sbatch launch.slurm Qwen2.5-3B rxnpred 123456

# Example: launch from scratch
sbatch launch.slurm Qwen2.5-3B rxnpred 0
```

A final fourth optional parameter is `[TASK_MODE]`, it can be used if you would like to use a specific mode in a task directly from the launch SLURM script. The goal of `[TASK_MODE]` is to allow to run the same recipe file with the same Task class with some small differences (without rewriting multiple subclasses of the same class), for example:
- If you would like to process the dataset in a different manner.
- If you would like to apply different chat templates / prompt templates.
- If you would like to compute the rewards in a different manner.
- Etc... (You can do whatever you want)

If you use it, the parameter `[TASK_MODE]` will be given to the task in `self.task_mode`. It is useful if you would like to run the same recipe files with multiple task modes without rewriting dozens of individual recipes and task classes.

Notes:
- If you would like to use the parameter `[TASK_MODE]`, you need to pass it as the fourth parameter. Therefore, you need to specify the third parameter `[RESUME_JOB_ID]` as well (even if you don't use it). If you do not want to use `[RESUME_JOB_ID]`, you can set it to 0.
- You can completely omit this fourth parameter, and it won't affect anything if you don't use it. The default value for `[TASK_MODE]` is `"base"`.
- The `task_mode` parameter should never be specified in a recipe file.
```bash
sbatch launch.slurm [MODEL] [TASK] [RESUME_JOB_ID] [TASK_MODE]

# Example: launch a job for training Qwen2.5-3B as specified in recipes/rxnpred.yaml, running from scratch with task mode "base"
sbatch launch.slurm Qwen2.5-3B rxnpred 0 base
```

Since the default values are:
- `[RESUME_JOB_ID]` = 0 (start from scratch)
- `[TASK_MODE]` = "base" (base task mode)

The 3 following commands are equivalent:
```bash
sbatch launch.slurm Qwen2.5-3B rxnpred
sbatch launch.slurm Qwen2.5-3B rxnpred 0
sbatch launch.slurm Qwen2.5-3B rxnpred 0 base
```

## 📖 Documentation

The documentation is built using Sphinx. To build and view the documentation locally:

```bash
cd docs
make html
python -m http.server -d build/html
```

Then open `http://localhost:8000` in your browser.

## Contributing New Tasks

**Sink** is designed to be easily extensible with new chemistry tasks suitable for reasoning. Each task inherits from the base `RLTask` class and implements specific logic for data handling and reward calculation.

### Creating a New Task

1. Create a new file in `src/open_r1/tasks/` for your task, e.g. `sampletask.py`
2. Inherit from the base `RLTask` class and implement required methods, e.g. `SampleTask(RLTask)`
    - During the GRPO training script, the methods `load`, `dataset_preprocess` and the different reward functions `*_reward` will be called.
3. Add class to `CHEMTASKS` in `src/open_r1/tasks/__init__.py`, e.g. `'sampletask': SampleTask`
4. Write a recipe with the same name as the task `recipes/sampletask.yaml`
    - The run will be logged on wandb under the project named r1-`[TASK]` (e.g. r1-sampletask). Therefore, runs using different recipe files will be logged under different wandb projects.
    - If you add a dot in your recipe filename (e.g. `sampletask.variant1.yaml`), the run will also be logged under r1-sampletask (everything after the dot will be ignored). This is useful if you want to run multiple experiments with different recipe files but keep them under the same wandb project for analysis.
5. Add documentation:
    - Create an entry under `docs/source/tasks/sampletask.rst` (use the template.rst)
    - Add it to the modules index: `docs/source/modules.rst` as `tasks/sampletask.rst`

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

##### Additional information to create a task class

Here is a list of the methods that you can implement in your task class:
1. **`load()`** (mandatory): This method is responsible for loading the dataset (it is called during the GRPO training). It usually uses the `dataset_id_or_path` defined in the recipe file (automatically parsed from the recipe file and set in `self.dataset_id_or_path` in your task) and create the class variable `self.dataset`.
    - Input: nothing
    - Output: nothing
2. **`dataset_preprocess()`** (optional): This method is called after the `load()` method in the GRPO training and is used to preprocess the dataset. The method is defined by default in the base class so you can omit it if you don't need to preprocess the dataset in a custom manner. If you would like to add custom preprocessing, you can override this method and takes inspiration from the implementation in the base class (`RLTask`).
    - Input: tokenizer (huggingface Tokenizer)
        - The tokenizer can be used in the method to apply a chat template for example (`tokenizer.apply_chat_template()`).
    - Output: dataset (huggingface Dataset)
        - This dataset should contain two splits: `dataset["train"]` and `dataset["test"]`.
        - Each of these splits should at least contain a column named `"prompt"`, you can also add as many other columns as you need (then the other columns can be used during the computations of the rewards).
3. **`*_reward()`** (optional): You can implement as many reward functions as you want. The list of reward functions used during an experiment should be specified in the recipe file.
    - Input: completions, **kwargs
        - During the GRPO training, the reward functions will be called automatically with the following parameters:
            - `completions`: list of strings containing the generated text completions (without the prompts). The completions usually contains the thinking and the answer if you follow the standard format.
            - `**kwargs`: any additional column found in `self.dataset` will be passed as keyword arguments (in a list in the same way as `completions`). For example, if you have a column named `"prompt"`, the parameter `prompt=...` will be taken as input. It is useful if you would like to compute rewards and checking if the solutions/expected answer is found in the completions.
    - Output: rewards (list of reward float values with the same length as completions)
4. **`accuracy_reward()`** (mandatory): This reward function is used to evaluate the accuracy of the reward (if the answer equals the expected solution).
5. **`format_reward()`** (optional): Predefined reward used to reward the correct formatting of the completions (<think> and <answer> tags correctly formatted in the completions).
6. **`reasoning_steps_reward()`** (optional): Predefined reward used to reward a step-by-step thinking in the completions.
7. **`get_metrics()`** (optional): Optional function that can be used to log additional metrics in the wandb run. This function is called automatically during the GRPO training.
    - Input: nothing
    - Output: dictionary of metrics with the format `{key[str]: value[float]}`
        - Each metrics will be logged in wandb in `custom/[key]` with the value `[value]`.
        - These metrics can be computed during the reward functions for example and saved in a class variable of your choice. The function `get_metrics()` just need to output these values for logging.

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

### Recipes

To specify a recipe for your task, copy from `recipes/template.yaml` and modify the section # Chemical Task Arguments:
```yaml
# recipes/my_task.yaml
chem_task: my_task
dataset_id_or_path: /iopsstor/store/cscs/swissai/a05/...  # can be a HF dataset
rewards:
- accuracy
- format
task_kwargs:
  my_kwarg1: my_value1
  my_kwarg2: my_value2
```

1. **chem_task**: Name of the task class. The task names are defined in the file `src/open_r1/tasks/__init__.py` in the keys of the variable `CHEMTASKS`. 
2. **dataset_id_or_path**: Path to the dataset. This argument can be used anywhere in the task class, but it is usually used in the `load` method.
3. **rewards**: List of reward functions to be used. The available reward functions depend on the task (you can implement as many as you want). However, each reward function in the task class should ends by `_reward` and the suffix is omitted in the recipe file. For example, if you want to use your function `accuracy_reward()`, you need to specify `accuracy` in the recipe file.
4. **task_kwargs**: Special argument (dict-like) that can contain any additional keyworded argument that you would like to pass to your task.
    - This argument is optional and can be omitted in the recipe file if you don't need it.
5. There are many other training parameters in the recipe file, you can keep the default values (as in `recipes/template.yaml`) but feel free to modify them if you need it (however it could lead to unexpected results or crashes). If you just built your task, it's recommended to keep the default parameters, ensure that your task is working and then modify these parameters to your needs.

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

1. Create a new RST file in `docs/source/tasks/` for your task. Use the template under `tasks/template.rst`.
2. Add your task to `docs/source/modules.rst`
3. Include examples and usage instructions
4. Build and verify the documentation:
    ```cd docs; make clean; make html; python -m http.server 7000```

### Current Tasks

- **Forward Reaction**: Chemical reaction product prediction
- [Add your task here]

For detailed examples and API reference, please check the [documentation](link-to-docs).

### Models
The list of models can be found in the file `model_paths.txt`.
- The models are stored in the folder `LLM_models/`.
- If you want to add a new model, you can add it in the file `model_paths.txt` (in a new line with the format `[model_id]: [path]`) and add the model in the appropriate folder (`LLM_models`).

Multiple custom models were pretrained:
- Qwen2.5-3B_pretrained-v1
  - Original name: qwen_3b_pretrained
- Qwen2.5-3B_pretrained-v1_cot-v1
  - Original name: qwen-cot
- Qwen2.5-3B_pretrained-v2
  - Original name: qwen_3b_sft
- Qwen2.5-3B_pretrained-v3
  - Original name: qwen_sft_full
- Qwen2.5-3B_pretrained-v4-cot
  - Original name: qwen_cot_v3
