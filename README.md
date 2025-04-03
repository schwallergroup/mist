# Sink

Chemical reasoning emerges from RL in simple chemical tasks.

This repo is heavily based on [Open-R1](https://github.com/huggingface/open-r1), an open reproduction of DeepSeek-R1, from the HF Team.


## 🔥 How to

> [!IMPORTANT]  
> Clone this repo into `$HOME/`. This partition has no cleaning policy and your code should be stored there. You can clone it into `$HOME/`, `$HOME/Documents/` or into any other folder of your choice.

Run the following script to setup all the necessary files/environments and follow the instructions. It will set you up for running jobs on the Kuma cluster.
```bash
python3 kuma_setup.py
```

A `launch.slurm` script will be generated based on the `launch.template.slurm` template file.

After this, you'll be able to run jobs as shown below. `[MODEL]` is any model specified in `model_paths.txt` (e.g. Qwen2.5-3B) and `[TASK]` is the short-name for task as specified under `recipes/`, and `[RESUME_JOB_ID]` is the job ID of a previous job you want to continue from, or 0 if run from scratch.

```bash
sbatch launch.slurm [MODEL] [TASK] [RESUME_JOB_ID]

# Launch a job for training Qwen2.5-3B as specified in recipes/rxnpred.yaml, continuing from job ID 123456
sbatch launch.slurm Qwen2.5-3B 123456

# Or launch from scratch
sbatch launch.slurm Qwen2.5-3B 0
```
Alternatively, you can also submit a job using the `submit_job.sh`, which provides you the option to specify the paramters of `sbatch` without modifying the `launch.template.slurm` script. 

```bash
# After editing the `submit_job.sh` script, run:
sh submit_job.sh
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

**Sink** is designed to be easily extensible with new chemsitry tasks suitable for reasoning. Each task inherits from the base `RLTask` class and implements specific logic for data handling and reward calculation.

### Creating a New Task

1. Create a new file in `src/open_r1/tasks/` for your task, e.g. `sampletask.py`
2. Inherit from the base `RLTask` class and implement required methods, e.g. `SampleTask(RLTask)`
3. Add class to `CHEMTASKS` in `tasks/__init__.py`, e.g. `'sampletask': SampleTask`
4. Write a recipe with the same name as the task `recipes/sampletask.yaml`
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

To specify a recipe for your task, copy from template.yaml and modify the section # Chemical Task Arguments:
```yaml
# recipes/my_task.yaml
chem_task: my_task
dataset_id_or_path: /iopsstor/store/cscs/swissai/a05/...  # can be a HF dataset
rewards:
- sequential
task_mode: my_task_mode
task_kwargs:
  my_kwarg: my_value
```

Where `task_mode` is a special argument whose meaning can be specified arbitrarily for each task, and `task_kwargs` are any other list of arguments. 

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

