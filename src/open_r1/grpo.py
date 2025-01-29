# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import wandb
from dataclasses import dataclass, field
from rdkit import Chem

# Method 1: Using warnings module
import warnings
warnings.filterwarnings('ignore')

# Method 2: Specifically for RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from dataset import ForwardReactionDataset
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )

def preprocess_response(response):
    """Preprocess the response before checking for accuracy."""
    pattern = r"<answer>.*?</answer>"
    if re.match(pattern, response):
        smi = response.split("<answer>")[1].split("</answer>")[0]

        # Maybe smiles contains [BEGIN_SMILES] and [END_SMILES]
        if "[BEGIN_SMILES]" in smi:
            smi = smi.replace("[BEGIN_SMILES]", "")
        if "[END_SMILES]" in smi:
            smi = smi.replace("[END_SMILES]", "")
        return smi
    else:
        return ""

def accuracy_reward(completions, solution, **kwargs):
    """Reward function - chack that completion is same as ground truth."""

    contents = [completion[0]["content"] for completion in completions]
    answers = [preprocess_response(c) for c in contents]

    rewards = []
    for content, sol in zip(answers, solution):
        try:
            gold_mol = Chem.MolToSmiles(Chem.MolFromSmiles(sol))
        except:
            # invalid smiles
            rewards.append(-1)
            continue
        try:
            completion_mol = Chem.MolToSmiles(Chem.MolFromSmiles(content))
        except:
            # invalid smiles
            rewards.append(-1) # penalize if invalid smiles
            continue
        if gold_mol == completion_mol:
            rewards.append(1)  # reward if correct
        else:
            rewards.append(-0.5) # no reward if incorrect
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    training_args.use_vllm = True

    # Only initialize wandb on the main process

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = ForwardReactionDataset(
        root_dir="data/USPTO",
        src_train_file="src-train.txt",
        tgt_train_file="tgt-train.txt",
        src_test_file="src-test.txt",
        tgt_test_file="tgt-test.txt",
    ).load()

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns("messages")

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    accelerator = trainer.accelerator

    if accelerator.is_main_process:
        wandb.init(
            project="GRPO",
            name=training_args.run_name,  # If you defined run_name in config
            config={
                **script_args.__dict__,
                **model_args.__dict__,
            }
        )
    accelerator.wait_for_everyone()

    # Train and push the model to the Hub
    trainer.train()

    if accelerator.is_main_process:
        wandb.finish()
    # # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
