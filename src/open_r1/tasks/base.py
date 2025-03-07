"""Base task definition.
A task needs to be initialized with a dataset
and habe fllowing methods: load() -> for dataset loading, and two functions for the rewards
format_reward, accuracy_reward
"""

import random
import re
import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from datasets import load_dataset

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class RLTask(BaseModel):
    dataset_id_or_path: Optional[str] = None
    dataset_splits: Optional[str] = None
    dataset: Optional[Any] = None
    task_mode: Optional[str] = "base"

    system_prompt: Optional[str] = Field(
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )
    response_print: str = "\n\n======<CORRECT_RESPONSE>========\n{}"
    begin_smiles_tag: str = "[BEGIN_SMILES]"
    end_smiles_tag: str = "[END_SMILES]"

    def load(self) -> Any:
        """Define load method if not hf dataset."""
        if self.dataset_id_or_path is None:
            raise NotImplementedError
        else:
            self.dataset = load_dataset(
                self.dataset_id_or_path,
                split=self.dataset_splits
            )
            return self.dataset

    def accuracy_reward(self, completions, target, **kwargs):
        """Define accuracy reward"""
        raise NotImplementedError

    def generate_prompt(self, problem, tokenizer, **kwargs):
        r1_prefix = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.question_template.format(problem)},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "problem": problem
        }

    def dataset_preprocess(self, tokenizer):
        self.dataset["train"] = self.dataset["train"].shuffle(seed=42).select(range(min(50000, len(self.dataset["train"]))))
        self.dataset["test"] = self.dataset["test"].shuffle(seed=42).select(range(min(10000, len(self.dataset["test"]))))

        self.dataset = self.dataset.map(lambda x: self.generate_prompt(x["problem"], tokenizer))
        return self.dataset

    def log_correct(self, content, p=0.05):
        if random.random() < p:
            print(self.response_print.format(content))

    def format_reward(self, completions, **kwargs):
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
        
        Returns:
            list[float]: Reward scores
        """
        rewards = []

        for completion in completions:
            completion = "<think>" + completion
            try:
                if random.random() < 0.01:  # 1% chance to print a completion
                    print(f"\n\n=======<RANDOM_RESPONSE>=======\n{completion}")
            
                regex = r"<think>(.*)<\/think>\n?<answer>(.*)<\/answer>"
                match = re.search(regex, completion, re.DOTALL) 
                # if the format is not correct, reward is 0
                if match is None or len(match.groups()) != 2:
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def reasoning_steps_reward(self, completions, **kwargs):
        r"""Reward function that checks for clear step-by-step reasoning.
        Regex pattern:
            Step \d+: - matches "Step 1:", "Step 2:", etc.
            ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
            \n- - matches bullet points with hyphens
            \n\* - matches bullet points with asterisks
            First,|Second,|Next,|Finally, - matches transition words
        """
        pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
        completion_contents = [completion for completion in completions]
        matches = [len(re.findall(pattern, content)) for content in completion_contents]

        # Magic number 3 to encourage 3 steps and more, otherwise partial reward
        return [min(1.0, count / 3) for count in matches]

    def get_metrics(self) -> dict:
        """
        Get task metrics to log in WANDB.
        This function takes no arguments and returns a dictionary of metrics {key[str]: value[float]}.
        """
        return dict()
    
