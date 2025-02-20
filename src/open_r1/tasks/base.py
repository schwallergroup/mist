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

class RLTask(BaseModel):
    dataset_id_or_path: Optional[str] = None
    dataset_splits: Optional[str] = None
    dataset: Optional[Any] = None
    system_prompt: Optional[str] = Field(
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

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

    def accuracy_reward(self, completions, solution, **kwargs):
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
                if random.random() < 0.001:  # 1% chance to write samples into a file
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "completion_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)
            
                # Check if the format is correct
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
