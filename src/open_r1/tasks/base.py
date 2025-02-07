"""Base task definition.
A task needs to be initialized with a dataset
and habe fllowing methods: load() -> for dataset loading, and two functions for the rewards
format_reward, accuracy_reward
"""

import random
import re
import os
from pydantic import BaseModel
from typing import Any, Optional

from datasets import load_dataset

class RLTask(BaseModel):
    dataset_id_or_path: Optional[str] = None
    dataset_splits: Optional[str] = None
    dataset: Optional[Any] = None

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
                if random.random() < 0.1:  # 1% chance to write samples into a file
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
