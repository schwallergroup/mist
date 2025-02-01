
import os
from typing import Dict
import re

from datasets import Dataset, DatasetDict
from typing import Dict, Optional, List
import json
import random

class CustomTask:
    """Base class for custom tasks."""
    def __init__(
        self,
        root_dir: str,
        src_train_file: str,
        tgt_train_file: str,
        src_test_file: Optional[str] = None,
        tgt_test_file: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.root_dir = root_dir

        self.src_train_file = os.path.join(root_dir, src_train_file)
        self.tgt_train_file = os.path.join(root_dir, tgt_train_file)
        self.src_test_file = os.path.join(root_dir, src_test_file) if src_test_file else None
        self.tgt_test_file = os.path.join(root_dir, tgt_test_file) if tgt_test_file else None
        self.cache_dir = cache_dir
        
    def accuracy_reward(self, completions, solution, **kwargs):
        """Define accuracy reward function for the specific dataset."""
        pass

    def format_reward(self, completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think><answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    def read_files(self, src_file: str, tgt_file: str) -> Dict:
        """Read source and target files and create dataset dictionary."""
        with open(src_file, 'r', encoding='utf-8') as f:
            problems = [line.strip() for line in f.readlines()]
            
        with open(tgt_file, 'r', encoding='utf-8') as f:
            solutions = [line.strip() for line in f.readlines()]
            
        # Create empty messages list for each example
        messages = [[] for _ in range(len(problems))]
        
        return {
            'problem': problems,
            'solution': solutions,
            'messages': messages
        }
    
    def add_messages(self, dataset: Dataset, messages: List[List[str]]) -> Dataset:
        """Add messages to the dataset."""
        assert len(messages) == len(dataset), "Messages length must match dataset length"
        dataset = dataset.add_column('messages', messages)
        return dataset
    
    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        # Load training data
        train_dict = self.read_files(self.src_train_file, self.tgt_train_file)
        train_dataset = Dataset.from_dict(train_dict)
        
        # Load or create test data
        if self.src_test_file and self.tgt_test_file:
            test_dict = self.read_files(self.src_test_file, self.tgt_test_file)
            test_dataset = Dataset.from_dict(test_dict)
        else:
            # Create test split from training data
            train_test_split = train_dataset.train_test_split(test_size=0.001)
            train_dataset = train_test_split['train']
            test_dataset = train_test_split['test']
        
        # Combine into DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        return dataset_dict
    
    def save_to_disk(self, dataset: DatasetDict, output_dir: str):
        """Save dataset to disk."""
        dataset.save_to_disk(output_dir)
    
    @staticmethod
    def load_from_disk(input_dir: str) -> DatasetDict:
        """Load dataset from disk."""
        return DatasetDict.load_from_disk(input_dir)

# Usage example:
if __name__ == "__main__":
    # Initialize loader
    loader = CustomTask(
        src_train_file='src-train.txt',
        tgt_train_file='tgt-train.txt',
        src_test_file='src-test.txt',
        tgt_test_file='tgt-test.txt'
    )
    
    # Load dataset
    dataset = loader.load()
    
    # Print structure
    print(dataset)
    
    # Save to disk
    loader.save_to_disk(dataset, 'path/to/save')
    
    # Load from disk
    loaded_dataset = CustomDatasetLoader.load_from_disk('path/to/save')