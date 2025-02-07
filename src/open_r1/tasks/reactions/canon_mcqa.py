
from ..base import RLTask
import numpy as np
from typing import Dict
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem
import pandas as pd

class CanonicalizeSmilesMCQA(RLTask):
    data_dir: str = ""
    question_template: str = ""

    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.question_template = (
            "What is the canonical SMILES for this molecule? Here is a non-canonical SMILES: {} "
            "Choose from the following options, respond only with the option letter. Options: A. {}\nB. {}\nC. {}\nD. {}\n"
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> [your response] </answer>. Think step by step inside <think> tags."
        )

        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.data_dir)
        shuffled = [np.random.permutation(row).tolist() for row in df[['SMILES_variant2', 'SMILES_variant3', 'SMILES_variant4', "SMILES"]].values]
        train_dict = {
            'problem': df['SMILES_variant1'].tolist(),
            'solution': df['SMILES'].tolist(),
            'options': shuffled
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split = train_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        # Combine into DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        return dataset_dict

    def accuracy_reward(self, completions, solution, options, **kwargs):
        """Reward function - check that completion is same as ground truth."""

        answers = [self.preprocess_response(c) for c in completions]
        rewards = []

        letters = "ABCD"
        for i, (ans, sol) in enumerate(zip(answers, solution)):
            try:
                idx = letters.index(ans)
                select = options[i][idx]
                if select == sol:
                    rewards.append(1)
                else:
                    rewards.append(-0.1)
            except:
                rewards.append(-0.1)
        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        pattern = r"<answer>(.*)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            ans = m.groups()[0]
            return ans
        else:
            return "NONE"
