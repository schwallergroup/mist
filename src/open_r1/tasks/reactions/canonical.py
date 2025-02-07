
from ..base import RLTask
from typing import Dict
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem
import pandas as pd

class CanonicalizeSmiles(RLTask):
    data_dir: str = ""
    question_template: str = ""

    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.question_template = (
            "What is the canonical SMILES for this molecule? Here is a non-canonical SMILES: {} "
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=C... </answer>. Think step by step inside <think> tags."
        )

        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.data_dir)
        train_dict = {
            'problem': df['SMILES_variant1'].tolist(),
            'solution': df['SMILES'].tolist()
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

    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - check that completion is same as ground truth."""

        answers = [self.preprocess_response(c) for c in completions]

        rewards = []

        # Here task is simple: check that the smiles is the same as the target smiles
        for content, sol in zip(answers, solution):
            if content == "NONE":
                rewards.append(-1)
                continue
            if content == sol:
                rewards.append(1)
            else:
                # It gets a point if when we canonicalize it, it's the same
                try:
                    completion_mol = Chem.MolToSmiles(Chem.MolFromSmiles(content))
                    if completion_mol == sol:
                        rewards.append(0.2) # as it didnt directly predict the correct canonical smiles
                    else:
                        rewards.append(-0.5) # at least its a valid smiles
                except:
                    # invalid generated smiles
                    rewards.append(-1)

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        pattern = r"<answer>(.*)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            smi = m.groups()[0]

            # Maybe smiles contains [BEGIN_SMILES] and [END_SMILES]
            if "[BEGIN_SMILES]" in smi:
                smi = smi.replace("[BEGIN_SMILES]", "")
            if "[END_SMILES]" in smi:
                smi = smi.replace("[END_SMILES]", "")

            return smi
        else:
            return "NONE"
