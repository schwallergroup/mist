from ..base import RLTask
from typing import Dict, Optional
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem
import pandas as pd
from Levenshtein import ratio as levenshtein_ratio

class PermuteSmiles(RLTask):
    question_template: str = ""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "Please permute the SMILES sequence for this molecule, in such a way that the SMILES sequence is different from the original SMILES sequence, but the original molecule does not change. Here is the original SMILES: {}. "
            "It is preferred that the resulted SMILES is different from the input SMILES as much as possible. "
            "A strategy you could try, but not obligatory to do, is to reverse the order of the atoms. "
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=C... </answer>. Think step by step inside <think> tags."
        )
    
    def load(self):
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        train_dict = {
            'problem': df['SMILES'].tolist(),
            'solution': df['SMILES'].tolist()
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split = train_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        # Combine into DatasetDict
        self.dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        return self.dataset
    
    def accuracy_reward(self, completions, solution, **kwargs):
        """
        Reward function - check that completed SMILES refers to the same molecule as the original SMILES.
        Bonus if the output SMILES is different from the original SMILES.
        """

        answers = [self.preprocess_response(c) for c in completions]

        rewards = []

        for content, ref in zip(answers, solution):
            if content == "NONE":
                rewards.append(-1)
                continue
            
            response_mol = Chem.MolFromSmiles(content)
            if response_mol is None:
                rewards.append(-1)
                continue
            
            if content == ref:
                rewards.append(-0.5)
                continue
                        
            canon_response = Chem.MolToSmiles(response_mol, canonical=True)
            canon_reference = Chem.MolToSmiles(Chem.MolFromSmiles(ref), canonical=True)
            if canon_response != canon_reference:
                rewards.append(-0.5)
                continue
              
            edit_distance_reward = 1 - levenshtein_ratio(content, ref) # range (0, 1)
            # reward = 0.2 + 0.8*edit_distance_reward
            reward = edit_distance_reward
            
            rewards.append(reward)
        
        return rewards

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