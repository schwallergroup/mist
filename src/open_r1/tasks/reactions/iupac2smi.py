
from ..base import RLTask
from typing import Dict
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from random import random

class Iupac2Smiles(RLTask):
    question_template: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "What is the SMILES for this molecule? {}. "
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=C... </answer>. Think step by step inside <think> tags."
        )
        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        train_dict = {
            'problem': df['IUPAC'].tolist(),
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
        """Reward function - check that completion is same as ground truth."""
        rewards = []

        # Here task is simple: check that the smiles is the same as the target smiles
        for content, sol in zip(completions, solution):
            ans = self.preprocess_response(content)
            if ans == "NONE":
                rewards.append(-1)
                continue
            if ans == sol:
                rewards.append(1)
                self.log_correct(content)
            else:
                # It gets a point if when we canonicalize it, it's the same
                try:
                    completion_mol = Chem.MolToSmiles(Chem.MolFromSmiles(ans))
                    if completion_mol == sol:
                        rewards.append(0.2) # as it didnt directly predict the correct canonical smiles
                    else:
                        rewards.append(-0.5) # at least its a valid smiles
                except:
                    # invalid generated smiles
                    rewards.append(-1)
        return rewards

    def tanimoto_accuracy_reward(self, completions, solution, **kwargs):
        """Reward function using Tanimoto similarity between prediction and ground truth."""
        answers = [self.preprocess_response(c) for c in completions]
        
        rewards = []
        for content, sol in zip(answers, solution):
            if content == "NONE":
                rewards.append(-1)
                continue
                
            try:
                # Convert ground truth SMILES to molecule and fingerprint
                gold_mol = Chem.MolFromSmiles(sol)
                if gold_mol is None:
                    rewards.append(-1)
                    continue
                gold_fp = AllChem.GetMorganFingerprintAsBitVect(gold_mol, 2)
                
                # Convert prediction SMILES to molecule and fingerprint
                pred_mol = Chem.MolFromSmiles(content)
                if pred_mol is None:
                    rewards.append(-1)
                    continue
                pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2)
                
                # Calculate Tanimoto similarity
                tanimoto = DataStructs.TanimotoSimilarity(gold_fp, pred_fp)
                
                # Scale the reward: 
                # 1.0 for perfect match
                # Proportional to similarity for partial matches
                # Still penalize very poor predictions
                if tanimoto == 1.0:
                    reward = 1.0
                    self.log_correct(content)
                elif tanimoto < 0.3:  # You can adjust this threshold
                    reward = -0.5
                else:
                    reward = tanimoto - 0.3  # Shifts the reward to be negative for very low similarities
                    
                rewards.append(reward)
                
            except Exception as e:
                rewards.append(-1)
                continue
                
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
