
from ..base import RLTask
from random import random
from typing import Dict, Optional
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from open_r1.download_data import download_data


class ForwardReaction(RLTask):
    src_train_file: str = ""
    tgt_train_file: str = ""
    src_test_file: str  = ""
    tgt_test_file: str = ""
    question_template: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(self.dataset_id_or_path):
            os.makedirs(self.dataset_id_or_path)
        download_data(self.dataset_id_or_path)

        self.src_train_file = os.path.join(self.dataset_id_or_path, "src-train.txt")
        self.tgt_train_file = os.path.join(self.dataset_id_or_path, "tgt-train.txt")
        self.src_test_file = os.path.join(self.dataset_id_or_path, "src-test.txt") if "src-test.txt" else None
        self.tgt_test_file = os.path.join(self.dataset_id_or_path, "tgt-test.txt") if "tgt-test.txt" else None
        self.question_template = (
            f"What is the product of the following reaction? Here are the reactants in SMILES notation: {self.begin_smiles_tag} {{}} {self.end_smiles_tag} "
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=C... </answer>. Think step by step inside <think> tags."
        )

    def process_line(self, line: str) -> str:
        """Process a line from the source file."""
        rm_space = re.sub(" +", "", line)
        return rm_space

    def read_files(self, src_file: str, tgt_file: str) -> Dict:
        """Read source and target files and create dataset dictionary."""
        with open(src_file, 'r', encoding='utf-8') as f:
            problems = [self.question_template.format(self.process_line(line)) for line in f.readlines()]
            
        with open(tgt_file, 'r', encoding='utf-8') as f:
            solutions = [self.process_line(line) for line in f.readlines()]
        
        return {
            'problem': problems,
            'solution': solutions,
        }

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
            train_test_split = train_dataset.train_test_split(test_size=0.1)
            train_dataset = train_test_split['train'].unique(column='solution')
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
        for content, sol in zip(completions, solution):
            ans = self.preprocess_response(content)
            if ans == "NONE":
                rewards.append(-1)
                continue
            try:
                gold_mol = Chem.MolToSmiles(Chem.MolFromSmiles(sol))
            except:
                # invalid target smiles
                rewards.append(-1)
                continue
            try:
                completion_mol = Chem.MolToSmiles(Chem.MolFromSmiles(ans))
            except:
                # invalid generated smiles
                rewards.append(-1) # penalize if invalid smiles
                continue
            if gold_mol == completion_mol:
                rewards.append(1)  # reward if correct
                self.log_correct(content)
            else:
                rewards.append(-0.5) # no reward if incorrect
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
