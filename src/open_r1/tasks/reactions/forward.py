
from ..base import RLTask
from typing import Dict
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem
from open_r1.download_data import download_data


class ForwardReaction(RLTask):
    src_train_file: str = ""
    tgt_train_file: str = ""
    src_test_file: str  = ""
    tgt_test_file: str = ""
    question_template: str = ""

    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            download_data(data_dir)

        self.src_train_file = os.path.join(data_dir, "src-train.txt")
        self.tgt_train_file = os.path.join(data_dir, "tgt-train.txt")
        self.src_test_file = os.path.join(data_dir, "src-test.txt") if "src-test.txt" else None
        self.tgt_test_file = os.path.join(data_dir, "tgt-test.txt") if "tgt-test.txt" else None
        self.question_template = (
            "What is the product of the following reaction? Here are the reactants in SMILES notation: {} "
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=NC2=C1C(=O)N(C(=O)N2C)C </answer>. Think step by step inside <think> tags."
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
            train_test_split = train_dataset.train_test_split(test_size=0.001)
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
        for content, sol in zip(answers, solution):
            if content == "NONE":
                rewards.append(-1)
                continue
            try:
                gold_mol = Chem.MolToSmiles(Chem.MolFromSmiles(sol))
            except:
                # invalid target smiles
                rewards.append(-1)
                continue
            try:
                completion_mol = Chem.MolToSmiles(Chem.MolFromSmiles(content))
            except:
                # invalid generated smiles
                rewards.append(-1) # penalize if invalid smiles
                continue
            if gold_mol == completion_mol:
                rewards.append(1)  # reward if correct
            else:
                rewards.append(-0.5) # no reward if incorrect
        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        pattern = r"<answer>.*?</answer>"
        if re.match(pattern, response):
            smi = response.split("<answer>")[1].split("</answer>")[0]

            # Maybe smiles contains [BEGIN_SMILES] and [END_SMILES]
            if "[BEGIN_SMILES]" in smi:
                smi = smi.replace("[BEGIN_SMILES]", "")
            if "[END_SMILES]" in smi:
                smi = smi.replace("[END_SMILES]", "")
            print(smi)
            return smi
        else:
            return "NONE"
