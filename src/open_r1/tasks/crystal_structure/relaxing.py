import os
import re
from random import random
from typing import Dict, Optional
from open_r1.download_data import download_data
import pandas as pd
from datasets import Dataset, DatasetDict
from rdkit import Chem
from ..base import RLTask
import requests


class BinaryCompoundRelaxing(RLTask):
    question_template: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(self.dataset_id_or_path):
            os.makedirs(self.dataset_id_or_path)
        download_data(self.dataset_id_or_path)

        self.src_train_file = os.path.join(
            self.dataset_id_or_path, "src-train.txt"
        )
        self.tgt_train_file = os.path.join(
            self.dataset_id_or_path, "tgt-train.txt"
        )
        self.src_test_file = (
            os.path.join(self.dataset_id_or_path, "src-test.txt")
            if "src-test.txt"
            else None
        )
        self.tgt_test_file = (
            os.path.join(self.dataset_id_or_path, "tgt-test.txt")
            if "tgt-test.txt"
            else None
        )
        self.question_template = (
            "<|im_start|>system You are a seasoned crystallographic structure analysis expert. "
            "Your task is to relax a binary compound to a stable state. <|im_end|>\n"
            "<|im_start|>user Given a perturbed binary compound:\n"
            "{}\n, perform multiple steps of Structural Relaxation on the given perturbed binary compound "
            "and reduce the internal energy. Please document your thought process within <think> </think> tags, and provide "
            "the final corrected structure in <answer> </answer> tags using the proper format as given in the example:\n"
            "serialized_cif formula Cd 1_int As 2_int \n"
            "space_group_symbol I4_122_sg\n"
            "lattice_parameters a 8.03811770 b 8.03811770 c 4.72563470 alpha 90.00000000 beta 90.00000000 gamma 90.00000000 \n"
            "Cd 4_int 0.00000000 0.00000000 0.00000000\n"
            "As 8_int 0.06170692 0.25000000 0.62500000\n"
            "<|im_end|>\n"
        )
        self.log_custom_metrics = True
        self.custom_metrics = {
            'val/rewards': [],
        }

        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

    def read_files(self, src_file: str, tgt_file: str) -> Dict:
        """Read source and target files and create dataset dictionary."""
        with open(src_file, "r", encoding="utf-8") as f:
            problems = [
                self.question_template.format(self.process_line(line))
                for line in f.readlines()
            ]

        with open(tgt_file, "r", encoding="utf-8") as f:
            solutions = [self.process_line(line) for line in f.readlines()]

        return {
            "problem": problems,
            "solution": solutions,
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
            train_dataset = train_test_split["train"].unique(column="solution")
            test_dataset = train_test_split["test"]

        # Combine into DatasetDict
        self.dataset = DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )

        return self.dataset
    
    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - check that completion is same as ground truth."""

        answers = [self.preprocess_response(c) for c in completions]

        rewards = []

        # Here task is simple: check that the smiles is the same as the target smiles
        for content, sol in zip(answers, solution):
            if content == "NONE":
                rewards.append(-10)
                continue

            server_url = os.environ.get("SERVER_URL", "http://10.197.48.175:9001/compute_score")
            if content == sol:
                rewards.append(-10)
                continue

            payload = {
                "answer_text": content,
                "ground_truth": sol
            }
            
            try:
                response = requests.post(server_url, json=payload, timeout=20)
                response.raise_for_status()
                data = response.json()
                reward = data.get("reward", -10)
                rewards.append(reward)
            except Exception as e:
                rewards.append(-10)
        if self.log_custom_metrics:
            self.custom_metrics['val/rewards'].extend(rewards)
        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        pattern = r"<answer>(.*)<\/answer>"
        m = re.findall(pattern, response, re.DOTALL)
        if m:
            return m[-1].strip()
        else:
            return "NONE"

    def get_metrics(self) -> Dict:
        """
        Get task metrics to log in WANDB.
        This function takes no arguments and returns a dictionary of metrics {key[str]: value[float]}.
        """
        metrics = dict()
        if self.log_custom_metrics:
            rewards = self.custom_metrics['val/rewards']
            if rewards:
                correct_count = sum(1 for r in rewards if r == 1)
                total_count = len(rewards)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                metrics['val/accuracy'] = accuracy
                self.custom_metrics['val/rewards'] = []
        return metrics