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
from dataclasses import field
from collections import Counter
from smact.screening import smact_validity
from pymatgen.core import Composition
import json



class ConditionalMaterialGeneration(RLTask):
    question_template: str = ""
    log_custom_metrics: bool = True
    custom_metrics: dict = field(default_factory=dict)
    seen_comps_set: set = field(default_factory=set)
    element_usage_counter: Counter = field(default_factory=Counter)
    space_group_usage_counter: Counter = field(default_factory=Counter)
    MAX_TRACKED: int = 0
    recent_compositions: list = field(default_factory=list)
    recent_space_groups: list = field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.question_template = f"""<|im_start|>assistant\You are a material science expert, and I have a task for you. Given the following elements, please generate valid and novel material from these elements. Show your reasoning in <think>...</think> tags and return the final answer in <material>...</material> tags.<|im_end|>\n<|im_start|>user\{{instruction}}. Please keep your reasoning as concise as possible. For example <material> A A B B B <sg12></material> where A, B refer to elements and <sg12> denotes the space group for example: \n<material> Pa In Tc Tc <sg225></material>.<|im_end|>\n<|im_start|>assistant\Response:\n<think>"""
        self.log_custom_metrics = True
        self.custom_metrics = {
            'val/rewards': [],
        }
        with open("/iopsstor/store/cscs/swissai/a05/chem/comps_used_in_sft.json", "r") as file:
            seen_comps = json.load(file)
        self.seen_comps_set = set() 
        for comp in seen_comps:
            comp = Composition(comp)
            self.seen_comps_set.add(comp)

        self.element_usage_counter = Counter()
        self.space_group_usage_counter = Counter()
        self.recent_compositions = []
        self.recent_space_groups = []
        self.MAX_TRACKED = 100

        

        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

    def read_files(self) -> Dict:
        with open(self.dataset_id_or_path, "r") as file:
            data = json.load(file)

        # Generate problems using the question template
        problems = [self.question_template.format(pt["instruction"]) for pt in data]
        # Solutions are the raw target records (assuming no further processing needed)
        solutions = []

        return {
            "problem": problems,
            "solution": solutions,
        }

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        # Load training data
        train_dict = self.read_files()
        full_dataset = Dataset.from_dict(train_dict)
        seed = 42
        full_dataset = full_dataset.shuffle(seed=seed)
        max_examples = 2200
        full_dataset = full_dataset.select(range(max_examples))
        train_dataset = Dataset.from_dict(full_dataset)

        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=seed)
        train_dataset = train_test_split["train"].unique(column="solution")
        test_dataset = train_test_split["test"]

        # Combine into DatasetDict
        self.dataset = DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )

        return self.dataset
    
    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - check that completion is same as ground truth."""
        rewards = []

        for c in completions:
            reward = 0

            # Extract elements from instruction
            input_pattern = r"Build a material that has\s+(.*)"
            match = re.search(input_pattern, c)
            input_elements = match.group(1).split(', ') if match else []

            # Extract elements and space group from output
            output_pattern = r"<material>\s*((?:[A-Z][a-z]?\s*)+?)\s*<sg(\d+)>\s*</material>"
            output_matches = re.findall(output_pattern, c)
            if len(output_matches) <= 2:
                rewards.append(reward)
                continue

            reward += 1
            elements_str, sg_str = output_matches[-1]
            output_sg = int(sg_str.strip())

            if not 1 <= output_sg <= 230:
                rewards.append(reward)
                continue
            reward += 1

            output_elements = elements_str.strip().split()

            # Penalize extra elements not in input
            extra_elements = set(output_elements) - set(input_elements)
            if extra_elements:
                # Calculate overuse for extra elements
                overuse_score = sum(self.element_usage_counter.get(e, 0) for e in extra_elements)
                max_possible_overuse = len(extra_elements) * self.MAX_TRACKED
                
                # Normalize penalty between 0 and 2
                if max_possible_overuse > 0:
                    normalized_penalty = 2 * (overuse_score / max_possible_overuse)
                    reward -= normalized_penalty

            # Penalize overused space group
            overuse_score_sg = self.space_group_usage_counter.get(output_sg, 0)
            max_possible_overuse_sg = self.MAX_TRACKED
            
            if max_possible_overuse_sg > 0:
                normalized_penalty_sg = 2 * (overuse_score_sg / max_possible_overuse_sg)
                reward -= normalized_penalty_sg

            # Check precision
            intersection = set(input_elements) & set(output_elements)
            precision = len(intersection) / len(input_elements)
            reward += precision * 2

            # Try building a composition after applying penalties
            try:
                comp = Composition(" ".join(output_elements))
                if not smact_validity(comp):  # your custom function
                    rewards.append(reward)
                    continue
            except Exception as e:
                print(f"Invalid composition: {output_elements} -> {e}")
                rewards.append(reward)
                continue

            reward += 2

            # Novelty bonus
            if comp not in self.seen_comps_set:
                reward += 2
                self.seen_comps_set.add(comp)

            # Update element and space group usage history
            self.element_usage_counter.update(output_elements)
            self.space_group_usage_counter.update([output_sg])
            self.recent_compositions.append(output_elements)
            self.recent_space_groups.append(output_sg)

            # Maintain rolling window of last 100 outputs for compositions and space groups
            if len(self.recent_compositions) > self.MAX_TRACKED:
                old_elements = self.recent_compositions.pop(0)
                old_space_group = self.recent_space_groups.pop(0)
                self.element_usage_counter.subtract(old_elements)
                self.space_group_usage_counter.subtract([old_space_group])
                # Remove zero or negative counts
                self.element_usage_counter += Counter()  # clean up
                self.space_group_usage_counter += Counter()  # clean up 

            self.custom_metrics['val/rewards'].extend(rewards)
        return rewards

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