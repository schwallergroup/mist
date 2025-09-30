import os
import re
import random
from typing import Dict, Optional, Any
# from open_r1.download_data import download_data
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
    random_log: Dict[str, Any] = {}
    # element_usage_counter: Counter = field(default_factory=Counter)
    # space_group_usage_counter: Counter = field(default_factory=Counter)
    # MAX_TRACKED: int = 0
    # recent_compositions: list = field(default_factory=list)
    # recent_space_groups: list = field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = "You are a careful model that must follow the Output Contract exactly.\n\nOUTPUT CONTRACT\n1) You may think only inside <think>...</think>.\n2) Your final answer must be a single line wrapped in <answer>...</answer>.\n3) Inside <answer>, list only element symbols separated by single spaces, followed by a space-group tag <sg space-group number>.\n4) If you generate a chemical formula with subscripts (e.g. Ni₂Fe₄LiO₁₀), you must expand them into repeated symbols (Ni Ni Fe Fe Fe Fe Li O O O O O O O O O O).\n5) Do not include any extra words, punctuation, examples, explanations, or text outside the tags.\n6) After you produce </answer>, you must stop. No tokens are allowed after </answer>.\n\nVALID EXAMPLE (format only):\n<think>reasoning</think>\n<answer> Ca O Sn Sn <sg62></answer>\n\nINVALID EXAMPLES:\n- Missing tags\n- Commas, bullets, or explanations inside <answer>\n- Multiple <answer> blocks\n- Extra output after </answer>\n\nAllowed tokens inside <answer>: element symbols (H He Li ... Og), single spaces, and the literal pattern <sg space-group number>."

        self.question_template = "You are a materials science expert.\nGiven the following elements: {}, propose one chemically valid and novel crystalline compound."


        self.log_custom_metrics = True
        self.custom_metrics = {
            'val/rewards': [],
        }
        with open("/capstor/store/cscs/swissai/a131/jmeng/sink/src/open_r1/tasks/condmatgen/comps_used_in_sft.json", "r") as file:
            seen_comps = json.load(file)
        self.seen_comps_set = set() 
        for comp in seen_comps:
            comp = Composition(comp)
            self.seen_comps_set.add(comp)

        # self.element_usage_counter = Counter()
        # self.space_group_usage_counter = Counter()
        # self.recent_compositions = []
        # self.recent_space_groups = []
        # self.MAX_TRACKED = 100

        

        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

    def read_files(self) -> Dict:
        dataset_path = os.path.join(self.dataset_id_or_path, "NatureLM_conditional_v2.json")
        with open(dataset_path, "r") as file:
            data = json.load(file)

        # Generate problems using the question template
        # problems = [self.question_template.format(pt["instruction"]) for pt in data]
        problems = []
        solutions = []

        for pt in data:
            try:
                problems.append(pt.get("elements"))
                solutions.append("")
            except KeyError as e:
                print(pt.keys())
                print(f"Missing expected key in data: {e}")

        # seed = 42
        # random.seed(seed)
        # print(f"\n\n\nproblems size: {len(problems)} solutions size: {len(solutions)}\n\n\n")
        # problems = random.sample(problems, 2200)
        # solutions = random.sample(solutions, 2200)
        return {
            "problem": problems,
            "solution": solutions,
        }
    
    def generate_prompt(self, problem, tokenizer, **kwargs):
        r1_prefix = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.question_template.format(problem),
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True
            ),
            "problem": problem,
        }

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        # Load training data
        train_dict = self.read_files()
        # print(train_dict)
        train_dataset = Dataset.from_dict(train_dict)
        # print(f"{type(train_dataset)}")
        seed = 42
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=seed)
        # train_dataset = train_test_split["train"].unique(column="solution")
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        print(f"{type(test_dataset)} {type(train_dataset)}")
    
        # Combine into DatasetDict
        # print(train_dataset)
        self.dataset = DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )

        return self.dataset

    # def dataset_preprocess(self, tokenizer):
    #     self.dataset = self.dataset.map(
    #         lambda x: self.generate_prompt(x["problem"], tokenizer)
    #     )
    #     return self.dataset
    
    def accuracy_reward(self, completions, solution, prompts, **kwargs):
        """Reward function - check that completion is same as ground truth."""
        rewards = []

        for completion, prompt in zip(completions, prompts):
            reward = 0

            # Format
            think_start = completion.find("<think>")
            think_end = completion.find("</think>")
            answer_start = completion.find("<answer>")
            answer_end = completion.find("</answer>")

            if think_start != -1:
                reward += 0.25
            if think_end != -1:
                reward += 0.25
            if answer_start != -1:
                reward += 0.25
            if answer_end != -1:
                reward += 0.25
            
            if think_start != -1 and think_end != -1:
                if think_start < think_end:
                    reward += 0.25
            if answer_start != -1 and answer_end != -1:
                if answer_start < answer_end:
                    reward += 0.25
            if think_start != -1 and think_end != -1 and answer_start != -1 and answer_end != -1:
                if think_start < think_end and answer_start < answer_end and think_end < answer_start:
                    reward += 1
                else:
                    reward -= 1
            
            if completion.strip().endswith("</answer>"):
                reward += 1
            else:
                reward -= 2

            matches = re.findall(r"<think>(.*?)</think>", completion, flags=re.DOTALL)
            if matches:
                reasoning_len = len(matches[-1])
            else:
                reasoning_len = 0

            if reasoning_len < 500:
                reward -= 5

            # Extract elements from instruction
            input_pattern = r"(?i)elements:\s*(.*?)\s*,\s*propose\b"
            match = re.search(input_pattern, prompt)
            input_elements = match.group(1).split(', ') if match else []

            # Extract elements and space group from output
            output_pattern = r"<answer>\s*((?:[A-Z][a-z]?\s*)+?)\s*<sg(\d+)>\s*</answer>"
            output_matches = re.findall(output_pattern, completion)
            if len(output_matches) < 1:
                rewards.append(reward)
                # output = {
                #     "prompt": prompt,
                #     "input_elements": input_elements,
                #     "output_matches": output_matches,
                #     "reward": reward,
                #     "completion": completion
                # }
                # print(output)
                continue
            elif len(output_matches) == 1:
                reward += 1
            else:
                reward -= 1
            elements_str, sg_str = output_matches[-1]
            output_sg = int(sg_str.strip())

            if not 1 <= output_sg <= 230:
                rewards.append(reward)
                continue
                # output = {
                #     "prompt": prompt,
                #     "input_elements": input_elements,
                #     "output_matches": output_matches,
                #     "reward": reward,
                #     "completion": completion
                # }
                # print(output)
            reward += 1

            output_elements = elements_str.strip().split()

            # Penalize extra elements not in input
            extra_elements = set(output_elements) - set(input_elements)
            reward -= len(extra_elements) * 0.5
            # if extra_elements:
            #     reward -= len(extra_elements) * 0.5
            #     # Calculate overuse for extra elements
            #     overuse_score = sum(self.element_usage_counter.get(e, 0) for e in extra_elements)
            #     max_possible_overuse = len(extra_elements) * self.MAX_TRACKED
                
            #     # Normalize penalty between 0 and 2
            #     if max_possible_overuse > 0:
            #         normalized_penalty = 2 * (overuse_score / max_possible_overuse)
            #         reward -= normalized_penalty

            # # Penalize overused space group
            # overuse_score_sg = self.space_group_usage_counter.get(output_sg, 0)
            # max_possible_overuse_sg = self.MAX_TRACKED
            
            # if max_possible_overuse_sg > 0:
            #     normalized_penalty_sg = 2 * (overuse_score_sg / max_possible_overuse_sg)
            #     reward -= normalized_penalty_sg

            # Check precision
            intersection = set(input_elements) & set(output_elements)
            precision = len(intersection) / len(input_elements)
            if precision == 1:
                reward += 3
            else:
                reward += precision

            # Try building a composition after applying penalties
            try:
                comp = Composition(" ".join(output_elements))
                if not smact_validity(comp):  # your custom function
                    rewards.append(reward)
                    # output = {
                    #     "prompt": prompt,
                    #     "input_elements": input_elements,
                    #     "output_matches": output_matches,
                    #     "reward": reward,
                    #     "completion": completion
                    # }
                    # print(output)
                    continue
            except Exception as e:
                print(f"Invalid composition: {output_elements} -> {e}")
                rewards.append(reward)
                continue
            if precision == 1:
                reward += 3
            else:
                reward += 1

            # Novelty bonus
            if comp not in self.seen_comps_set:
                reward += 2
                self.seen_comps_set.add(comp)

            # # Update element and space group usage history
            # self.element_usage_counter.update(output_elements)
            # self.space_group_usage_counter.update([output_sg])
            # self.recent_compositions.append(output_elements)
            # self.recent_space_groups.append(output_sg)

            # # Maintain rolling window of last 100 outputs for compositions and space groups
            # if len(self.recent_compositions) > self.MAX_TRACKED:
            #     old_elements = self.recent_compositions.pop(0)
            #     old_space_group = self.recent_space_groups.pop(0)
            #     self.element_usage_counter.subtract(old_elements)
            #     self.space_group_usage_counter.subtract([old_space_group])
            #     # Remove zero or negative counts
            #     self.element_usage_counter += Counter()  # clean up
            #     self.space_group_usage_counter += Counter()  # clean up 

            self.random_log = {
                "prompt": prompt,
                "output_elements": output_elements,
                "output_sg": output_sg,
                "accuracy_reward": reward,
                "full_completion": completion,
            }
            self.good_print(self.random_log)
            # if reward > 4:
                # self.good_print(self.random_log)
            # else:
                # self.random_print(self.random_log)

            rewards.append(reward)
            self.custom_metrics['val/rewards'].extend(rewards)
        # print(rewards)
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
