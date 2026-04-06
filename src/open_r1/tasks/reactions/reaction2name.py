import difflib
import os
import re
from random import random
from typing import Dict

import pandas as pd
from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from open_r1.paths import expand_path

from ..base import RLTask


class Smiles2Name(RLTask):
    question_template: str = ""
    printed_sample_prompt: bool = False

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        printed_sample_prompt: bool = False
        self.question_template = """
            <|im_start|>assistant
            You are a useful Chemistry assistant and you will answer the following class prediction question. Give your reasoning inside the <think>...</think> tags and then respond inside <answer>...</answer> tags, think and reason for all the options before giving your answer. Structure your reasoning such that you think through all options before giving the answer.<|im_end|>
            
            <|im_start|>user
            Question: What is the name of this chemical reaction ? {} Choose ONLY from the following options and write your response choice inside <answer>...</answer>: [Acylation, Aromatic Heterocycle Formation, C-C Coupling, Deprotection, Functional Group Addition, Functional Group Interconversion, Heteroatom Alkylation and Arylation, Miscellaneous, Protection, Reduction]. Do not provide final answer different than what it provided in this list. 
            <|im_end|>

            <|im_start|>assistant
            <think>"""

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(expand_path(self.dataset_id_or_path))
        train_dict = {
            "problem": df["REACTION_PROMPT"].tolist(),
            "solution": df["CLASS"].tolist(),
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split = train_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Combine into DatasetDict
        self.dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
        return self.dataset

    def generate_prompt(self, problem, tokenizer, **kwargs):
        prompt = {
            "prompt": self.question_template.format(problem),
            "problem": problem,
        }

        if not self.printed_sample_prompt:  # print sample prompt once
            print(f"***SAMPLE PROMPT:\n{prompt['prompt']}")
            self.printed_sample_prompt = True

        return prompt

    def dataset_preprocess(self, tokenizer=None):
        # only drop the raw "problem" column—keep "solution" so reward_fn gets it
        self.dataset = self.dataset.map(
            lambda ex: self.generate_prompt(ex["problem"], tokenizer),
            remove_columns=["problem"],
        )
        return self.dataset

    def preprocess_completions(self, completions: list[str]) -> list[str]:
        """
        Ensure each completion string starts with a <think> tag.
        If it already does, leave it as-is; otherwise prepend '<think>'.
        """
        processed = []
        for c in completions:
            clean = c.lstrip()
            if not clean.startswith("<think>"):
                prefix = c[: len(c) - len(clean)]
                processed.append(f"{prefix}<think>{clean}")
            else:
                processed.append(c)
        return processed

    def accuracy_reward(self, completions, **kwargs):

        completions = self.preprocess_completions(completions)

        valid_choices = [
            "Acylation",
            "Aromatic Heterocycle Formation",
            "C-C Coupling",
            "Deprotection",
            "Functional Group Addition",
            "Functional Group Interconversion",
            "Heteroatom Alkylation and Arylation",
            "Miscellaneous",
            "Protection",
            "Reduction",
        ]

        solution = kwargs.get("solution", [])

        lc_choices = [c.lower() for c in valid_choices]
        rewards = []

        for completion, gold in zip(completions, solution):
            answers = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            answers = [ans.strip() for ans in answers]

            selected = answers[0] if answers else ""

            reward = 0.0

            if any(ans.lower() == gold.lower() for ans in answers):
                reward = 1.0
                print("======= FULL_COMPLETION_CORRECT =======")
                print(f"All Completion:   {completion}")
                print(f"All answers:   {answers}")
                print(f"Selected ans:  {selected!r}")
                print(f"Ground truth:  {gold!r}\n")

            else:
                matching = [ans for ans in answers if ans.lower() in lc_choices]
                if matching:
                    reward = 0.2
                    print("======= FULL_COMPLETION_VALID_CHOICE =======")
                    print(f"All Completion:   {completion}")
                    print(f"All answers:       {answers}")
                    print(f"Valid choices hit: {matching}")
                    print(f"Selected ans:      {selected!r}")
                    print(f"Ground truth:      {gold!r}\n")

                else:
                    reward = -0.2

                    if random() < 0.05:
                        print("======= FULL_COMPLETION_INVALID =======")
                        print(f"All Completion:   {completion}")
                        print(f"All answers:  {answers}")
                        print(f"Selected ans: {selected!r}")
                        print(f"Ground truth: {gold!r}\n")

            # Reward if metnioning the correct classes
            tm = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
            if tm:
                think_text = tm.group(1).lower()
                if any(choice in think_text for choice in lc_choices):
                    reward += 0.1
                    print("  (+0.1 bonus: reasoning mentioned a valid choice)")

            rewards.append(reward)

        return rewards

    def format_reward(self, completions, **kwargs):
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers

        Returns:
            list[float]: Reward scores
        """

        rewards = []

        for completion in completions:
            try:
                if not completion.startswith("<think>"):
                    completion = "<think>" + completion
                regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
                match = re.search(regex, completion, re.DOTALL)
                # if the format is not correct, reward is 0
                if match is None or len(match.groups()) != 2:
                    rewards.append(0.0)
                else:
                    # The model tends to generate gibberish outside of the tags
                    reward = len(match.group()) / len(completion)
                    rewards.append(reward)
            except Exception:
                rewards.append(0.0)
        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        if not response.startswith("<think>"):
            response = "<think>" + response
        pattern = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m and len(m.groups()) == 2:
            return m.groups()[1]
        else:
            return "NONE"
