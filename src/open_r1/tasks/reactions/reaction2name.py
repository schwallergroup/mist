
from ..base import RLTask
from typing import Dict
import re
import os
import hashlib
from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from random import random
import difflib
from dataclasses import field

class Smiles2Name(RLTask):
    question_template: str = ""
    printed_sample_prompt: bool = False

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        printed_sample_prompt: bool = False
        self.question_template = ("""
            <|im_start|>assistant
            You are a useful Chemistry assistant and you will answer the following class prediction question. Give your reasoning inside the <think>...</think> tags and then respond inside <answer>...</answer> tags, think and reason for all the options before giving your answer. Structure your reasoning such that you think through all options before giving the answer.<|im_end|>
            
            <|im_start|>user
            Question: What is the name of this chemical reaction ? {} Choose ONLY from the following options and write your response choice inside <answer>...</answer>: [Acylation, Aromatic Heterocycle Formation, C-C Coupling, Deprotection, Functional Group Addition, Functional Group Interconversion, Heteroatom Alkylation and Arylation, Miscellaneous, Protection, Reduction]. Do not provide final answer different than what it provided in this list. 
            <|im_end|>

            <|im_start|>assistant
            <think>"""
        )

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        train_dict = {
            'problem': df['REACTION_PROMPT'].tolist(),
            'solution': df['CLASS'].tolist()
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split_seed = 42
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=train_test_split_seed)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        # Print hash of the first train example
        first_train_problem_hash = hashlib.sha256(train_dataset[0]['problem'].encode()).hexdigest()[:8]
        first_test_problem_hash = hashlib.sha256(test_dataset[0]['problem'].encode()).hexdigest()[:8]
        print(f"Smiles2Name train_test_split shuffling seed: {train_test_split_seed}")
        print(f"First train problem hash: {first_train_problem_hash}")
        print(f"First test problem hash: {first_test_problem_hash}")
        
        # Combine into DatasetDict
        self.dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
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
            remove_columns=["problem"]
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
                prefix = c[:len(c) - len(clean)]
                processed.append(f"{prefix}<think>{clean}")
            else:
                processed.append(c)
        return processed

    
    def accuracy_reward(self, completions, **kwargs):

        completions = self.preprocess_completions(completions)

        valid_choices = ['Acylation', 'Aromatic Heterocycle Formation', 'C-C Coupling', 'Deprotection', 'Functional Group Addition', 'Functional Group Interconversion', 'Heteroatom Alkylation and Arylation', 'Miscellaneous', 'Protection', 'Reduction']

        solution = kwargs.get("solution", [])

        lc_choices = [c.lower() for c in valid_choices]
        rewards = []

        for completion, gold in zip(completions, solution):
            answers = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            answers = [ans.strip() for ans in answers]

            selected = answers[0] if answers else ""

            reward = 0.0

            if any(ans.lower() == gold.lower() for ans in answers):
                reward = 1.0 / len(answers)
                if random() < 0.1:
                    print("======= FULL_COMPLETION_CORRECT =======")
                    print(f"All Completion:   {completion}")
                    print(f"All answers:   {answers}")
                    print(f"Selected ans:  {selected!r}")
                    print(f"Ground truth:  {gold!r}\n")

            else:
                matching = [ans for ans in answers if ans.lower() in lc_choices]
                if matching:
                    reward = 0.2
                    if random() < 0.02:
                        print("======= FULL_COMPLETION_VALID_CHOICE =======")
                        print(f"All Completion:   {completion}")
                        print(f"All answers:       {answers}")
                        print(f"Valid choices hit: {matching}")
                        print(f"Selected ans:      {selected!r}")
                        print(f"Ground truth:      {gold!r}\n")

                else:
                    reward = -0.2
                    if random() < 0.01:
                        print("======= FULL_COMPLETION_INVALID =======")
                        print(f"All Completion:   {completion}")
                        print(f"All answers:  {answers}")
                        print(f"Selected ans: {selected!r}")
                        print(f"Ground truth: {gold!r}\n")
            
            # Reward if mentioning the correct classes
            tm = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
            if tm:
                think_text = tm.group(1).lower()
                if any(choice in think_text for choice in lc_choices):
                    reward += 0.1
                    print("  (+0.1 bonus: reasoning mentioned a valid choice)")

            rewards.append(reward)

        return rewards

    def accuracy_percentage_reward(self, completions, **kwargs):
        completions = self.preprocess_completions(completions)
        completions = [c.lower() for c in completions]

        solutions = kwargs.get("solution", [])
        solutions = [sol.lower() for sol in solutions]

        rewards = []

        for completion, solution in zip(completions, solutions):
            # Parse answer
            answers = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            answers = [ans.strip() for ans in answers]
            answer = answers[0] if answers else ""

            # Reward answer
            if answer == solution:
                reward = 1.0
            else:
                reward = 0.0
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
                if random.random() < 0.00:  # 0% chance to print a completion
                    print(f"\n\n=======<RANDOM_RESPONSE>=======\n{completion}")
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

    def format_continuous_reward(
        self, completions, **kwargs
    ):
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers

        Returns:
            list[float]: Reward scores
        """
        # Reward goal: ensure a good format
        # Reward range: -1 to 1

        completions = self.preprocess_completions(completions)
        rewards = []

        for completion_id, completion in enumerate(completions):
            current_reward = 0.0
            try:
                # 0.2 reward if each tag is present once
                for tag_word in [
                    "<think>",
                    "</think>",
                    "<answer>",
                    "</answer>",
                ]:
                    if completion.count(tag_word) == 1:
                        current_reward += 0.05
                    else:
                        current_reward -= 0.05
                # 0.1 reward if the completion starts with <think> and ends with </answer>
                if completion.startswith("<think>"):
                    current_reward += 0.05
                else:
                    current_reward -= 0.05
                if completion.endswith("</answer>"):
                    current_reward += 0.05
                else:
                    current_reward -= 0.05
                # 0.1 reward if the thinking is followed by an answer
                if completion.count("</think>\n<answer>") == 1:
                    current_reward += 0.1
                else:
                    current_reward -= 0.1
                # 0.2 reward is the answer is present
                pattern = r"<answer>(.*)<\/answer>"
                match = re.search(pattern, completion, re.DOTALL)
                if match is None:
                    current_reward -= 0.2
                elif len(match.groups()) != 1:
                    current_reward -= 0.05
                else:
                    current_reward += 0.2
                # 0.4 reward is the format is correct
                pattern = r"<think>(.*)<\/think>\n<answer>(.*)<\/answer>"
                match = re.search(pattern, completion, re.DOTALL)
                if match is None:
                    current_reward -= 0.4
                elif len(match.groups()) != 2:
                    current_reward -= 0.1
                else:
                    current_reward += 0.4
            except:
                pass
            rewards.append(current_reward)

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

class Smiles2NameV2(Smiles2Name):
    answer_history: dict = field(default_factory=dict)
    completion_hash_history: list = field(default_factory=list)
    answer_history_buffer_size: int = 10
    answer_history_penalty: float = 0.4
    accuracy_give_penalty_to_invalid_answer: bool = False
    completion_hash_history_buffer_size: int = 1024
    completion_hash_history_penalty: float = 2.0
    completion_hash_history_max_repetitions: int = 2
    completion_size_penalty: float = 1.0
    completion_size_minimum: int = 50
    completion_shortening_penalty: float = 10.0
    completion_shortening_interval_high: int = 25000
    completion_shortening_interval_low: int = 1000
    train_test_split_seed: int = 42

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.answer_history = dict()
        self.completion_hash_history = []

        self.answer_history_buffer_size = self.task_kwargs.get("answer_history_buffer_size", 10)
        self.answer_history_penalty = self.task_kwargs.get("answer_history_penalty", 0.4)
        self.completion_hash_history_buffer_size = self.task_kwargs.get("completion_hash_history_buffer_size", 1024)
        self.completion_hash_history_penalty = self.task_kwargs.get("completion_hash_history_penalty", 2.0)
        self.completion_hash_history_max_repetitions = self.task_kwargs.get("completion_hash_history_max_repetitions", 2)
        self.completion_size_penalty = self.task_kwargs.get("completion_size_penalty", 1.0)
        self.completion_size_minimum = self.task_kwargs.get("completion_size_minimum", 50)
        self.accuracy_give_penalty_to_invalid_answer = self.task_kwargs.get("accuracy_give_penalty_to_invalid_answer", False)
        self.completion_shortening_penalty = self.task_kwargs.get("completion_shortening_penalty", 10.0)
        self.completion_shortening_interval_high = self.task_kwargs.get("completion_shortening_interval_high", 25000)
        self.completion_shortening_interval_low = self.task_kwargs.get("completion_shortening_interval_low", 1000)
        self.train_test_split_seed = self.task_kwargs.get("train_test_split_seed", 42)

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        train_dict = {
            'problem': df['REACTION_PROMPT'].tolist(),
            'solution': df['CLASS'].tolist()
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split_seed = self.train_test_split_seed
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=train_test_split_seed)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        # Print hash of the first train example
        first_train_problem_hash = hashlib.sha256(train_dataset[0]['problem'].encode()).hexdigest()[:8]
        first_test_problem_hash = hashlib.sha256(test_dataset[0]['problem'].encode()).hexdigest()[:8]
        print(f"Smiles2Name train_test_split shuffling seed: {train_test_split_seed}")
        print(f"First train problem hash: {first_train_problem_hash}")
        print(f"First test problem hash: {first_test_problem_hash}")

        # Combine into DatasetDict
        self.dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        return self.dataset

    def accuracy_reward(self, completions, **kwargs):
        completions = self.preprocess_completions(completions)
        completions = [c.lower() for c in completions]

        valid_choices = ['Acylation', 'Aromatic Heterocycle Formation', 'C-C Coupling', 'Deprotection',
                         'Functional Group Addition', 'Functional Group Interconversion',
                         'Heteroatom Alkylation and Arylation', 'Miscellaneous', 'Protection', 'Reduction']
        valid_choices = [c.lower() for c in valid_choices]
        for choice in valid_choices:
            if choice not in self.answer_history.keys():
                self.answer_history[choice] = []
        def _get_choice_counts(text, choices):
            """
            Count the occurrences of each choice in the text.
            """
            counts = {choice: text.count(choice) for choice in choices}
            return counts

        solutions = kwargs.get("solution", [])
        solutions = [sol.lower() for sol in solutions]

        rewards = []

        for completion, solution in zip(completions, solutions):
            reward = 0.0
            info_correct_solution = False
            reward_answer_penalty = None
            info_correct_think = False
            info_answer_given = "__unknown"

            # Parse answer
            answers = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            answers = [ans.strip() for ans in answers]
            answer = answers[0] if answers else ""
            answer_counts = _get_choice_counts(answer, valid_choices)
            answer_counts = {k: v for k, v in answer_counts.items() if v > 0}
            n_answers = sum(answer_counts.values())

            # Reward answer
            # if 1 < n_answers <= 3:
            #     if solution in answer_counts.keys():
            #         reward += 0.2 / (n_answers - 1)
            if n_answers == 1:
                answer_given = list(answer_counts.keys())[0]
                info_answer_given = answer_given
                reward += 0.1
                if answer_given == solution:
                    reward += 0.6
                    info_correct_solution = True
                else:
                    reward_answer_penalty = -(self.answer_history[solution].count(answer_given) / max(len(self.answer_history[solution]), 1)) * self.answer_history_penalty
                    reward += reward_answer_penalty
                if len(self.answer_history[solution]) >= self.answer_history_buffer_size:
                    self.answer_history[solution].pop(0)
                self.answer_history[solution].append(answer_given)
            else:
                if self.accuracy_give_penalty_to_invalid_answer:
                    reward -= self.answer_history_penalty

            # Parse & reward think
            think_matches = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
            if think_matches:
                think_text = think_matches.group(1)
                think_text_counts = _get_choice_counts(think_text, valid_choices)
                think_text_max_count = max(think_text_counts.values())
                think_text_max_counts = {k: v for k, v in think_text_counts.items() if v == think_text_max_count}
                if len(think_text_max_counts) == 1:
                    if list(think_text_max_counts.keys())[0] == solution:
                        reward += 0.3
                        info_correct_think = True

            rewards.append(reward)

            # Print
            print_proba = 0.01
            if n_answers == 1:
                print_proba = print_proba * 5
            if random() < print_proba:
                print("======= RANDOM_COMPLETION =======")
                print(f"Solution: {solution}")
                answer_formatted = answer.replace('\n',' ').replace('\t', ' ').replace('\r', '')[:128]
                print(f"Answer:   {answer_formatted}")
                print(f"Solution buffer: {self.answer_history[solution]}")
                current_buffer_size = min(len(self.answer_history[solution]), self.answer_history_buffer_size)
                print(f"    Correct solution: {self.answer_history[solution].count(solution)}/{current_buffer_size}")
                if info_answer_given != "__unknown":
                    print(f"    This answer: {self.answer_history[solution].count(info_answer_given)}/{current_buffer_size}")
                else:
                    print(f"    This answer: unknown")
                print(f"Accuray reward: {reward:.4f}")
                print(f"    n_answers: {n_answers}{' (+0.1)' if n_answers == 1 else ''}")
                print(f"    correct solution: {info_correct_solution}{' (+0.6)' if info_correct_solution else ''}")
                if reward_answer_penalty is None:
                    print(f"    answer penalty: {reward_answer_penalty}")
                else:
                    print(f"    answer penalty: {reward_answer_penalty:.4f}")
                print(f"    correct think: {info_correct_think}{' (+0.3)' if info_correct_think else ''}")
                print(f"Completion:\n{completion}\n")

        return rewards

    def completion_difference_reward(self, completions, **kwargs):
        rewards = []

        for completion_id, completion in enumerate(completions):
            reward = 0.0
            completion_hash = hashlib.sha256(completion.encode()).hexdigest()[:8]

            # Give penalty if the completion is a repetition
            if self.completion_hash_history.count(completion_hash) >= self.completion_hash_history_max_repetitions:
                reward = -self.completion_hash_history_penalty

            # Update hash history
            self.completion_hash_history.append(completion_hash)
            if len(self.completion_hash_history) > self.completion_hash_history_buffer_size:
                self.completion_hash_history.pop(0)

            rewards.append(reward)

        return rewards

    def completion_size_reward(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            reward = 0.0
            size = len(completion)
            if size < self.completion_size_minimum:
                reward = -self.completion_size_penalty * (self.completion_size_minimum - size) / self.completion_size_minimum
            rewards.append(reward)
        return rewards

    def format_continuous_negative_reward(self, completions, **kwargs):
        # Initial range: [-1, 1]
        rewards = super().format_continuous_reward(completions, **kwargs)
        # Final range: [-1, -0.5]
        rewards = [(r - 3) / 4 for r in rewards]
        return rewards

    def completion_shortening_reward(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            reward = 0.0
            size = len(completion)
            if size > self.completion_shortening_interval_high:
                reward = -self.completion_shortening_penalty
            elif size > self.completion_shortening_interval_low:
                reward = -self.completion_shortening_penalty * (size - self.completion_shortening_interval_low) / (self.completion_shortening_interval_high - self.completion_shortening_interval_low)
            rewards.append(reward)
        return rewards

