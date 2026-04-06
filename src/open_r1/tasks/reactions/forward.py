import os
import re
from random import random
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from open_r1.download_data import download_data
from open_r1.paths import expand_path

from ..base import RLTask, SMILESBasedTask
from .utils import tanimoto_sim


class ForwardReaction(SMILESBasedTask):
    src_train_file: str = ""
    tgt_train_file: str = ""
    src_test_file: str = ""
    tgt_test_file: str = ""
    question_template: str = ""

    custom_metrics: Dict[str, Any] = {}
    random_log: Dict[str, Any] = {}
    printed_sample_prompt: bool = False

    @staticmethod
    def _has_local_reaction_files(dataset_dir: str) -> bool:
        required_files = [
            "src-train.txt",
            "tgt-train.txt",
            "src-test.txt",
            "tgt-test.txt",
        ]
        return all(os.path.exists(os.path.join(dataset_dir, filename)) for filename in required_files)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_id_or_path = expand_path(self.dataset_id_or_path)
        if not os.path.exists(self.dataset_id_or_path):
            os.makedirs(self.dataset_id_or_path)
        if not self._has_local_reaction_files(self.dataset_id_or_path):
            download_data(self.dataset_id_or_path)

        self.src_train_file = os.path.join(self.dataset_id_or_path, "src-train.txt")
        self.tgt_train_file = os.path.join(self.dataset_id_or_path, "tgt-train.txt")
        self.src_test_file = os.path.join(self.dataset_id_or_path, "src-test.txt") if "src-test.txt" else None
        self.tgt_test_file = os.path.join(self.dataset_id_or_path, "tgt-test.txt") if "tgt-test.txt" else None
        self.question_template = (
            "You are an organic chemistry expert, and I have a task for you. "
            "Given the following reagents in SMILES notation, please predict the most likely product(s) of the reaction between them. "
            "Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags. "
            "Here are the reagents: {}. "
            "Note that individual reagents are separated by a dot '.', and that some of them might just be observers.\n"
        )

        self.custom_metrics = {
            "n_samples": 0,
            "n_waits": [],
            "reasoning_score": [],
            "answer_scores": [],
        }

    def generate_prompt(self, problem, tokenizer, **kwargs):
        prompt = {
            "prompt": self.question_template.format(problem),
            "problem": problem,
        }

        if not self.printed_sample_prompt:  # print sample prompt once
            print(f"***SAMPLE PROMPT:\n{prompt['prompt']}")
            self.printed_sample_prompt = True

        return prompt

    def process_line(self, line: str) -> str:
        """Process a line from the source file."""
        line = re.sub(r" +", "", line)
        line = line.strip()
        return line

    def read_files(self, src_file: str, tgt_file: str) -> Dict:
        """Read source and target files and create dataset dictionary."""
        with open(src_file, "r", encoding="utf-8") as f:
            problems = [self.question_template.format(self.process_line(line)) for line in f.readlines()]

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
        self.dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

        return self.dataset

    def extract_smiles_from_answer(self, answer, prompt):
        """To prevent the longest smiles in answer turns out to be the copy of the starting reagents"""
        answer_smiles = self.extract_smiles(answer)
        input_smiles = self.extract_smiles(prompt)

        answer_smiles = [s for s in answer_smiles if s not in input_smiles]
        answer_smiles = max(answer_smiles, key=len) if answer_smiles else None
        return answer_smiles

    def accuracy_reward(self, completions, solution, prompts, **kwargs):
        """Reward function - check that completion is same as ground truth."""

        def _calc_score(mol1: str, mol2: str, beta=20):
            if Chem.MolFromSmiles(mol1) is None or Chem.MolFromSmiles(mol2) is None:
                return 0.0
            sim = tanimoto_sim(mol1, mol2)
            return sim**beta

        rewards = []

        for completion, ref, prompt in zip(completions, solution, prompts):
            reasoning = completion.rsplit("<answer>", maxsplit=1)[0]
            reasoning_smiles = self.extract_smiles(reasoning)
            scores = [_calc_score(smi, ref) for smi in reasoning_smiles]
            max_score = max(scores) if scores else -0.5
            best_smiles_reasoning = reasoning_smiles[scores.index(max_score)] if max_score in scores else "None"
            reasoning_score = max_score
            if reasoning_score == 1.0:
                reasoning_score += 1.0  # massive bonus for truly correct reasoning

            answer = self.preprocess_response(completion)
            answer_smiles = self.extract_smiles_from_answer(answer, prompt)
            answer_score = _calc_score(answer_smiles, ref) if answer_smiles else 0
            if answer_score == 1.0:
                answer_score += 1.0  # massive bonus for truly correct answer

            reward = reasoning_score + answer_score

            answer_smiles = answer_smiles if answer_smiles else "None"
            self.random_log = {
                "prompt": prompt,
                "reference": ref,
                "answer": answer_smiles,
                "best_smiles_in_reasoning": best_smiles_reasoning,
                "reasoning_score [0, 2]": reasoning_score,
                "answer_score [0, 2]": answer_score,
                "accuracy_reward [0, 4]": reward,
                "full_completion": completion,
            }

            if reward > 0.3:
                self.good_print(self.random_log)
            else:
                self.random_print(self.random_log)

            rewards.append(reward)

            self.custom_metrics["n_samples"] += 1
            self.custom_metrics["n_waits"].append(self.count_waits(completion))
            self.custom_metrics["reasoning_score"].append(reasoning_score)
            self.custom_metrics["answer_scores"].append(answer_score)

        return rewards


class ForwardReactionWithTags(ForwardReaction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "<|im_start|>You are an organic chemistry expert, and I have a task for you. "
            "Given the following reagents in SMILES notation, please predict the most likely product(s) of the reaction between them. "
            "Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags. "
            "Here are the reactants: [START_SMILES] {} [END_SMILES]. "
            "Note that individual reagents are separated by a dot '.', and that some of them might just be observers.\n"
            "Your response: <think> "
        )

    def extract_smiles(self, completion: str):
        smiles = re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", completion)
        smiles = [s.replace(" ", "") for s in smiles]
        return smiles
