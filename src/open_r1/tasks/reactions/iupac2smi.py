import os
import re
import hashlib
import math
from random import random
from typing import Any, Dict

import pandas as pd
from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ..base import RLTask, SMILESBasedTask
from .utils import tanimoto_sim
from ..task_utils import compute_tanimoto_similarity


class Iupac2Smiles(SMILESBasedTask):
    question_template: str = ""

    custom_metrics: Dict[str, Any] = {}
    random_log: Dict[str, Any] = {}
    printed_sample_prompt: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "Question: You are an expert in Cheminformatics, who is very familiar with Simplified Molecular Input Line Entry System (SMILES) notation, and here's a task for you. "
            "Given a molecule with the IUPAC name as below, please provide the corresponding SMILES notation.\n"
            # "What is the SMILES for this molecule? {}. "
            "Here is the IUPAC name: {}.\n"
            # "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=C... </answer>. Think step by step inside <think> tags."
        )
        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv
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

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        df = df.drop_duplicates(subset=["SMILES"])
        train_dict = {
            "problem": df["IUPAC"].tolist(),
            "solution": df["SMILES"].tolist(),
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split = train_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Combine into DatasetDict
        self.dataset = DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )
        return self.dataset

    def accuracy_reward(self, completions, solution, prompts, **kwargs):
        """Reward function - check that completion is same as ground truth."""
        # def tanimoto_sim(mol1, mol2):
        #     mol1 = Chem.MolFromSmiles(mol1)
        #     mol2 = Chem.MolFromSmiles(mol2)

        #     fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, useChirality=True)
        #     fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, useChirality=True)

        #     return DataStructs.TanimotoSimilarity(fp1, fp2)

        def _calc_score(mol1: str, mol2: str, beta=10):
            if (
                Chem.MolFromSmiles(mol1) is None
                or Chem.MolFromSmiles(mol2) is None
            ):
                return 0.0
            sim = tanimoto_sim(mol1, mol2)
            return sim**beta

        rewards = []

        for completion, ref, prompt in zip(completions, solution, prompts):
            reasoning = completion.rsplit("<answer>", maxsplit=1)[0]
            reasoning_smiles = self.extract_smiles(reasoning)
            scores = [_calc_score(smi, ref) for smi in reasoning_smiles]
            max_score = max(scores) if scores else -0.5
            best_smiles_reasoning = (
                reasoning_smiles[scores.index(max_score)]
                if max_score in scores
                else "None"
            )
            reasoning_score = max_score
            if reasoning_score == 1.0:
                reasoning_score += (
                    1.0  # massive bonus for truly correct reasoning
                )

            answer = self.preprocess_response(completion)
            answer_smiles = self.extract_smiles_from_answer(answer)
            answer_score = (
                _calc_score(answer_smiles, ref) if answer_smiles else 0
            )
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

            self.random_print(self.random_log)
            if reward > 0.3:
                self.good_print(self.random_log)

            rewards.append(reward)

            self.custom_metrics["n_samples"] += 1
            self.custom_metrics["n_waits"].append(self.count_waits(completion))
            self.custom_metrics["reasoning_score"].append(reasoning_score)
            self.custom_metrics["answer_scores"].append(answer_score)

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
                    reward = (
                        tanimoto - 0.3
                    )  # Shifts the reward to be negative for very low similarities

                rewards.append(reward)

            except Exception as e:
                rewards.append(-1)
                continue

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

    def get_metrics(self):
        return super().get_metrics()


class Iupac2SmilesWithTags(Iupac2Smiles):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "Question: You are an expert in Cheminformatics, who is very familiar with Simplified Molecular Input Line Entry System (SMILES) notation, and here's a task for you. "
            "Given a molecule with the IUPAC name as below, please provide the corresponding SMILES notation.\n"
            # "What is the SMILES for this molecule? {}. "
            "Here is the IUPAC name: [START_MOL] {} [END_MOL].\n"
            # "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=C... </answer>. Think step by step inside <think> tags."
            "Your response: <think> "
        )

    def extract_smiles(self, completion: str):
        smiles = re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", completion)
        smiles = [s.replace(" ", "") for s in smiles]
        return smiles

class Iupac2SmilesV2(RLTask):
    question_template: str = ""
    printed_sample_prompt: bool = False
    tanimoto_coeff: float = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tanimoto_coeff = self.task_kwargs.get("tanimoto_coeff", 0.0)

        if self.task_mode == 'base':
            self.question_template = (
                "<|im_start|>assistant\n"
                "You are an useful chemistry assistant and you need to answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituents position and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags.<|im_end|>\n"
                "<|im_start|>user\n"
                "Please write the SMILES representation of the molecule: {}.<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>"
            )
        elif self.task_mode == 'promptBT':
            self.question_template = (
                "<|im_start|>assistant\n"
                "You are an useful chemistry assistant and you need to answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituents position and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags.<|im_end|>\n"
                "<|im_start|>user\n"
                "Please write the SMILES representation of the molecule [START_MOL] {} [END_MOL].<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>"
            )
        elif self.task_mode == 'promptP2': # Vu's prompt
            self.question_template = (
                "<|im_start|>assistant"
                "\You are an useful chemistry assistant and you need to answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituents position and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags.<|im_end|>\n"
                "<|im_start|>user"
                "\Please write the SMILES representation of the molecule [START_MOL] {} [END_MOL].<|im_end|>\n"
                "<|im_start|>assistant"
                "\<think>"
            )
        elif self.task_mode == 'promptP3': # Shai's prompt
            self.question_template = (
                "<|im_start|>assistant\You are a useful chemistry assistant and answer the question to change IUPAC to SMILES. Reason out your answer inside <think> tags and give your confident final answer inside the answer tags.<|im_end|>\n"
                "<|im_start|>user\What is the SMILES of [START_MOL] {} [END_MOL] ?\n"
                "Do only the necessary reasoning and backtracking to get to the final answer<|im_end|>\n"
                "<|im_start|>assistant\<think>"
            )
        else:
            raise ValueError(f"Unknown task mode: {self.task_mode}")

    def generate_prompt(self, problem, tokenizer, **kwargs):
        prompt = {
            "prompt": self.question_template.format(problem),
            "problem": problem,
        }

        if not self.printed_sample_prompt:  # print sample prompt once
            print(f"***SAMPLE PROMPT:\n{prompt['prompt']}")
            self.printed_sample_prompt = True

        return prompt

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        df = df.drop_duplicates(subset=["SMILES"])
        train_dict = {
            "problem": df["IUPAC"].tolist(),
            "solution": df["SMILES"].tolist(),
        }
        train_dataset = Dataset.from_dict(train_dict)

        train_test_split_seed = 42
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=train_test_split_seed)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        # Print hash of the first train example
        first_train_problem_hash = hashlib.sha256(train_dataset[0]['problem'].encode()).hexdigest()[:8]
        first_test_problem_hash = hashlib.sha256(test_dataset[0]['problem'].encode()).hexdigest()[:8]
        print(f"Iupac2SmilesV2 train_test_split shuffling seed: {train_test_split_seed}")
        print(f"First train problem hash: {first_train_problem_hash}")
        print(f"First test problem hash: {first_test_problem_hash}")

        # Combine into DatasetDict
        self.dataset = DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )
        return self.dataset

    def dataset_preprocess(self, tokenizer):
        self.dataset["train"] = (
            self.dataset["train"]
            .shuffle(seed=42)
            .select(range(min(100000, len(self.dataset["train"]))))
        )
        self.dataset["test"] = (
            self.dataset["test"]
            .shuffle(seed=42)
            .select(range(min(10000, len(self.dataset["test"]))))
        )

        self.dataset = self.dataset.map(
            lambda x: self.generate_prompt(x["problem"], tokenizer)
        )
        return self.dataset

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

    def tanimoto_tenth_reward(self, completions, solution, **kwargs):
        """Reward function - compute Tanimoto similarity reward between answer and solution."""
        # Reward goal: logging purpose
        # Reward range: 0 to 0.1

        answers = [self.preprocess_response(c).strip() for c in completions]

        rewards = []
        for answer, sol in zip(answers, solution):
            reward = 0.0
            try:
                tanimoto_similarity = compute_tanimoto_similarity(sol, answer)
                if tanimoto_similarity is not None:
                    reward = tanimoto_similarity / 10
            except:
                pass
            rewards.append(reward)

        return rewards

    def tanimoto_accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - compute Tanimoto similarity reward between answer and solution."""
        # Reward goal: foster answers with high Tanimoto similarity to the solution
        # Reward range: 0 to 1

        answers = [self.preprocess_response(c).strip() for c in completions]

        rewards = []
        for answer, sol, completion in zip(answers, solution, completions):
            reward = 0.0
            tanimoto_similarity = "Error"
            try:
                tanimoto_similarity = compute_tanimoto_similarity(sol, answer)
                if tanimoto_similarity is not None:
                    if self.tanimoto_coeff != 0.0:
                        base_coeff = math.e ** self.tanimoto_coeff
                        reward = (base_coeff ** tanimoto_similarity - 1) / (base_coeff - 1)
                    else:
                        reward = tanimoto_similarity
            except:
                pass
            rewards.append(reward)

            # Print
            print_proba = 0.01
            if reward >= 0.9:
                print_proba = print_proba * 5
            elif reward >= 0.5:
                print_proba = print_proba * 2
            if random() < print_proba:
                print("======= RANDOM_COMPLETION =======")
                print(f"Solution: {solution}")
                answer_formatted = answer.replace('\n',' ').replace('\t', ' ').replace('\r', '')[:128]
                print(f"Answer:   {answer_formatted}")
                if isinstance(tanimoto_similarity, (float, int)):
                    print(f"Tanimoto similarity: {tanimoto_similarity:.4f}")
                else:
                    print(f"Tanimoto similarity: {tanimoto_similarity}")
                print(f"Tanimoto reward:     {reward:.4f}")
                print(f"Completion:\n{completion}\n")

        return rewards

    def accuracy_percentage_reward(self, completions, solution, **kwargs):
        answers = [self.preprocess_response(c).strip() for c in completions]

        rewards = []
        for answer, sol in zip(answers, solution):
            reward = 0.0
            try:
                tanimoto_similarity = compute_tanimoto_similarity(sol, answer)
                if tanimoto_similarity is not None:
                    if tanimoto_similarity == 1.0:
                        reward = 1.0
                    else:
                        reward = 0.0
            except:
                pass
            rewards.append(reward)

        return rewards

    def preprocess_completion_before_answer(self, completion):
        if '<answer>' in completion:
            completion = completion.split('<answer>')[0]
        return completion

    def extract_smiles(self, text: str):
        smiles = re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", text)
        smiles = [s.replace(" ", "").strip() for s in smiles]
        return smiles

    def completion_tanimoto_tenth_reward(self, completions, solution, **kwargs):
        """Reward function - compute Tanimoto similarity reward between completion smiles and solution."""
        # Reward goal: logging purpose
        # Reward range: 0 to 0.1

        completions_smiles = [self.extract_smiles(self.preprocess_completion_before_answer(c)) for c in completions]

        rewards = []
        for completion_smiles, sol in zip(completions_smiles, solution):
            rewards_local = [0.0]
            for smiles in completion_smiles:
                try:
                    tanimoto_similarity = compute_tanimoto_similarity(sol, smiles)
                    if tanimoto_similarity is not None:
                        reward = tanimoto_similarity / 10
                        rewards_local.append(reward)
                except:
                    pass
            reward = max(rewards_local)
            rewards.append(reward)

        return rewards

    def completion_tanimoto_accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - compute Tanimoto similarity reward between completion smiles and solution."""
        # Reward goal: foster completions containing smiles with high Tanimoto similarity to the solution
        # Reward range: 0 to 0.5

        completions_smiles = [self.extract_smiles(self.preprocess_completion_before_answer(c)) for c in completions]

        rewards = []
        for completion_smiles, sol in zip(completions_smiles, solution):
            rewards_local = [0.0]
            for smiles in completion_smiles:
                try:
                    tanimoto_similarity = compute_tanimoto_similarity(sol, smiles)
                    if tanimoto_similarity is not None:
                        if self.tanimoto_coeff != 0.0:
                            base_coeff = math.e ** self.tanimoto_coeff
                            reward = (base_coeff ** tanimoto_similarity - 1) / (base_coeff - 1)
                        else:
                            reward = tanimoto_similarity
                        rewards_local.append(reward)
                except:
                    pass
            reward = max(rewards_local)
            rewards.append(reward)

        return rewards

