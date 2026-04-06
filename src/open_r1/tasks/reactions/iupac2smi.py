import os
import re
from random import random
from typing import Any, Dict

import pandas as pd
from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ..base import RLTask, SMILESBasedTask
from .utils import tanimoto_score


class Iupac2Smiles(SMILESBasedTask):
    question_template: str = ""

    # custom_metrics: Dict[str, Any] = {}
    random_log: Dict[str, Any] = {}
    printed_sample_prompt: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.task_mode == "base":
            self.question_template = "<|im_start|>assistant\You are an useful chemistry assistant and you need to answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituents position and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags.<|im_end|>\n<|im_start|>user\Please write the SMILES representation of the molecule {}.<|im_end|>\n<|im_start|>assistant\Your response:\n<think> Okay"

        elif self.task_mode == "tagged":
            self.question_template = "<|im_start|>assistant\You are an useful chemistry assistant and you need to answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituents position and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags.<|im_end|>\n<|im_start|>user\Please write the SMILES representation of the molecule [START_MOL] {} [END_MOL].<|im_end|>\n<|im_start|>assistant\Your response:\n<think> Okay"

        elif self.task_mode == "tagged_no_okay":
            self.question_template = "<|im_start|>assistant\You are an useful chemistry assistant and you need to answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituents position and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags.<|im_end|>\n<|im_start|>user\Please write the SMILES representation of the molecule [START_MOL] {} [END_MOL].<|im_end|>\n<|im_start|>assistant\Your response:\n<think>"

        elif self.task_mode == "tagged_no_okay_short":
            self.question_template = "<|im_start|>assistant\You are an useful chemistry assistant and you need to answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituents position and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags. Keep the total length of your response, both reasoning and answer, no more than 4000 words.<|im_end|>\n<|im_start|>user\Please write the SMILES representation of the molecule [START_MOL] {} [END_MOL].<|im_end|>\n<|im_start|>assistant\Your response:\n<think>"

        elif self.task_mode == "qwen_base":

            self.question_template = (
                "<|im_start|>assistant\n"
                "You are an useful chemistry assistant and you need to answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituents position and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags.<|im_end|>\n"
                "<|im_start|>user\n"
                "Please write the SMILES representation of the molecule: {}.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>"
            )

        elif self.task_mode == "no_instruct":
            self.question_template = (
                "Question: You are an expert in Cheminformatics, who is very familiar with Simplified Molecular Input Line Entry System (SMILES) notation, and here's a task for you. "
                "Given a molecule with the IUPAC name as below, please provide the corresponding SMILES notation.\n"
                # "What is the SMILES for this molecule? {}. "
                "Here is the IUPAC name: {}.\n"
                # "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=C... </answer>. Think step by step inside <think> tags."
            )
        else:
            raise ValueError(f"Unknown task mode: {self.task_mode}. Supported modes: base, tagged, no_instruct")
        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

        # self.custom_metrics = {
        #     "n_samples": 0,
        #     "n_waits": [],
        #     "reasoning_score": [],
        #     "answer_scores": [],
        # }

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
        # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_dict = {
            "problem": df["IUPAC"].tolist(),
            "solution": df["SMILES"].tolist(),
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split = train_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Combine into DatasetDict
        self.dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
        return self.dataset

    def accuracy_reward(self, completions, solution, prompts, **kwargs):
        """Reward function - check that completion is same as ground truth."""

        #     return DataStructs.TanimotoSimilarity(fp1, fp2)

        # def _calc_score(mol1: str, mol2: str, beta=10):
        #     if (
        #         Chem.MolFromSmiles(mol1) is None
        #         or Chem.MolFromSmiles(mol2) is None
        #     ):
        #         return 0.0
        #     sim = tanimoto_sim(mol1, mol2)
        #     return sim**beta

        rewards = []

        for completion, ref, prompt in zip(completions, solution, prompts):
            # reasoning = completion.rsplit("<answer>", maxsplit=1)[0]
            # reasoning_smiles = self.extract_smiles(reasoning)
            # scores = [tanimoto_score(smi, ref) for smi in reasoning_smiles]
            # max_score = max(scores) if scores else -0.5
            # best_smiles_reasoning = (
            #     reasoning_smiles[scores.index(max_score)]
            #     if max_score in scores
            #     else "None"
            # )
            # reasoning_score = max_score
            # if reasoning_score == 1.0:
            #     reasoning_reward = 1.0
            # elif reasoning_score < 0:
            #     reasoning_reward = -1
            # else:
            #     reasoning_reward = -0.5

            answer = self.preprocess_response(completion)
            answer_smiles = self.extract_smiles_from_answer(answer)
            answer_score = tanimoto_score(answer_smiles, ref) if answer_smiles else -0.5
            if answer_score == 1.0:
                answer_reward = 1.0  # massive bonus for truly correct answer
            elif answer_score < 0:
                answer_reward = -1
            else:
                answer_reward = -0.5

            # reward = reasoning_reward + answer_reward
            reward = answer_reward

            answer_smiles = answer_smiles if answer_smiles else "None"
            self.random_log = {
                "prompt": prompt,
                "reference": ref,
                "answer": answer_smiles,
                # "best_smiles_in_reasoning": best_smiles_reasoning,
                # "reasoning_tanimoto_score [0, 1]": reasoning_score,
                "answer_tanimoto_score [0, 1]": answer_score,
                # "reasoning_reward [-0.5, 1]": reasoning_reward,
                # "answer_reward [-0.5, 1]": answer_reward,
                "accuracy_reward [-1, 1]": reward,
                "full_completion": completion,
            }

            if reward > 0.3:
                self.good_print(self.random_log)
            else:
                self.random_print(self.random_log)

            rewards.append(reward)

            self.custom_metrics["n_samples"] += 1
            self.custom_metrics["n_waits"].append(self.count_waits(completion))
            # self.custom_metrics["reasoning_reward"].append(reasoning_reward)
            self.custom_metrics["answer_reward"].append(answer_reward)
            # self.custom_metrics["reasoning_tanimoto"].append(reasoning_score)
            self.custom_metrics["answer_tanimoto"].append(answer_score)

        return rewards

    def reasoning_reward(self, completions, solution, prompts, **kwargs):
        """Reward function - check that reasoning is same as ground truth."""
        rewards = []

        for completion, ref, prompt in zip(completions, solution, prompts):
            reasoning = completion.rsplit("<answer>", maxsplit=1)[0]
            reasoning_smiles = self.extract_smiles(reasoning)
            scores = [tanimoto_score(smi, ref) for smi in reasoning_smiles]
            max_score = max(scores) if scores else -0.5
            best_smiles_reasoning = reasoning_smiles[scores.index(max_score)] if max_score in scores else "None"
            reasoning_score = max_score
            if reasoning_score == 1.0:
                reasoning_reward = 1.0
            elif reasoning_score < 0:
                reasoning_reward = -1
            else:
                reasoning_reward = -0.5

            self.custom_metrics["reasoning_reward"].append(reasoning_reward)
            self.custom_metrics["reasoning_tanimoto"].append(reasoning_score)

            rewards.append(reasoning_reward)

        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        if not response.startswith("<think>"):
            response = "<think>" + response
        answer_pattern = r"(?<=<answer>)(.*?)(?=<\/answer>)"
        answer = re.findall(answer_pattern, response)
        if answer:
            return answer[-1].strip()
        else:
            return "NONE"
        # pattern = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
        # m = re.search(pattern, response, re.DOTALL)
        # if m and len(m.groups()) == 2:
        #     return m.groups()[1]
        # else:
        #     return "NONE"

    # def get_metrics(self):
    #     return super().get_metrics()


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
