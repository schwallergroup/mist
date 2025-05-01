import os
import re
# from random import random
import random
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import exmol

from open_r1.download_data import download_data

from ..base import RLTask, SMILESBasedTask
from .utils import tanimoto_sim

def tanimoto_score(mol1: str, mol2: str, beta=1):
    if (
        Chem.MolFromSmiles(mol1) is None
        or Chem.MolFromSmiles(mol2) is None
    ):
        return 0.0
    sim = tanimoto_sim(mol1, mol2)
    return sim**beta

class ForwardReaction(SMILESBasedTask):
    src_train_file: str = ""
    tgt_train_file: str = ""
    src_test_file: str = ""
    tgt_test_file: str = ""
    question_template: str = ""

    custom_metrics: Dict[str, Any] = {}
    random_log: Dict[str, Any] = {}
    printed_sample_prompt: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if not os.path.exists(self.dataset_id_or_path):
        #     os.makedirs(self.dataset_id_or_path)
        # if not os.listdir(self.dataset_id_or_path):
        #     download_data(self.dataset_id_or_path)

        self.src_train_file = os.path.join(self.dataset_id_or_path, "src-train.txt")
        self.tgt_train_file = os.path.join(self.dataset_id_or_path, "tgt-train.txt")
        self.src_test_file = os.path.join(self.dataset_id_or_path, "src-test.txt") if "src-test.txt" else None
        self.tgt_test_file = os.path.join(self.dataset_id_or_path, "tgt-test.txt") if "tgt-test.txt" else None
       
        if self.task_mode == "base":
            # self.question_template = (
            #     "You are an organic chemistry expert, and I have a task for you. "
            #     "Given the following reagents in SMILES notation, please predict the most likely product(s) of the reaction between them. "
            #     "Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags. "
            #     "Here are the reagents: [START_SMILES] {} [END_SMILES]. "
            #     "Note that individual reagents are separated by a dot '.', and that some of them might just be observers.\n"
            # )
            self.question_template = "<|im_start|>assistant\You are an organic chemistry expert, and I have a task for you. Given the following reagents in SMILES notation, please predict the most likely product(s) of the reaction between them. Show your reasoning in <think>...</think> tags and return the final answer in <answer>...</answer> tags.<|im_end|>\n<|im_start|>user\Reason and predict the correct product in SMILES notation from the following reaction: {}.<|im_end|>\n<|im_start|>assistant\Response:\n<think>"
        elif self.task_mode == "tagged":
            self.question_template = "<|im_start|>assistant\You are an organic chemistry expert, and I have a task for you. Given the following reagents in SMILES notation, please predict the most likely product(s) of the reaction between them. Show your reasoning in <think>...</think> tags and return the final answer in <answer>...</answer> tags.<|im_end|>\n<|im_start|>user\Reason and predict the correct product in SMILES notation from the following reaction [START_SMILES] {} [END_SMILES].<|im_end|>\n<|im_start|>assistant\Response:\n<think>"
        
        elif self.task_mode == "fg_tagged":
            self.question_template = (
                "<|im_start|>assistant\You are an organic chemistry expert, and I have a task for you. Given the following reagents in SMILES notation, please predict the most likely product(s) of the reaction between them. Show your reasoning in <think>...</think> tags and return the final answer in <answer>...</answer> tags.<|im_end|>\n"
                "<|im_start|>user\Reason and predict the correct product in SMILES notation from the following reaction [START_SMILES] {} [END_SMILES]. As a hint, I also provide the functional group information of each molecule:\n\t{}\n"
                "Therefore, you don't have to parse the full structure of each molecule, instead focus on identifying which functional group(s) would react and edits the reactant SMILES accordingly to find the product.<|im_end|>\n"
                "<|im_start|>assistant\Response:\n"
                "<think>"
            )
            
        else:
            raise ValueError(f"Unknown task mode: {self.task_mode}")

        self.custom_metrics = {
            "n_samples": 0,
            "n_waits": [],
            # "reasoning_tanimoto_score": [],
            # "answer_scores": [],
            "reasoning_reward": [],
            "answer_reward": [],
            "reasoning_tanimoto": [],
            "answer_tanimoto": [],
        }
    def _question_template_format_with_fgs(self, reactants: str):
        reactant_list = reactants.split('.')
        reactant_with_fgs = []
        for reactant in reactant_list:
            fgs = exmol.get_functional_groups(reactant)
            if fgs:
                if 'tagged' in self.task_mode:
                    reactant_with_fgs.append(f'[START_SMILES] {reactant} [END_SMILES]: {", ".join(fgs)}')
                else:
                    reactant_with_fgs.append(f'{reactant}: {", ".join(fgs)}')
        reactant_with_fgs = '\n\t'.join(reactant_with_fgs)
        return self.question_template.format(reactants, reactant_with_fgs)
        
    
    def _extract_smiles_in_tags(self, completion: str):
        smiles = re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", completion)
        smiles = [s.replace(" ", "") for s in smiles]
        return smiles
    
    def question_template_format(self, reactants: str):
        if 'fg' in self.task_mode:
            return self._question_template_format_with_fgs(reactants)
        else:
            return self.question_template.format(reactants)
    
    def extract_smiles(self, completion: str):
        if self.task_mode == 'base':
            return super().extract_smiles(completion)
        elif 'tagged' in self.task_mode:
            return self._extract_smiles_in_tags(completion)

    def generate_prompt(self, problem, tokenizer, **kwargs):
        prompt = {
            "prompt": problem,
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
            problems = [
                self.question_template_format(self.process_line(line))
                # self.process_line(line)
                for line in f.readlines()
            ]

        with open(tgt_file, "r", encoding="utf-8") as f:
            solutions = [self.process_line(line) for line in f.readlines()]
        
        # randomly shuffle problems and solutions accordingly
        # shuffled_idx = list(range(len(problems)))
        # random.shuffle(shuffled_idx)
        # problems = [problems[i] for i in shuffled_idx]
        # solutions = [solutions[i] for i in shuffled_idx]

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
    
    def extract_smiles_from_response(self, reasoning: str, prompt: str=None):
        smiles = self.extract_smiles(reasoning)
        if prompt is None:
            return smiles
        smiles_prompt = self.extract_smiles(prompt)
        smiles_prompt_flatten = [s for smiles in smiles_prompt for s in smiles.split('.')]
        smiles_prompt.extend(smiles_prompt_flatten)
        smiles = [s for s in smiles if s not in smiles_prompt]
        return smiles

    def extract_smiles_from_answer(self, answer, prompt):
        """To prevent the longest smiles in answer turns out to be the copy of the starting reagents"""
        answer_smiles = self.extract_smiles_from_response(answer, prompt)
        # input_smiles = self.extract_smiles(prompt)
        # input_smiles_flatten = [s for smiles in input_smiles for s in smiles.split('.')]
        
        # answer_smiles = [s for s in answer_smiles if s not in input_smiles and s not in input_smiles_flatten]
        answer_smiles = max(answer_smiles, key=len) if answer_smiles else None
        return answer_smiles

    def accuracy_reward(self, completions, solution, prompts, **kwargs):
        """Reward function - check that completion is same as ground truth."""

        

        rewards = []

        for completion, ref, prompt in zip(completions, solution, prompts):
            reasoning = completion.rsplit("<answer>", maxsplit=1)[0]
            reasoning_smiles = self.extract_smiles_from_response(reasoning, prompt)
            reasoning_tanimoto_scores = [tanimoto_score(smi, ref) for smi in reasoning_smiles]
            max_reasoning_tanimoto_score = max(reasoning_tanimoto_scores) if reasoning_tanimoto_scores else -0.5
            best_smiles_reasoning = (
                reasoning_smiles[reasoning_tanimoto_scores.index(max_reasoning_tanimoto_score)]
                if max_reasoning_tanimoto_score in reasoning_tanimoto_scores
                else "None"
            )
            reasoning_tanimoto_score = max_reasoning_tanimoto_score
            if reasoning_tanimoto_score == 1.0:
                # reasoning_tanimoto_score += (
                #     1.0  # massive bonus for truly correct reasoning
                # )
                reasoning_reward = 1.0
            elif reasoning_tanimoto_score < 0:
                reasoning_reward = -1  # no SMILES found
            else:
                reasoning_reward = -0.5 # SMILES found but not correct

            answer = self.preprocess_response(completion)
            answer_smiles = self.extract_smiles_from_answer(answer, prompt)
            answer_tanimoto_score = (
                tanimoto_score(answer_smiles, ref) if answer_smiles else -0.5
            )
            if answer_tanimoto_score == 1.0:
                # answer_tanimoto_score += 1.0  # massive bonus for truly correct answer
                answer_reward = 1.0
            elif answer_tanimoto_score < 0:
                answer_reward = -1
            else:
                answer_reward = -0.5

            reward = reasoning_reward + answer_reward

            answer_smiles = answer_smiles if answer_smiles else "None"
            self.random_log = {
                "prompt": prompt,
                "reference": ref,
                "answer": answer_smiles,
                "best_smiles_in_reasoning": best_smiles_reasoning,
                "reasoning_tanimoto_score [0, 1]": reasoning_tanimoto_score,
                "answer_tanimoto_score [0, 1]": answer_tanimoto_score,
                "reasoning_reward [-0.5, 1]": reasoning_reward,
                "answer_reward [-0.5, 1]": answer_reward,
                "accuracy_reward [-1, 2]": reward,
                "full_completion": completion,
            }

            if reward > 0.3:
                self.good_print(self.random_log)
            else:
                self.random_print(self.random_log)

            rewards.append(reward)

            self.custom_metrics["n_samples"] += 1
            self.custom_metrics["n_waits"].append(self.count_waits(completion))
            self.custom_metrics["reasoning_reward"].append(reasoning_reward)
            self.custom_metrics["answer_reward"].append(answer_reward)
            self.custom_metrics["reasoning_tanimoto"].append(reasoning_tanimoto_score)
            self.custom_metrics["answer_tanimoto"].append(answer_tanimoto_score)

        return rewards

    def get_metrics(self):
        metrics = {}
        if self.custom_metrics['n_samples'] > 0:
            metrics['n_samples'] = self.custom_metrics['n_samples']
            for k, v in self.custom_metrics.items():
                if k != 'n_samples':
                    metrics[k] = sum(v) / len(v)
                    self.custom_metrics[k] = []
        
        return metrics

class ForwardReactionWithTags(ForwardReaction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.question_template = (
        #     "<|im_start|>You are an organic chemistry expert, and I have a task for you. "
        #     "Given the following reagents in SMILES notation, please predict the most likely product(s) of the reaction between them. "
        #     "Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags. "
        #     "Here are the reactants: [START_SMILES] {} [END_SMILES]. "
        #     "Note that individual reagents are separated by a dot '.', and that some of them might just be observers.\n"
        #     "Your response: <think> "
        # )
        self.question_template = "<|im_start|>assistant\You are an organic chemistry expert, and I have a task for you. Given the following reagents in SMILES notation, please predict the most likely product(s) of the reaction between them. Show your reasoning in <think>...</think> tags and return the final answer in <answer>...</answer> tags.<|im_end|>\n<|im_start|>user\Reason and predict the correct product in SMILES notation from the following reaction [START_SMILES] {} [END_SMILES].<|im_end|>\n<|im_start|>assistant\Response:\n<think>"

    def extract_smiles(self, completion: str):
        smiles = re.findall(r"\[START_SMILES\](.*?)\[END_SMILES\]", completion)
        smiles = [s.replace(" ", "") for s in smiles]
        return smiles
