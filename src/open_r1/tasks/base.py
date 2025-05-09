"""Base task definition.
A task needs to be initialized with a dataset
and habe fllowing methods: load() -> for dataset loading, and two functions for the rewards
format_reward, accuracy_reward
"""

import os
import random
import re
from dataclasses import field
from typing import Any, Dict, Optional

from datasets import load_dataset
from rdkit import Chem, RDLogger

from pydantic import BaseModel, Field

RDLogger.DisableLog("rdApp.*")


class RLTask(BaseModel):
    """
    Reinforcement Learning Task configuration class.

    This class handles the configuration for RL tasks, including dataset specifications
    and prompt formatting settings.

    Attributes:
        dataset_id_or_path (Optional[str]):
            HuggingFace dataset identifier or path to local dataset.

        dataset_splits (Optional[str]):
            Specification of dataset splits to use (e.g., "train", "validation").

        dataset (Optional[Any]):
            Loaded dataset object.

        system_prompt (Optional[str]):
            Template for system prompts. Default format:
            ```
            A conversation between User and Assistant. The user asks a question,
            and the Assistant solves it. The assistant first thinks about the
            reasoning process in the mind and then provides the user with the
            answer. The reasoning process and answer are enclosed within
            <think> </think> and <answer> </answer> tags.
            ```

        response_print (str):
            Template for formatting correct responses.
            Default: "======<CORRECT_RESPONSE>========{}"

        begin_smiles_tag (str):
            Start delimiter for SMILES notation. Default: "[BEGIN_SMILES]"

        end_smiles_tag (str):
            End delimiter for SMILES notation. Default: "[END_SMILES]"
    """

    dataset_id_or_path: Optional[str] = None
    dataset_splits: Optional[str] = None
    dataset: Optional[Any] = None
    task_mode: Optional[str] = "base"
    task_kwargs: Dict[str, Any] = field(default_factory=dict)

    system_prompt: Optional[str] = Field(
        "A conversation between User and Assistant. The user asks a question, and the "
        "Assistant solves it. The assistant first thinks about the reasoning process "
        "in the mind and then provides the user with the answer. The reasoning process "
        "and answer are enclosed within <think> </think> and <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think>"
        "<answer> answer here </answer>"
    )
    response_print: str = "\n\n======<CORRECT_RESPONSE>========\n{}"
    begin_smiles_tag: str = "[BEGIN_SMILES]"
    end_smiles_tag: str = "[END_SMILES]"

    def load(self) -> Any:
        """Define load method if not hf dataset."""
        if self.dataset_id_or_path is None:
            raise NotImplementedError
        else:
            self.dataset = load_dataset(self.dataset_id_or_path, split=self.dataset_splits)
            return self.dataset

    def accuracy_reward(self, completions, **kwargs):
        """Define accuracy reward"""
        raise NotImplementedError

    def generate_prompt(self, problem, tokenizer, **kwargs):
        r1_prefix = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.question_template.format(problem),
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "problem": problem,
        }

    def dataset_preprocess(self, tokenizer):
        self.dataset["train"] = (
            self.dataset["train"].shuffle(seed=42).select(range(min(50000, len(self.dataset["train"]))))
        )
        self.dataset["test"] = (
            self.dataset["test"].shuffle(seed=42).select(range(min(10000, len(self.dataset["test"]))))
        )

        self.dataset = self.dataset.map(lambda x: self.generate_prompt(x["problem"], tokenizer))
        return self.dataset

    def log_correct(self, content, p=0.5):
        if random.random() < p:
            print(self.response_print.format(content))

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
                if random.random() < 0.05:
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

    def reasoning_steps_reward(self, completions, **kwargs):
        r"""Reward function that checks for clear step-by-step reasoning.
        Regex pattern:
            Step \d+: - matches "Step 1:", "Step 2:", etc.
            ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
            \n- - matches bullet points with hyphens
            \n\* - matches bullet points with asterisks
            First,|Second,|Next,|Finally, - matches transition words
        """
        pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,|Wait|But|but|However)"
        completion_contents = [completion for completion in completions]
        matches = [len(re.findall(pattern, content)) for content in completion_contents]

        # Magic number 3 to encourage 3 steps and more, otherwise partial reward
        return [min(1.0, count / 3) for count in matches]

    def get_metrics(self) -> dict:
        """
        Get task metrics to log in WANDB.
        This function takes no arguments and returns a dictionary of metrics {key[str]: value[float]}.
        """
        return dict()

    def random_print(self, print_data: dict, out_rate=0.01):
        if random.random() < out_rate:  # 1% chance to print a completion
            out = "\n\n=======<RANDOM_RESPONSE>=======\n"
            for k, v in print_data.items():
                out += f"*** {k.upper()}: {v}\n"
            print(out)

    def good_print(self, print_data: dict, out_rate=0.1):
        if random.random() < out_rate:  # 10% chance to print a completion
            # print(f"\n\n=======<RANDOM_RESPONSE>=======\n{completion}")
            out = "\n\n=======<GOOD_RESPONSE>=======\n"
            for k, v in print_data.items():
                out += f"*** {k.upper()}: {v}\n"

            print(out)

    def count_waits(self, completion: str):
        return completion.lower().count("wait")


class SMILESBasedTask(RLTask):
    def _post_process_smiles(self, smiles):
        smiles = re.sub(r"(?<=[A-Za-z]|\)|\])-(?=[A-Za-z]|\(|\[)", "", smiles)
        smiles = re.sub(r"\[CH\d?\]", "C", smiles)
        smiles = re.sub(
            r"\[(?:Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)\]",
            lambda m: m.group(0).strip("[]"),
            smiles,
        )
        return smiles

    def extract_smiles(self, completion: str, **kwargs):

        excluded_smiles = set(("I"))
        words = completion.split()
        words = [w.strip(" !\"#$%&'*+,-./:;<=>?@\\^_`{|}~") for w in words]
        # words_tkns = [smiles_tokenizer(w) for w in words]
        # smiles = [w for w, w_tokens in zip(words, words_tkns) if w_tokens.replace(' ', '') == w]
        smiles = words
        smiles = [s for s in smiles if s and s not in excluded_smiles]
        smiles = [self._post_process_smiles(s) for s in smiles]
        smiles = [s for s in smiles if Chem.MolFromSmiles(s)]
        return smiles

    def extract_smiles_from_answer(self, answer: str, **kwargs):
        """Extract the longest SMILES from the answer"""
        smiles = self.extract_smiles(answer)
        smiles = max(smiles, key=len) if smiles else None
        return smiles

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
