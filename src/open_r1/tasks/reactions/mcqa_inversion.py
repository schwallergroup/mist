import re
from random import random

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from ..base import RLTask


class SmilesInversion(RLTask):
    question_template: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = """
        <|im_start|>assistant
        You are a useful Chemistry assistant and will answer the following MCQ. 
        Give your reasoning inside <think>...</think> tags, then respond with the option letter 
        inside <answer>...</answer> tags. Make sure to think through all four options (A, B, C, D) 
        before choosing your final answer.
        <|im_end|>

        <|im_start|>user
        Question: Which chemical reaction is correct? Choose from the following options and make sure to think before answering.
        Options:
        A. {}
        B. {}
        C. {}
        D. {}

        When reasoning about each possible options, make sure to only use the information provided within the option..
        <|im_end|>

        <|im_start|>assistant
        <think>"""

    def load(self) -> DatasetDict:
        "loading & preping the dataset"

        df = pd.read_csv(self.dataset_id_or_path)
        shuffled = [
            np.random.permutation(row).tolist() for row in df[["true_reaction", "fake1", "fake2", "fake3"]].values
        ]
        train_dict = {
            "solution": df["true_reaction"].tolist(),
            "options": shuffled,
        }

        train_dataset = Dataset.from_dict(train_dict)
        train_test_split = train_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Combine into DatasetDict
        self.dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
        return self.dataset

    def generate_prompt(self, tokenizer, **kwargs):
        """Prompt for the MC task"""
        options = kwargs.get("options", [])
        r1_prefix = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.question_template.format(*options),
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "options": options,
        }

    def dataset_preprocess(self, tokenizer):
        self.dataset["train"] = (
            self.dataset["train"].shuffle(seed=42).select(range(min(50000, len(self.dataset["train"]))))
        )
        self.dataset["test"] = (
            self.dataset["test"].shuffle(seed=42).select(range(min(10000, len(self.dataset["test"]))))
        )

        self.dataset = self.dataset.map(lambda x: self.generate_prompt(tokenizer, options=x["options"]))

        return self.dataset

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        pattern = r"<answer>(.*)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            ans = m.groups()[0]
            return ans
        else:
            return "NONE"

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

    def accuracy_reward(self, completions, solution, options, **kwargs):

        completions = self.preprocess_completions(completions)
        rewards = []
        letters = "ABCD"

        for i, (completion, sol) in enumerate(zip(completions, solution)):
            ans = self.preprocess_response(completion).strip()
            ans = re.sub(r"[^ABCD]", "", ans)
            reward = 0.0

            if len(ans) == 1 and ans in letters:
                idx = letters.index(ans)
                select = options[i][idx]
                correct = select == sol

                info = {"choice": ans, "selected_text": select, "gold": sol}

                if correct:
                    reward = 1.0
                    print(f"\n\n====== CORRECT COMPLETION DUMP (idx={i}) ======")
                    print(f"Choice: {ans!r}  Selected: {select!r}  Gold: {sol!r}\n")
                    print(completion)
                    print("====== END DUMP ======\n")

                    self.good_print(info)

                else:
                    reward = 0.0
                    if random() < 0.5:
                        print(f"\n\n====== INCORRECT COMPLETION DUMP (idx={i}) ======")
                        print(f"Choice: {ans!r}  Selected: {select!r}  Gold: {sol!r}\n")
                        print(completion)
                        print("====== END DUMP ======\n")

                    self.random_print(info)
            else:
                if random() < 0.5:
                    print(f"\n\n====== BAD ANSWER FORMATING DUMP (idx={i}) ======")
                    print(f"The answer given: {ans!r}")
                    print(completion)

            rewards.append(reward)

        return rewards

    def thinking_length_reward(self, completions, **kwargs):
        """
        +1.0 if the model’s <think>…</think> block is at least X words long, else 0.0.
        """
        rewards = []
        threshold = 100

        for c in completions:
            m = re.search(r"<think>(.*?)</think>", c, re.DOTALL)
            think_text = m.group(1).strip() if m else ""
            word_count = len(think_text.split())
            rewards.append(1.0 if word_count >= threshold else 0.0)

        return rewards

    def format_reward(self, prompts, completions, **kwargs):
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

                # if answer present 0.1
                if completion.endswith("</answer>"):
                    current_reward += 0.1
                else:
                    current_reward -= 0.1

                # if thinking followed by answer 0.1
                if completion.count("</think>\n<answer>") == 1:
                    current_reward += 0.1
                else:
                    current_reward -= 0.1

                # if answer is not empty 0.2 and if answer is letter 0.2
                m_ans = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
                if not m_ans:
                    current_reward -= 0.20
                else:
                    current_reward += 0.20

                    letter = m_ans.group(1).strip().upper()

                    if re.fullmatch(r"[ABCD]", letter):
                        current_reward += 0.20
                    else:
                        current_reward -= 0.20

                # if entire format is correct 0.2
                pattern = r"<think>(.*)<\/think>\n<answer>(.*)</answer>"
                match = re.search(pattern, completion, re.DOTALL)
                if match is None:
                    current_reward -= 0.2
                elif len(match.groups()) != 2:
                    current_reward -= 0.1
                else:
                    current_reward += 0.2

            except:
                pass
            rewards.append(current_reward)

        return rewards
