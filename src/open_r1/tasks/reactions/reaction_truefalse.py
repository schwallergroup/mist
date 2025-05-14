
from ..base import RLTask
import numpy as np
import re
from random import random
from datasets import Dataset, DatasetDict
import pandas as pd

class ReactionTrueFalse(RLTask):
    question_template: str = ""
    printed_sample_prompt: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        printed_sample_prompt: bool = False
        self.question_template = """
        <|im_start|>assistant
        You are a useful Chemistry assistant and will answer the following question by saying if the chemical reaction provided is True or False. 
        Give your reasoning inside <think>...</think> tags, then respond with the option letter 
        inside <answer>...</answer> tags. Make sure to think through before choosing your final answer.
        <|im_end|>

        <|im_start|>user
        Question: Is this chemical reaction correct ? {} Answer True or False.
    
        Make sure to reason about the mechanism of the reaction and show me your understanding by argumenting your choice. 
        <|im_end|>

        <|im_start|>assistant
        <think>"""

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        train_dict = {
            'problem': df['reaction'].tolist(),
            'solution': df['label'].tolist()
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed = 42)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
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
        """
        For each completion:
        1. Extract all spans inside <answer>...</answer> and <think>...</think>.
        2. Within those spans, collect every literal "true" or "false" (case‐insensitive).
        3. Do a majority vote: if count_true > count_false → predict "true",
            elif count_false > count_true → predict "false",
            else (tie) → take the first mentioned token (if any), else "".
        4. Compare that prediction to the provided gold solution (normalized to
            lower‐case string) and issue +1.0 if equal, else –0.5.
        """
        completions = self.preprocess_completions(completions)
        solutions = kwargs.get("solution", [])
        rewards = []

        for completion, gold in zip(completions, solutions):
            answer_spans = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE)
            think_spans  = re.findall(r"<think>(.*?)</think>",   completion, re.DOTALL | re.IGNORECASE)
            spans = answer_spans + think_spans
            tokens = []
            for span in spans:
                tokens += re.findall(r"\b(true|false)\b", span, flags=re.IGNORECASE)
            tokens = [t.lower() for t in tokens]

            count_true  = tokens.count("true")
            count_false = tokens.count("false")

            if count_true > count_false:
                predicted = "true"
            elif count_false > count_true:
                predicted = "false"
            else:
                predicted = tokens[0] if tokens else ""

            gold_str = str(gold).strip().lower()
            if predicted == gold_str:
                reward = 1.0 
                print("======= FULL_COMPLETION_CORRECT =======")
                print(f"All Completion: {completion}")
                print(f"All true answer count:    {count_true}")
                print(f"All false answer count:    {count_false}")
                print(f"Predicted:    {predicted}")
                print(f"Ground truth:   {gold_str!r}\n")

            else:
                reward = -0.5
                if random() < 0.2:
                    print("======= FULL_COMPLETION_CORRECT =======")
                    print(f"All Completion: {completion}")
                    print(f"All true answer count:    {count_true}")
                    print(f"All false answer count:    {count_false}")
                    print(f"Predicted:    {predicted}")
                    print(f"Ground truth:   {gold_str!r}\n")

            rewards.append(reward)

        return rewards


    
    # def accuracy_reward(self, completions, **kwargs):
    #     completions = self.preprocess_completions(completions)
    #     solution = kwargs.get("solution", [])
    #     rewards = []

    #     for completion, gold in zip(completions, solution):
    #         # extract all <answer>…</answer> spans
    #         answers = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    #         answers = [ans.strip() for ans in answers]

    #         # pick the first answer as "selected"
    #         selected = answers[0].strip().lower() if answers else ""
    #         reward = 0.0

    #         # normalize gold to string
    #         gold_str = str(gold).strip().lower()

    #         if any(ans.lower() == gold_str for ans in answers):
    #             reward = 1.0
    #             print("======= FULL_COMPLETION_CORRECT =======")
    #             print(f"All Completion: {completion}")
    #             print(f"All answers:    {answers}")
    #             print(f"Selected ans:   {selected!r}")
    #             print(f"Ground truth:   {gold_str!r}\n")
    #         else:
    #             reward = -0.5
    #             if random() < 0.05:
    #                 print("======= FULL_COMPLETION_INVALID =======")
    #                 print(f"All Completion: {completion}")
    #                 print(f"All answers:    {answers}")
    #                 print(f"Selected ans:   {selected!r}")
    #                 print(f"Ground truth:   {gold_str!r}\n")

    #         rewards.append(reward)

    #     return rewards

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
                if random.random() < 0.00:  
                    print(f"\n\n=======<RANDOM_RESPONSE>=======\n{completion}")
                if not completion.startswith("<think>"):
                    completion = "<think>" + completion
                regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
                match = re.search(regex, completion, re.DOTALL)
                # if the format is not correct, reward is 0
                if match is None or len(match.groups()) != 2:
                    rewards.append(0.0)
                else:
                   
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



 


