import json
import os
import re
from dataclasses import field
from typing import Any, Dict

from datasets import Dataset, DatasetDict

from open_r1.paths import expand_path

from ..base import RLTask


class ConditionalMaterialGeneration(RLTask):
    question_template: str = ""
    log_custom_metrics: bool = True
    custom_metrics: dict = field(default_factory=dict)
    seen_comps_set: set = field(default_factory=set)
    random_log: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = (
            "You are a careful model that must follow the Output Contract exactly.\n\n"
            "OUTPUT CONTRACT\n"
            "1) You may think only inside <think>...</think>.\n"
            "2) Your final answer must be a single line wrapped in <answer>...</answer>.\n"
            "3) Inside <answer>, list only element symbols separated by single spaces, "
            "followed by a space-group tag <sg space-group number>.\n"
            "4) If you generate a chemical formula with subscripts (e.g. Ni\u2082Fe\u2084LiO\u2081\u2080), "
            "you must expand them into repeated symbols "
            "(Ni Ni Fe Fe Fe Fe Li O O O O O O O O O O).\n"
            "5) Do not include any extra words, punctuation, examples, explanations, "
            "or text outside the tags.\n"
            "6) After you produce </answer>, you must stop. No tokens are allowed after </answer>.\n\n"
            "VALID EXAMPLE (format only):\n"
            "<think>reasoning</think>\n"
            "<answer> Ca O Sn Sn <sg62></answer>\n\n"
            "INVALID EXAMPLES:\n"
            "- Missing tags\n"
            "- Commas, bullets, or explanations inside <answer>\n"
            "- Multiple <answer> blocks\n"
            "- Extra output after </answer>\n\n"
            "Allowed tokens inside <answer>: element symbols (H He Li ... Og), "
            "single spaces, and the literal pattern <sg space-group number>."
        )

        self.question_template = (
            "You are a materials science expert.\n"
            "Given the following elements: {}, propose one chemically valid and novel crystalline compound."
        )

        self.log_custom_metrics = True
        self.custom_metrics = {
            "val/rewards": [],
        }

        comps_path = os.path.join(os.path.dirname(__file__), "comps_used_in_sft.json")
        with open(comps_path, "r") as file:
            seen_comps = json.load(file)

        from pymatgen.core import Composition

        self.seen_comps_set = set()
        for comp in seen_comps:
            comp = Composition(comp)
            self.seen_comps_set.add(comp)

    def read_files(self) -> Dict:
        dataset_path = expand_path(os.path.join(self.dataset_id_or_path, "NatureLM_conditional_v2.json"))
        with open(dataset_path, "r") as file:
            data = json.load(file)

        problems = []
        solutions = []

        for pt in data:
            try:
                problems.append(pt.get("elements"))
                solutions.append("")
            except KeyError as e:
                print(f"Missing expected key in data: {e}")

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
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "problem": problem,
        }

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        train_dict = self.read_files()
        train_dataset = Dataset.from_dict(train_dict)
        seed = 42
        train_test_split = train_dataset.train_test_split(test_size=0.1, seed=seed)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        self.dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
        return self.dataset

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
            input_elements = match.group(1).split(", ") if match else []

            # Extract elements and space group from output
            output_pattern = r"<answer>\s*((?:[A-Z][a-z]?\s*)+?)\s*<sg(\d+)>\s*</answer>"
            output_matches = re.findall(output_pattern, completion)
            if len(output_matches) < 1:
                rewards.append(reward)
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
            reward += 1

            output_elements = elements_str.strip().split()

            # Penalize extra elements not in input
            extra_elements = set(output_elements) - set(input_elements)
            reward -= len(extra_elements) * 0.5

            # Check precision
            intersection = set(input_elements) & set(output_elements)
            precision = len(intersection) / len(input_elements)
            if precision == 1:
                reward += 3
            else:
                reward += precision

            # Try building a composition after applying penalties
            try:
                from pymatgen.core import Composition
                from smact.screening import smact_validity

                comp = Composition(" ".join(output_elements))
                if not smact_validity(comp):
                    rewards.append(reward)
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

            self.random_log = {
                "prompt": prompt,
                "output_elements": output_elements,
                "output_sg": output_sg,
                "accuracy_reward": reward,
                "full_completion": completion,
            }
            self.good_print(self.random_log)

            rewards.append(reward)
            self.custom_metrics["val/rewards"].extend(rewards)
        return rewards

    def get_metrics(self) -> Dict:
        """
        Get task metrics to log in WANDB.
        This function takes no arguments and returns a dictionary of metrics {key[str]: value[float]}.
        """
        metrics = dict()
        if self.log_custom_metrics:
            rewards = self.custom_metrics["val/rewards"]
            if rewards:
                correct_count = sum(1 for r in rewards if r == 1)
                total_count = len(rewards)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                metrics["val/accuracy"] = accuracy
                self.custom_metrics["val/rewards"] = []
        return metrics
