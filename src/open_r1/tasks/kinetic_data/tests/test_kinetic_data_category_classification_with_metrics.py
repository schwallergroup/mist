import re
from pathlib import Path

import pytest
from transformers import AutoTokenizer

import yaml
from kinetic_data_category_classification_with_metrics import KineticDataCategoryClassificationWithMetrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


response_wrong_format = """
    <think> Hi, I need to figure out the reaction class for these reaction data. Okay, let's see. The user provided four data runs with different initial conditions and time data, and their corresponding substrate and product concentrations. I remember that reaction classes like M1 to M20 are used to classify reactions based on their rate laws and intermediates.

        First, I should look for the rate law patterns. Maybe they're both first-order with respect to the substrate ([S]₀) and the catalyst [S], so maybe second-order overall, which is M2 or M20.

        But then I notice that some runs haven't reached 95% conversion yet, which might indicate the reaction is reversible. Hmm, for reversible reactions, M20 is used, which considers the equilibrium constant.

        Looking at the data runs, in each one the product is initially present and then react, but doesn't always reach near 95% conversion. Still, the rate constants in run 4 are higher for the forward than reverse reactions, which suggests a tendency toward product formation, but without equilibration, the conversion isn't as high.

        The product concentration increases over time, which supports the presence of a forward reaction, possibly consistent with M2, but without reaching 95% yet. Maybe M20 is the correct class because it involves both forward and reverse reactions.

        I'm a bit confused because with forward and reverse reactions, the overall reaction is reversible, which is more accurately represented by M20. However, the product is accumulating but doesn't seem to peak as drastically as M20 would predict without equilibration.

        I think M20 is the most appropriate class for these data points. It accounts for the competing rates and doesn't require equilibration, giving a clearer picture of the reaction kinetics.
        </think>

        Based on the reaction data provided, the reaction class is M20, which accounts for the competing rates of forward and reverse reactions, as well as the presence of a product with initial activation, avoiding equilibration to 95%.
        \\boxed{M20}
"""

response_correct_format = """
    <think> Hi, I need to figure out the reaction class for these reaction data. Okay, let's see. The user provided four data runs with different initial conditions and time data, and their corresponding substrate and product concentrations. I remember that reaction classes like M1 to M20 are used to classify reactions based on their rate laws and intermediates.

        First, I should look for the rate law patterns. Maybe they're both first-order with respect to the substrate ([S]₀) and the catalyst [S], so maybe second-order overall, which is M2 or M20.

        But then I notice that some runs haven't reached 95% conversion yet, which might indicate the reaction is reversible. Hmm, for reversible reactions, M20 is used, which considers the equilibrium constant.

        Looking at the data runs, in each one the product is initially present and then react, but doesn't always reach near 95% conversion. Still, the rate constants in run 4 are higher for the forward than reverse reactions, which suggests a tendency toward product formation, but without equilibration, the conversion isn't as high.

        The product concentration increases over time, which supports the presence of a forward reaction, possibly consistent with M2, but without reaching 95% yet. Maybe M20 is the correct class because it involves both forward and reverse reactions.

        I'm a bit confused because with forward and reverse reactions, the overall reaction is reversible, which is more accurately represented by M20. However, the product is accumulating but doesn't seem to peak as drastically as M20 would predict without equilibration.

        I think M20 is the most appropriate class for these data points. It accounts for the competing rates and doesn't require equilibration, giving a clearer picture of the reaction kinetics.
        </think>
        <answer>Core Mechanism</answer>
"""


class TestAccuracyReward:
    def setup_method(self):
        # Load configuration
        config_path = "/home/kuroki/sink/recipes/kinetic_metrics_category.yaml"
        config = load_config(config_path)

        self.classification_task = (
            KineticDataCategoryClassificationWithMetrics(
                dataset_id_or_path=config["dataset_id_or_path"],
                model_revision=config["model_revision"],
                torch_dtype=config["torch_dtype"],
                attn_implementation=config["attn_implementation"],
                bf16=config["bf16"],
                tf32=config["tf32"],
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "/work/liac/LLM_models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
            revision=config["model_revision"],
            trust_remote_code=False,
        )

    def test_prompt_format(self):
        self.classification_task.load()
        self.classification_task.dataset_preprocess(self.tokenizer)
        problem = self.classification_task.dataset["train"][0]["problem"]
        question_template = self.classification_task.question_template
        question = question_template.format(problem)

        regex = r"<answer>(.*?)</answer>"
        match = re.search(regex, question)
        assert len(match.groups()) == 1

        for i in range(10):
            solution = self.classification_task.dataset["train"][i]["solution"]
            assert solution in [
                "Core mechanism",
                "Mechanism with bicatalytic steps",
                "Mechanism with catalyst activation steps",
                "Mechanism with catalyst deactivation steps",
            ]

        for i in range(10):
            solution = self.classification_task.dataset["test"][i]["solution"]
            assert solution in [
                "Core mechanism",
                "Mechanism with bicatalytic steps",
                "Mechanism with catalyst activation steps",
                "Mechanism with catalyst deactivation steps",
            ]

    def test_format_reward(self):
        responses = [response_wrong_format, response_correct_format]
        regex = r"<think>(.*?)</think>.*?<answer>(.*?)</answer>"
        match = re.search(regex, response_correct_format, re.DOTALL)

        regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
        match = re.search(regex, response_correct_format, re.DOTALL)
        print("----match groups----")
        print(match.groups())

        assert [
            round(r, 1)
            for r in self.classification_task.format_reward(responses)
        ] == [
            float(0.0),
            float(1.0),
        ]

    def test_format_continuous_reward(self):
        responses = [response_wrong_format, response_correct_format]
        regex = r"<think>(.*?)</think>.*?<answer>(.*?)</answer>"
        match = re.search(regex, response_correct_format, re.DOTALL)

        regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
        match = re.search(regex, response_correct_format, re.DOTALL)
        print("----match groups----")
        print(match.groups())

        for r in self.classification_task.format_continuous_reward(responses):
            assert r >= -1.0 and r <= 1.0

    def test_answer_covered_in_reasoning_traces_reward(self):
        responses = [response_wrong_format, response_correct_format]
        rewards = (
            self.classification_task.answer_covered_in_reasoning_traces_reward(
                responses, ["Core Mechanism", "Core Mechanism"]
            )
        )
        assert rewards == [float(0), float(0.1)]

    def test_accuracy_reward(self):
        responses = [response_wrong_format, response_correct_format]
        rewards = self.classification_task.accuracy_reward(
            responses, ["Core Mechanism", "Core Mechanism"]
        )
        assert rewards == [float(0), float(1)]

    def test_preprocess_response(self):
        ans = self.classification_task.preprocess_response(
            response_correct_format
        )
        assert ans == "Core Mechanism"
        ans = self.classification_task.preprocess_response(
            response_wrong_format
        )
        assert ans == "NONE"
