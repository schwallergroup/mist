import os
import pickle
import random
import re
from typing import List

from datasets import Dataset, DatasetDict

from open_r1.tasks.base import RLTask
from open_r1.tasks.kinetic_data.calculation_metrics import KineticMetricsCalculator


class KineticDataClassificationWithMetrics(RLTask):
    question_template: str = ""
    x1_train: List = None
    x2_train: List = None
    y_train: List = None
    x1_test: List = None
    x2_test: List = None
    y_test: List = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = """
        Analyze the reaction behaviour and catalyst performance, then estimate the reaction class based on the following metrics.
        The possible reaction classes are M1 to M20 indicated as follows.
        Please begin your response with "<think>", then provide a detailed, step-by-step reasoning process (including any intermediate reflections or re-evaluations), 
        then end with </think>, and finally put your final answer within \\boxed{{}} tags, for example \\boxed{{M1}}.

        # Possible reaction classes
        // M1 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2

        // M2 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2cat<=>cat2;k3,k-3

        // M3 Mechanism
        S+cat2<=>((cat)2S);k1,k-1|((cat)2S)<=>P+cat2;k2,k-2|2cat<=>cat2;k3,k-3

        // M4 Mechanism
        X+catS<=>S+cat;k1,k-1|X+catS<=>P+cat;k2,k-2

        // M5 Mechanism
        S+cat<=>catS;k1,k-1|catS+cat<=>catP;k2,k-2|catP<=>P+cat;k3,k-3

        // M6 Mechanism
        cat<=>cat*;k1,0|S+cat*<=>cat*S;k1,k-1|cat*S<=>P+cat*;k2,k-2

        // M7 Mechanism
        S+cat<=>catS;k1,k-1|S+catS<=>catS2;k3,k-3|catS<=>P+cat;k2,k-2

        // M8 Mechanism
        S+cat*<=>cat*S;k1,k-1|cat*S<=>P+cat*;k2,k-2|cat+L<=>cat*;k3,k-3

        // M9 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat<=>inactive cat;k3,0

        // M10 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|inhibitor+cat<=>inactive catI;k3,0

        // M11 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|S+cat<=>inactive catS;k-3,0

        // M12 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|P+cat<=>inactive catP;k-3,0

        // M13 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2cat<=>inactive cat2;k-3,0

        // M14 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|catS<=>inactive catS;k-3,0

        // M15 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|inhibitor+catS<=>inactive catSI;k-3,0

        // M16 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|S+catS<=>inactive catS2;k-3,0

        // M17 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|P+catS<=>inactive catSP;k-3,0

        // M18 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2catS<=>inactive cat2S2;k-3,0

        // M19 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat+catS<=>inactive cat2S;k3,0

        // M20 Mechanism
        S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat<=>inactive cat;k3,0|catS<=>inactive catS;k4,0
        
        {}

        <think>
        """

    def load(self) -> DatasetDict:
        """
        Load and prepare the dataset for the task.

        Returns:
            DatasetDict: Dataset with 'train' and 'test' splits
        """

        # Assume the dataset is in the same directory as the script
        x1_train_path = os.path.join(
            self.dataset_id_or_path,
            "x1_train_M1_M20_train_val_test_set_part_0.pkl",
        )
        x2_train_path = os.path.join(
            self.dataset_id_or_path,
            "x2_train_M1_M20_train_val_test_set_part_0.pkl",
        )
        y_train_path = os.path.join(
            self.dataset_id_or_path,
            "y_train_M1_M20_train_val_test_set_part_0.pkl",
        )

        x1_test_path = os.path.join(
            self.dataset_id_or_path,
            "x1_val_M1_M20_train_val_test_set_part_0.pkl",
        )
        x2_test_path = os.path.join(
            self.dataset_id_or_path,
            "x2_val_M1_M20_train_val_test_set_part_0.pkl",
        )
        y_test_path = os.path.join(
            self.dataset_id_or_path,
            "y_val_M1_M20_train_val_test_set_part_0.pkl",
        )

        # Implement dataset loading logic
        with open(x1_train_path, "rb") as f:
            self.x1_train = pickle.load(f)
        with open(x2_train_path, "rb") as f:
            self.x2_train = pickle.load(f)
        with open(y_train_path, "rb") as f:
            self.y_train = pickle.load(f)

        # Validate data shapes
        if not (len(self.x1_train) == len(self.x2_train) == len(self.y_train)):
            raise ValueError(
                f"Data shapes mismatch: x1_train={len(self.x1_train)}, "
                f"x2_train={len(self.x2_train)}, y_train={len(self.y_train)}"
            )

        with open(x1_test_path, "rb") as f:
            self.x1_test = pickle.load(f)
        with open(x2_test_path, "rb") as f:
            self.x2_test = pickle.load(f)
        with open(y_test_path, "rb") as f:
            self.y_test = pickle.load(f)

        # Validate test data shapes
        if not (len(self.x1_test) == len(self.x2_test) == len(self.y_test)):
            raise ValueError(
                f"Test data shapes mismatch: x1_test={len(self.x1_test)}, "
                f"x2_test={len(self.x2_test)}, y_test={len(self.y_test)}"
            )

        self.y_test = self.y_test.reshape(-1, 1)[: self.x1_test.shape[0]]

        prompt_template_data = f"""
        # Metrics that calculated from the data
        The following metrics were calculated from four experimental runs. For each metric, the mean, standard deviation (std), minimum, and maximum values are provided to help assess catalyst behavior and reaction performance.

        ## Reaction Order
        Indicates how the reaction rate depends on reactant concentration. Values >1 suggest nonlinear behavior.
        - Mean: {{reaction_order_mean}}
        - Std: {{reaction_order_std}}

        ## Turnover frequency (TOF)
        The number of product molecules formed per catalyst site per second. Higher is more active.
        - Mean: {{TOF_mean}}
        - Std: {{TOF_std}}
        - Min: {{TOF_min}}
        - Max: {{TOF_max}}

        ## Turnover number (TON)
        Total catalytic lifetime (stability); higher TON means catalyst is more robust and longer-lasting.
        - Mean: {{TON_mean}}
        - Std: {{TON_std}}
        - Min: {{TON_min}}
        - Max: {{TON_max}}

        ## Catalyst stability
        A normalized value from 0 to 1 showing how stable the catalyst is during the reaction. Higher is more stable.
        - Mean: {{catalyst_stability_mean}}
        - Std: {{catalyst_stability_std}}
        - Min: {{catalyst_stability_min}}
        - Max: {{catalyst_stability_max}}

        ## Induction period
        The time before the reaction rate becomes significant. A long period may suggest catalyst activation delays.
        - Mean: {{induction_period_mean}}
        - Std: {{induction_period_std}}
        - Min: {{induction_period_min}}
        - Max: {{induction_period_max}}

        ## Mass balance gap
        Indicates how much mass is “missing” from expected total. High gaps may imply side reactions or errors.
        - Mean: {{mass_balance_gap_mean}}
        - Std: {{mass_balance_gap_std}}
        - Min: {{mass_balance_gap_min}}
        - Max: {{mass_balance_gap_max}}

        ## Deactivation rate constant
        Rate at which the catalyst deactivates. Higher values mean faster loss of activity.
        - Mean: {{deactivation_rate_constant_mean}}
        - Std: {{deactivation_rate_constant_std}}
        - Min: {{deactivation_rate_constant_min}}
        - Max: {{deactivation_rate_constant_max}}

        ## Time of maximum curvature
        Time at which the rate of change in conversion is highest. Indicates key kinetic transitions.
        - Mean: {{time_max_curvature_mean}}
        - Std: {{time_max_curvature_std}}
        - Min: {{time_max_curvature_min}}
        - Max: {{time_max_curvature_max}}

        ## Active catalyst fraction
        Proportion of catalyst that remains active during the reaction. 1.0 means fully active.
        - Mean: {{active_catalyst_fraction_mean}}
        - Std: {{active_catalyst_fraction_std}}
        - Min: {{active_catalyst_fraction_min}}
        - Max: {{active_catalyst_fraction_max}}

        ## Catalyst activity half-life
        Time required for catalyst activity to reduce by half. Longer means more durable catalyst.
        - Mean: {{catalyst_activity_half_life_mean}}
        - Std: {{catalyst_activity_half_life_std}}
        - Min: {{catalyst_activity_half_life_min}}
        - Max: {{catalyst_activity_half_life_max}}

        ## Equilibrium constant (Keq)
        Ratio of product to reactant at equilibrium. Higher values indicate more favorable product formation.
        - Mean: {{Keq_mean}}
        - Std: {{Keq_std}}
        - Min: {{Keq_min}}
        - Max: {{Keq_max}}

        ## SP_mid_ratio
        Ratio of substrate to product at midpoint of the reaction. Useful for understanding reaction progress.
        - Mean: {{SP_mid_ratio_mean}}
        - Std: {{SP_mid_ratio_std}}
        - Min: {{SP_mid_ratio_min}}
        - Max: {{SP_mid_ratio_max}}

        ## Mass gap at midpoint
        Mass balance gap measured at the halfway point of the reaction. Can signal mid-reaction instabilities.
        - Mean: {{mass_gap_mid_mean}}
        - Std: {{mass_gap_mid_std}}
        - Min: {{mass_gap_mid_min}}
        - Max: {{mass_gap_mid_max}}
        """

        prompts_train = []
        for i in range(self.x1_train.shape[0]):
            data = self.generate_data_pass_to_prompt(i, is_test=False)
            metrics = KineticMetricsCalculator(data).summarize_metrics_for_ml()
            prompt = prompt_template_data.format(**metrics)
            prompts_train.append(prompt)

        train_dict = {
            "problem": prompts_train,
            "solution": [
                "M" + str(int(y[0]) + 1) for y in self.y_train.tolist()
            ],
            "options": [
                ["M" + str(i) for i in range(1, 21)]
                for _ in range(self.x1_train.shape[0])
            ],
        }

        prompts_test = []
        for i in range(self.x1_test.shape[0]):
            data = self.generate_data_pass_to_prompt(i, is_test=True)
            metrics = KineticMetricsCalculator(data).summarize_metrics_for_ml()
            prompt = prompt_template_data.format(**metrics)
            prompts_test.append(prompt)

        test_dict = {
            "problem": prompts_test,
            "solution": [
                "M" + str(int(y[0]) + 1) for y in self.y_test.tolist()
            ],
            "options": [
                ["M" + str(i) for i in range(1, 21)]
                for _ in range(self.x1_test.shape[0])
            ],
        }

        self.dataset = DatasetDict(
            {
                "train": Dataset.from_dict(train_dict),
                "test": Dataset.from_dict(test_dict),
            }
        )
        return self.dataset

    def generate_data_pass_to_prompt(self, index, is_test=False):
        """
        Generate data dictionary for prompt template.

        Args:
            index (int): Index of the data point
            is_test (bool): Whether this is for test data or not

        Returns:
            dict: Dictionary containing data for all runs
        """
        x1_data = self.x1_test if is_test else self.x1_train
        x2_data = self.x2_test if is_test else self.x2_train

        return {
            "run_1": {
                "initial_concentration_of_catalyst": float(x1_data[index, 0]),
                "time_data": x2_data[index, :, 0].tolist(),
                "substrate_data": x2_data[index, :, 1].tolist(),
                "product_data": x2_data[index, :, 2].tolist(),
            },
            "run_2": {
                "initial_concentration_of_catalyst": float(x1_data[index, 1]),
                "time_data": x2_data[index, :, 3].tolist(),
                "substrate_data": x2_data[index, :, 4].tolist(),
                "product_data": x2_data[index, :, 5].tolist(),
            },
            "run_3": {
                "initial_concentration_of_catalyst": float(x1_data[index, 2]),
                "time_data": x2_data[index, :, 6].tolist(),
                "substrate_data": x2_data[index, :, 7].tolist(),
                "product_data": x2_data[index, :, 8].tolist(),
            },
            "run_4": {
                "initial_concentration_of_catalyst": float(x1_data[index, 3]),
                "time_data": x2_data[index, :, 9].tolist(),
                "substrate_data": x2_data[index, :, 10].tolist(),
                "product_data": x2_data[index, :, 11].tolist(),
            },
        }

    def accuracy_reward(self, completions, solutions, **kwargs):
        """Reward function - check that the answer is same as ground truth"""
        answers = [self.preprocess_response(c) for c in completions]
        rewards = []
        rewards_dict = {
            "accuracy_reward": [],
            "category_reward": [],
            "class_coverage_reward": [],
        }

        for i in range(len(solutions)):
            sol = solutions[i]
            answer = answers[i]
            completion = completions[i]

            # exact match reward
            accuracy_reward = self.exact_match_reward(sol, answer)

            # sometimes the answer is None
            if answer != "NONE":
                # category match reward
                category_reward = self.category_reward(sol, answer)

                # class coverage reward
                class_coverage_reward = self.class_coverage_reward(completion)
            else:
                category_reward = 0
                class_coverage_reward = 0

            reward = (
                0.6 * accuracy_reward
                + 0.2 * category_reward
                + 0.2 * class_coverage_reward
            )
            rewards.append(reward)
            rewards_dict["accuracy_reward"].append(accuracy_reward)
            rewards_dict["category_reward"].append(category_reward)
            rewards_dict["class_coverage_reward"].append(class_coverage_reward)

        self._metrics_output = {
            "accuracy_reward_ave": sum(rewards_dict["accuracy_reward"])
            / len(rewards_dict["accuracy_reward"]),
            "category_reward_ave": sum(rewards_dict["category_reward"])
            / len(rewards_dict["category_reward"]),
            "class_coverage_reward_ave": sum(
                rewards_dict["class_coverage_reward"]
            )
            / len(rewards_dict["class_coverage_reward"]),
            "accuracy_reward_max": max(rewards_dict["accuracy_reward"]),
            "category_reward_max": max(rewards_dict["category_reward"]),
            "class_coverage_reward_max": max(
                rewards_dict["class_coverage_reward"]
            ),
            "accuracy_reward_min": min(rewards_dict["accuracy_reward"]),
            "category_reward_min": min(rewards_dict["category_reward"]),
            "class_coverage_reward_min": min(
                rewards_dict["class_coverage_reward"]
            ),
        }
        return rewards

    def generate_prompt(self, problem, tokenizer, **kwargs):
        r1_prefix = [
            {
                "role": "user",
                "content": self.question_template.format(problem),
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True
            ),
            "problem": problem,
        }

    def dataset_preprocess(self, tokenizer):
        self.dataset["train"] = (
            self.dataset["train"]
            .shuffle(seed=42)
            .select(range(min(50000, len(self.dataset["train"]))))
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
        pattern = r"\\boxed{(.*?)}"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            ans = m.groups()[-1]
            return ans
        else:
            return "NONE"

    def exact_match_reward(self, sol, answer):
        if sol == answer:
            return 1
        else:
            return 0

    def category_reward(self, sol, answer):
        category_dict = {
            "M1": 0,
            "M2": 1,
            "M3": 1,
            "M4": 1,
            "M5": 1,
            "M6": 2,
            "M7": 2,
            "M8": 2,
            "M9": 3,
            "M10": 3,
            "M11": 3,
            "M12": 3,
            "M13": 3,
            "M14": 3,
            "M15": 3,
            "M16": 3,
            "M17": 3,
            "M18": 3,
            "M19": 3,
            "M20": 3,
        }
        if answer not in category_dict.keys():
            return 0
        if category_dict[sol] == category_dict[answer]:
            population_of_category = len(
                [i for i in category_dict.values() if i == category_dict[sol]]
            )
            return 1 / population_of_category
        else:
            return 0

    def class_coverage_reward(self, response):
        # M1〜M20のうち、何種類に言及したか
        classes_mentioned = set(re.findall(r"\bM\d+\b", response))
        score_class_coverage = len(classes_mentioned) / 20
        return score_class_coverage

    def format_reward(self, completions, **kwargs):
        """
        Format: <think>...</think>\boxed{}
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers

        Returns:
            list[float]: Reward scores
        """
        rewards = []

        for completion in completions:
            completion = "<think>" + completion
            try:
                if random.random() < 0.01:  # 1% chance to print a completion
                    print(f"\n\n=======<RANDOM_RESPONSE>=======\n{completion}")

                regex = r"<think>(.*?)</think>.*?\\boxed{(.*?)}"
                match = re.search(regex, completion, re.DOTALL)
                # if the format is not correct, reward is 0
                if match is None or len(match.groups()) != 2:
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def get_metrics(self):
        return self._metrics_output
