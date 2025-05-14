import os
import pickle
import random
import re
from typing import List

from datasets import Dataset, DatasetDict

from open_r1.tasks.base import RLTask
from open_r1.tasks.kinetic_data.calculation_metrics import KineticMetricsCalculator


prompt_template_data = f"""
        # Metrics that calculated from the data
        The following metrics were calculated from the experimental data in advance.

        ## Run 1
        - Initial concentration of catalyst: {{run_1[initial_concentration_of_catalyst]}}
        - The initial concentration of substrate: {{run_1[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_1[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_1[initial_concentration_of_product]}}
        - The final concentration of product: {{run_1[final_concentration_of_product]}}
        - Mass balance gap: {{run_1[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_1[mass_gap_mid]}}
        - Catalyst stability: {{run_1[catalyst_stability]}}
        - Induction period: {{run_1[induction_period]}}

        ## Run 2
        - Initial concentration of catalyst: {{run_2[initial_concentration_of_catalyst]}}
        - The initial concentration of substrate: {{run_2[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_2[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_2[initial_concentration_of_product]}}
        - The final concentration of product: {{run_2[final_concentration_of_product]}}
        - Mass balance gap: {{run_2[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_2[mass_gap_mid]}}
        - Catalyst stability: {{run_2[catalyst_stability]}}
        - Induction period: {{run_2[induction_period]}}

        ## Run 3
        - Initial concentration of catalyst: {{run_3[initial_concentration_of_catalyst]}}
        - The initial concentration of substrate: {{run_3[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_3[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_3[initial_concentration_of_product]}}
        - The final concentration of product: {{run_3[final_concentration_of_product]}}
        - Mass balance gap: {{run_3[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_3[mass_gap_mid]}}
        - Catalyst stability: {{run_3[catalyst_stability]}}
        - Induction period: {{run_3[induction_period]}}

        ## Run 4
        - Initial concentration of catalyst: {{run_4[initial_concentration_of_catalyst]}}
        - The initial concentration of substrate: {{run_4[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_4[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_4[initial_concentration_of_product]}}
        - The final concentration of product: {{run_4[final_concentration_of_product]}}
        - Mass balance gap: {{run_4[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_4[mass_gap_mid]}}
        - Catalyst stability: {{run_4[catalyst_stability]}}
        - Induction period: {{run_4[induction_period]}}
        
        ## Explanation of the metrics
        Mass balance gap:
        This metrics indicates how much mass is "missing" from expected total. High gaps my suggest side reactions or deactivation.
        This metrics measures the absolute difference between the substrate loss and the product gain.

        Mass balance gap at midpoint:
        This metrics indicates how much mass is "missing" from expected total at midpoint of the reaction. Can singal mid-reaction instabilities.
        This metrics measures the absolute difference between the substrate loss and the product gain at the midpoint of the reaction.

        Catalyst stability:
        This metrics indicates how stable the catalyst is. If the value is close to 1, the catalyst is stable. Much less than 1 means that the catalyst is unstable.
        This metrics is calculated by the final reaction rate devided by the initial reaction rate.

        Induction period:
        Indicates the time before the reaction rate becomes significant. A long period may suggest catalyst activation delays. 
        This metrics is calculated by detecting the first time point where product exceeds 5% of total growth.
        """


prompt_template_data_improved = f"""
        # Metrics that calculated from the data
        The following metrics were calculated from the experimental data in advance.
        The following data correspond to four experimental runs conducted under the same reaction system. 
        Runs 1–3 share the same initial substrate concentration ([S]₀) but differ in the initial catalyst concentration ([cat]₀), which ranges from 1–10 mol%. Run 4 is a 'same-excess' experiment, in which the initial substrate concentration is reduced and product ([P]₀) is added.

        ## Run 1
        - Initial concentration of catalyst: {{run_1[initial_concentration_of_catalyst]}}
        - The initial concentration of substrate: {{run_1[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_1[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_1[initial_concentration_of_product]}}
        - The final concentration of product: {{run_1[final_concentration_of_product]}}
        - Mass balance gap: {{run_1[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_1[mass_gap_mid]}}
        - Catalyst stability: {{run_1[catalyst_stability]}}
        - Induction period: {{run_1[induction_period]}}

        ## Run 2
        - Initial concentration of catalyst: {{run_2[initial_concentration_of_catalyst]}}
        - The initial concentration of substrate: {{run_2[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_2[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_2[initial_concentration_of_product]}}
        - The final concentration of product: {{run_2[final_concentration_of_product]}}
        - Mass balance gap: {{run_2[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_2[mass_gap_mid]}}
        - Catalyst stability: {{run_2[catalyst_stability]}}
        - Induction period: {{run_2[induction_period]}}

        ## Run 3
        - Initial concentration of catalyst: {{run_3[initial_concentration_of_catalyst]}}
        - The initial concentration of substrate: {{run_3[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_3[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_3[initial_concentration_of_product]}}
        - The final concentration of product: {{run_3[final_concentration_of_product]}}
        - Mass balance gap: {{run_3[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_3[mass_gap_mid]}}
        - Catalyst stability: {{run_3[catalyst_stability]}}
        - Induction period: {{run_3[induction_period]}}

        ## Run 4
        - Initial concentration of catalyst: {{run_4[initial_concentration_of_catalyst]}}
        - The initial concentration of substrate: {{run_4[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_4[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_4[initial_concentration_of_product]}}
        - The final concentration of product: {{run_4[final_concentration_of_product]}}
        - Mass balance gap: {{run_4[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_4[mass_gap_mid]}}
        - Catalyst stability: {{run_4[catalyst_stability]}}
        - Induction period: {{run_4[induction_period]}}
        
        ## Explanation of the metrics
        Mass balance gap:
        This metrics indicates how much mass is "missing" from expected total. High gaps my suggest side reactions or deactivation.
        This metrics measures the absolute difference between the substrate loss and the product gain.

        Mass balance gap at midpoint:
        This metrics indicates how much mass is "missing" from expected total at midpoint of the reaction. Can singal mid-reaction instabilities.
        This metrics measures the absolute difference between the substrate loss and the product gain at the midpoint of the reaction.

        Catalyst stability:
        This metrics indicates how stable the catalyst is. If the value is close to 1, the catalyst is stable. Much less than 1 means that the catalyst is unstable.
        This metrics is calculated by the final reaction rate devided by the initial reaction rate.

        Induction period:
        Indicates the time before the reaction rate becomes significant. A long period may suggest catalyst activation delays. 
        This metrics is calculated by detecting the first time point where product exceeds 5% of total growth.
        """


prompt_template_with_raw_data = f"""
        # Raw data and Metrics
        ## Run 1
        - Initial concentration of catalyst (normalized to [S]0): {{run_1[initial_concentration_of_catalyst]}}
        - Time_data (normalized, unitless): {{run_1[time_data]}}
        - Substrate_data (normalized to [S]0): {{run_1[substrate_data]}}
        - Product_data (normalized to [S]0): {{run_1[product_data]}}

        The following metrics were calculated from the above data.
        - The initial concentration of substrate: {{run_1[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_1[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_1[initial_concentration_of_product]}}
        - The final concentration of product: {{run_1[final_concentration_of_product]}}
        - Mass balance gap: {{run_1[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_1[mass_gap_mid]}}
        - Catalyst stability: {{run_1[catalyst_stability]}}
        - Induction period: {{run_1[induction_period]}}

        ## Run 2
        - Initial concentration of catalyst (normalized to [S]0): {{run_2[initial_concentration_of_catalyst]}}
        - Time_data (normalized, unitless): {{run_2[time_data]}}
        - Substrate_data (normalized to [S]0): {{run_2[substrate_data]}}
        - Product_data (normalized to [S]0): {{run_2[product_data]}}

        The following metrics were calculated from the above data.
        - The initial concentration of substrate: {{run_2[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_2[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_2[initial_concentration_of_product]}}
        - The final concentration of product: {{run_2[final_concentration_of_product]}}
        - Mass balance gap: {{run_2[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_2[mass_gap_mid]}}
        - Catalyst stability: {{run_2[catalyst_stability]}}
        - Induction period: {{run_2[induction_period]}}

        ## Run 3
        - Initial concentration of catalyst (normalized to [S]0): {{run_3[initial_concentration_of_catalyst]}}
        - Time_data (normalized, unitless): {{run_3[time_data]}}
        - Substrate_data (normalized to [S]0): {{run_3[substrate_data]}}
        - Product_data (normalized to [S]0): {{run_3[product_data]}}

        The following metrics were calculated from the above data.
        - The initial concentration of substrate: {{run_3[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_3[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_3[initial_concentration_of_product]}}
        - The final concentration of product: {{run_3[final_concentration_of_product]}}
        - Mass balance gap: {{run_3[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_3[mass_gap_mid]}}
        - Catalyst stability: {{run_3[catalyst_stability]}}
        - Induction period: {{run_3[induction_period]}}

        ## Run 4
        - Initial concentration of catalyst (normalized to [S]0): {{run_4[initial_concentration_of_catalyst]}}
        - Time_data (normalized, unitless): {{run_4[time_data]}}
        - Substrate_data (normalized to [S]0): {{run_4[substrate_data]}}
        - Product_data (normalized to [S]0): {{run_4[product_data]}}

        The following metrics were calculated from the above data.
        - The initial concentration of substrate: {{run_4[initial_concentration_of_substrate]}}
        - The final concentration of substrate: {{run_4[final_concentration_of_substrate]}}
        - The initial concentration of product: {{run_4[initial_concentration_of_product]}}
        - The final concentration of product: {{run_4[final_concentration_of_product]}}
        - Mass balance gap: {{run_4[mass_balance_gap]}}
        - Mass balance gap at midpoint: {{run_4[mass_gap_mid]}}
        - Catalyst stability: {{run_4[catalyst_stability]}}
        - Induction period: {{run_4[induction_period]}}
        
        ## Explanation of the metrics
        Mass balance gap:
        This metrics indicates how much mass is "missing" from expected total. High gaps my suggest side reactions or deactivation.
        This metrics measures the absolute difference between the substrate loss and the product gain.

        Mass balance gap at midpoint:
        This metrics indicates how much mass is "missing" from expected total at midpoint of the reaction. Can singal mid-reaction instabilities.
        This metrics measures the absolute difference between the substrate loss and the product gain at the midpoint of the reaction.

        Catalyst stability:
        This metrics indicates how stable the catalyst is. If the value is close to 1, the catalyst is stable. Much less than 1 means that the catalyst is unstable.
        This metrics is calculated by the final reaction rate devided by the initial reaction rate.

        Induction period:
        Indicates the time before the reaction rate becomes significant. A long period may suggest catalyst activation delays. 
        This metrics is calculated by detecting the first time point where product exceeds 5% of total growth.
"""


class KineticDataCategoryClassificationWithMetrics(RLTask):
    question_template: str = ""
    x1_train: List = None
    x2_train: List = None
    y_train: List = None
    x1_test: List = None
    x2_test: List = None
    y_test: List = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = ("""
            <|im_start|>assistant
            You are a useful Chemistry assistant and you will answer the following category prediction question. Give your reasoning inside the <think>...</think> tags and then respond inside <answer>...</answer> tags,
            think and reason for all the options before giving your answer. Structure your reasoning such that you think through all options before giving the answer.<|im_end|>

            <|im_start|>user
            Observe all four runs and evaluate all possible categories. Then choose the most likely one based on your observation.
            Choose ONLY from the following options and write your response choice inside <answer>...</answer>: [Core mechanism, Mechanism with bicatalytic steps, Mechanism with catalyst activation steps, Mechanism with catalyst deactivation steps]. 
            Do not provide final answer different than what it provided in this list. 

            # Reaction Categories Options
            All reactions are conversion from substrate S to product P with catalyst.

            1. Core mechanism
            Core mechanism is represented by the following reaction.
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2

            2. Mechanism with bicatalytic steps
            Examples:
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2cat<=>cat2;k3,k-3
            - S+cat2<=>((cat)2S);k1,k-1|((cat)2S)<=>P+cat2;k2,k-2|2cat<=>cat2;k3,k-3
            - X+catS<=>S+cat;k1,k-1|X+catS<=>P+cat;k2,k-2
            - S+cat<=>catS;k1,k-1|catS+cat<=>catP;k2,k-2|catP<=>P+cat;k3,k-3

            3 Mechanism with catalyst activation steps
            Examples:
            - cat<=>cat*;k1,0|S+cat*<=>cat*S;k1,k-1|cat*S<=>P+cat*;k2,k-2
            - S+cat<=>catS;k1,k-1|S+catS<=>catS2;k3,k-3|catS<=>P+cat;k2,k-2
            - S+cat*<=>cat*S;k1,k-1|cat*S<=>P+cat*;k2,k-2|cat+L<=>cat*;k3,k-3

            4. Mechanism with catalyst deactivation steps
            Examples:
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat<=>inactive cat;k3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|inhibitor+cat<=>inactive catI;k3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|S+cat<=>inactive catS;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|P+cat<=>inactive catP;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2cat<=>inactive cat2;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|catS<=>inactive catS;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|inhibitor+catS<=>inactive catSI;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|S+catS<=>inactive catS2;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|P+catS<=>inactive catSP;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2catS<=>inactive cat2S2;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat+catS<=>inactive cat2S;k3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat<=>inactive cat;k3,0|catS<=>inactive catS;k4,0
            
            {}
            <|im_end|>

            <|im_start|>assistant
            <think>
            """
        )

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

        prompts_train = []
        for i in range(self.x1_train.shape[0]):
            data = self.generate_data_pass_to_prompt(i, is_test=False)
            calculator = KineticMetricsCalculator(data)
            metrics = calculator.summarize_minimum_important_value()
            prompt = prompt_template_data_improved.format(**metrics)
            prompts_train.append(prompt)

        convert_from_class_to_category = {
            0: "Core mechanism",
            1: "Mechanism with bicatalytic steps",
            2: "Mechanism with bicatalytic steps",
            3: "Mechanism with bicatalytic steps",
            4: "Mechanism with bicatalytic steps",
            5: "Mechanism with catalyst activation steps",
            6: "Mechanism with catalyst activation steps",
            7: "Mechanism with catalyst activation steps",
            8: "Mechanism with catalyst deactivation steps",
            9: "Mechanism with catalyst deactivation steps",
            10: "Mechanism with catalyst deactivation steps",
            11: "Mechanism with catalyst deactivation steps",
            12: "Mechanism with catalyst deactivation steps",
            13: "Mechanism with catalyst deactivation steps",
            14: "Mechanism with catalyst deactivation steps",
            15: "Mechanism with catalyst deactivation steps",
            16: "Mechanism with catalyst deactivation steps",
            17: "Mechanism with catalyst deactivation steps",
            18: "Mechanism with catalyst deactivation steps",
            19: "Mechanism with catalyst deactivation steps",
        }

        train_dict = {
            "problem": prompts_train,
            "solution": [
                convert_from_class_to_category[y]
                for y in self.y_train.flatten().tolist()
            ],
            "options": [
                [
                    "Core mechanism",
                    "Mechanism with bicatalytic steps",
                    "Mechanism with catalyst activation steps",
                    "Mechanism with catalyst deactivation steps",
                ]
                for _ in range(self.x1_train.shape[0])
            ],
        }

        prompts_test = []
        for i in range(self.x1_test.shape[0]):
            data = self.generate_data_pass_to_prompt(i, is_test=True)
            calculator = KineticMetricsCalculator(data)
            metrics = calculator.summarize_minimum_important_value()
            prompt = prompt_template_data_improved.format(**metrics)
            prompts_test.append(prompt)

        test_dict = {
            "problem": prompts_test,
            "solution": [
                convert_from_class_to_category[y]
                for y in self.y_test.flatten().tolist()
            ],
            "options": [
                [
                    "Core mechanism",
                    "Mechanism with bicatalytic steps",
                    "Mechanism with catalyst activation steps",
                    "Mechanism with catalyst deactivation steps",
                ]
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

    def format_reward(self, completions, prompts, **kwargs):
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers

        Returns:
            list[float]: Reward scores
        """

        rewards = []
        completions = self.preprocess_completions(completions)

        for completion, prompt in zip(completions, prompts):
        # for completion in completions:
            try:
                if random.random() < 0.00:  # 0% chance to print a completion
                    print("===========prompt===========")
                    print(prompt)
                    print("===========completion===========")
                    print(completion)

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

    def format_continuous_reward(self, completions, prompts, **kwargs):
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

        rewards = []
        completions = self.preprocess_completions(completions)

        for completion, prompt in zip(completions, prompts):
            current_reward = 0.0
            try:
                if random.random() < 0.01:  # 1% chance to print a completion
                    print("===========prompt===========")
                    print(prompt)
                    print("===========completion===========")
                    print(completion)
                    
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
                # 0.1 reward if the completion starts with <think> and ends with </answer>
                if completion.startswith("<think>"):
                    current_reward += 0.05
                else:
                    current_reward -= 0.05
                if completion.endswith("</answer>"):
                    current_reward += 0.05
                else:
                    current_reward -= 0.05
                # 0.1 reward if the thinking is followed by an answer
                if completion.count("</think>\n<answer>") == 1:
                    current_reward += 0.1
                else:
                    current_reward -= 0.1
                # 0.2 reward is the answer is present
                pattern = r"<answer>(.*)<\/answer>"
                match = re.search(pattern, completion, re.DOTALL)
                if match is None:
                    current_reward -= 0.2
                elif len(match.groups()) != 1:
                    current_reward -= 0.05
                else:
                    current_reward += 0.2
                # 0.4 reward is the format is correct
                pattern = r"<think>(.*)<\/think>\n<answer>(.*)<\/answer>"
                match = re.search(pattern, completion, re.DOTALL)
                if match is None:
                    current_reward -= 0.4
                elif len(match.groups()) != 2:
                    current_reward -= 0.1
                else:
                    current_reward += 0.4
            except:
                pass
            rewards.append(current_reward)

        return rewards

    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - check that the answer is same as ground truth"""
        completions = self.preprocess_completions(completions)
        answers = self.extract_answer(completions)
        rewards = []

        for i in range(len(solution)):
            sol = solution[i]
            answer = answers[i]

            if sol == answer:
                rewards.append(1)
            else:
                rewards.append(0)

        return rewards

    def correct_option_reward(self, completions, solution, **kwargs):
        """Reward function - check that the answer is one of the correct options"""
        completions = self.preprocess_completions(completions)
        answers = self.extract_answer(completions)

        options = (
            "Core mechanism",
            "Mechanism with bicatalytic steps",
            "Mechanism with catalyst activation steps",
            "Mechanism with catalyst deactivation steps",
        )
        options = [option.lower() for option in options]
        
        rewards = []
        for ans in answers:
            if ans.lower() in options:
                rewards.append(1)
            else:
                rewards.append(0)

        rewards = [reward * 0.2 for reward in rewards]
        return rewards
                
    def run_coverage_reward(self, completions, solution, **kwargs):
        """Reward function - check that the answer considers the four experimental runs"""
        rewards = []
        for completion, sol in zip(completions, solution):
            data_mentioned = set(re.findall(r"run\s*([1-4])", completion.lower()))
            score_data_coverage = len(data_mentioned) / 4
            rewards.append(score_data_coverage)
        
        rewards = [reward * 0.2 for reward in rewards]
        return rewards
    
    def category_coverage_reward(self, completions, solution, **kwargs):
        """Reward function - check that the answer considers the four categories"""
        rewards = []
        categories = (
            "Core mechanism",
            "Mechanism with bicatalytic steps",
            "Mechanism with catalyst activation steps",
            "Mechanism with catalyst deactivation steps",
        )

        for completion in completions:
            categories_mentioned = {
                category for category in categories if category.lower() in completion.lower()
            }
            score_class_coverage = len(categories_mentioned) / len(categories)
            rewards.append(score_class_coverage)
        
        rewards = [reward * 0.2 for reward in rewards]
        return rewards

    def metrics_coverage_reward(self, completions, solution, **kwargs):
        rewards = []
        metrics = (
            "Initial concentration of catalyst",
            "The initial concentration of substrate",
            "The final concentration of substrate",
            "The initial concentration of product",
            "The final concentration of product",
            "Mass balance gap",
            "Mass balance gap at midpoint",
            "Catalyst stability",
            "Induction period",
        )

        for completion in completions:
            metrics_mentioned = {
                m for m in metrics if m.lower() in completion.lower()
            }
            score_metrics_coverage = len(metrics_mentioned) / len(metrics)
            rewards.append(score_metrics_coverage)
        
        rewards = [reward * 0.2 for reward in rewards]
        return rewards

    def answer_covered_in_reasoning_traces_reward(
        self, completions, solution, **kwargs
    ):
        """Reward function - check that the answer is covered in the reasoning traces"""
        rewards = []
        for sol, completion in zip(solution, completions):
            if sol in completion:
                rewards.append(0.2)
            else:
                rewards.append(0)

        return rewards

    def generate_prompt(self, problem, tokenizer, **kwargs):
        return {
            "prompt": self.question_template.format(problem),
            "problem": problem,
        }

    def dataset_preprocess(self, tokenizer):
        # カテゴリごとにサンプルをグループ化
        train_samples_by_category = {
            "Core mechanism": [],
            "Mechanism with bicatalytic steps": [],
            "Mechanism with catalyst activation steps": [],
            "Mechanism with catalyst deactivation steps": [],
        }

        # トレーニングデータをカテゴリごとに分類
        for i in range(len(self.dataset["train"])):
            category = self.dataset["train"][i]["solution"]
            train_samples_by_category[category].append(i)

        # 各カテゴリから均等にサンプルを選択
        min_samples_per_category = min(
            len(samples) for samples in train_samples_by_category.values()
        )
        selected_indices = []

        for category_samples in train_samples_by_category.values():
            # 各カテゴリからmin_samples_per_category個のサンプルをランダムに選択
            selected_indices.extend(
                random.sample(category_samples, min_samples_per_category)
            )

        # 選択したサンプルをシャッフル
        random.shuffle(selected_indices)

        # データセットを更新
        self.dataset["train"] = self.dataset["train"].select(selected_indices)

        # テストデータも同様に処理
        test_samples_by_category = {
            "Core mechanism": [],
            "Mechanism with bicatalytic steps": [],
            "Mechanism with catalyst activation steps": [],
            "Mechanism with catalyst deactivation steps": [],
        }

        for i in range(len(self.dataset["test"])):
            category = self.dataset["test"][i]["solution"]
            test_samples_by_category[category].append(i)

        min_test_samples_per_category = min(
            len(samples) for samples in test_samples_by_category.values()
        )
        selected_test_indices = []

        for category_samples in test_samples_by_category.values():
            selected_test_indices.extend(
                random.sample(category_samples, min_test_samples_per_category)
            )

        random.shuffle(selected_test_indices)
        self.dataset["test"] = self.dataset["test"].select(
            selected_test_indices
        )

        # プロンプトを生成
        self.dataset = self.dataset.map(
            lambda x: self.generate_prompt(x["problem"], tokenizer)
        )
        return self.dataset

    def extract_answer(self, completions: list[str]) -> list[str]:
        """Extract the answer from the completion."""
        answers = []
        for completion in completions:
            pattern = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
            m = re.search(pattern, completion, re.DOTALL)
            if m:
                ans = m.groups()[-1]
                if len(m.groups()) != 2:
                    answers.append("NONE")
                else:
                    answers.append(ans)
            else:
                answers.append("NONE")
        return answers

    def preprocess_completions(self, completions: list[str]) -> list[str]:
        """Add <think> tags to the completions."""
        processed_completions = []
        for completion in completions:
            processed_completions.append(f"<think>{completion}")
        return processed_completions


class KineticDataCategoryClassificationWithRawDataMetrics(
    KineticDataCategoryClassificationWithMetrics
):
    question_template: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.question_template = ("""
            <|im_start|>assistant
            You are a useful Chemistry assistant and you will answer the following class prediction question. 
            Please begin your response with "<think>", then provide a detailed, step-by-step reasoning process.
            Then end with </think>, and finally put your final answer within <answer>...</answer> tags, for example <answer>Core Mechanism</answer>.
            
            <|im_start|>user
            Estimate the reaction categories based on the following data and metrics.
            Choose ONLY from the following options and write your response choice inside <answer>...</answer>: [Core mechanism, Mechanism with bicatalytic steps, Mechanism with catalyst activation steps, Mechanism with catalyst deactivation steps]. 
            Do not provide final answer different than what it provided in this list. 

            # Reaction Categories Options
            All reactions are conversion from substrate S to product P with catalyst.

            1. Core mechanism
            Core mechanism is represented by the following reaction.
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2

            2. Mechanism with bicatalytic steps
            Examples:
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2cat<=>cat2;k3,k-3
            - S+cat2<=>((cat)2S);k1,k-1|((cat)2S)<=>P+cat2;k2,k-2|2cat<=>cat2;k3,k-3
            - X+catS<=>S+cat;k1,k-1|X+catS<=>P+cat;k2,k-2
            - S+cat<=>catS;k1,k-1|catS+cat<=>catP;k2,k-2|catP<=>P+cat;k3,k-3

            3 Mechanism with catalyst activation steps
            Examples:
            - cat<=>cat*;k1,0|S+cat*<=>cat*S;k1,k-1|cat*S<=>P+cat*;k2,k-2
            - S+cat<=>catS;k1,k-1|S+catS<=>catS2;k3,k-3|catS<=>P+cat;k2,k-2
            - S+cat*<=>cat*S;k1,k-1|cat*S<=>P+cat*;k2,k-2|cat+L<=>cat*;k3,k-3

            4. Mechanism with catalyst deactivation steps
            Examples:
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat<=>inactive cat;k3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|inhibitor+cat<=>inactive catI;k3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|S+cat<=>inactive catS;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|P+cat<=>inactive catP;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2cat<=>inactive cat2;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|catS<=>inactive catS;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|inhibitor+catS<=>inactive catSI;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|S+catS<=>inactive catS2;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|P+catS<=>inactive catSP;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|2catS<=>inactive cat2S2;k-3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat+catS<=>inactive cat2S;k3,0
            - S+cat<=>catS;k1,k-1|catS<=>P+cat;k2,k-2|cat<=>inactive cat;k3,0|catS<=>inactive catS;k4,0
            
            {}
            <|im_end|>

            <|im_start|>assistant
            <think>"""
        )

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

        prompts_train = []
        for i in range(self.x1_train.shape[0]):
            data = self.generate_data_pass_to_prompt(i, is_test=False)
            calculator = KineticMetricsCalculator(data)
            # calculator.process_sample()
            metrics = calculator.summarize_minimum_important_value()

            for key, value in metrics.items():
                for k, v in value.items():
                    data[key][k] = v

            prompt = prompt_template_with_raw_data.format(**data)
            prompts_train.append(prompt)

        convert_from_class_to_category = {
            0: "Core mechanism",
            1: "Mechanism with bicatalytic steps",
            2: "Mechanism with bicatalytic steps",
            3: "Mechanism with bicatalytic steps",
            4: "Mechanism with bicatalytic steps",
            5: "Mechanism with catalyst activation steps",
            6: "Mechanism with catalyst activation steps",
            7: "Mechanism with catalyst activation steps",
            8: "Mechanism with catalyst deactivation steps",
            9: "Mechanism with catalyst deactivation steps",
            10: "Mechanism with catalyst deactivation steps",
            11: "Mechanism with catalyst deactivation steps",
            12: "Mechanism with catalyst deactivation steps",
            13: "Mechanism with catalyst deactivation steps",
            14: "Mechanism with catalyst deactivation steps",
            15: "Mechanism with catalyst deactivation steps",
            16: "Mechanism with catalyst deactivation steps",
            17: "Mechanism with catalyst deactivation steps",
            18: "Mechanism with catalyst deactivation steps",
            19: "Mechanism with catalyst deactivation steps",
        }

        train_dict = {
            "problem": prompts_train,
            "solution": [
                convert_from_class_to_category[y]
                for y in self.y_train.flatten().tolist()
            ],
            "options": [
                [
                    "Core mechanism",
                    "Mechanism with bicatalytic steps",
                    "Mechanism with catalyst activation steps",
                    "Mechanism with catalyst deactivation steps",
                ]
                for _ in range(self.x1_train.shape[0])
            ],
        }

        prompts_test = []
        for i in range(self.x1_test.shape[0]):
            data = self.generate_data_pass_to_prompt(i, is_test=True)
            calculator = KineticMetricsCalculator(data)
            metrics = calculator.summarize_minimum_important_value()

            for key, value in metrics.items():
                for k, v in value.items():
                    data[key][k] = v

            prompt = prompt_template_with_raw_data.format(**data)
            prompts_test.append(prompt)

        test_dict = {
            "problem": prompts_test,
            "solution": [
                convert_from_class_to_category[y]
                for y in self.y_test.flatten().tolist()
            ],
            "options": [
                [
                    "Core mechanism",
                    "Mechanism with bicatalytic steps",
                    "Mechanism with catalyst activation steps",
                    "Mechanism with catalyst deactivation steps",
                ]
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
