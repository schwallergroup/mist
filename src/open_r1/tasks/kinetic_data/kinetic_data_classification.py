import os
import re
import pickle
from typing import List

from open_r1.tasks.base import RLTask
from datasets import DatasetDict, Dataset

class KineticDataClassification(RLTask):
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
        Reason and estimate the reaction class for the following reaction.
        The possible reaction classes are M1 to M20 indicated as follows.
        Please begin your response with "<think>", then provide a detailed, step-by-step reasoning process (including any intermediate reflections or re-evaluations), 
        and finally put your final answer within \\boxed{{}}.

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
        """)
        
    def load(self) -> DatasetDict:
        """
        Load and prepare the dataset for the task.
        
        Returns:
            DatasetDict: Dataset with 'train' and 'test' splits
        """

        # Assume the dataset is in the same directory as the script
        x1_train_path = os.path.join(self.dataset_id_or_path, "x1_train_M1_M20_train_val_test_set_part_0.pkl")
        x2_train_path = os.path.join(self.dataset_id_or_path, "x2_train_M1_M20_train_val_test_set_part_0.pkl")
        y_train_path = os.path.join(self.dataset_id_or_path, "y_train_M1_M20_train_val_test_set_part_0.pkl")

        x1_test_path = os.path.join(self.dataset_id_or_path, "x1_val_M1_M20_train_val_test_set_part_0.pkl")
        x2_test_path = os.path.join(self.dataset_id_or_path, "x2_val_M1_M20_train_val_test_set_part_0.pkl")
        y_test_path = os.path.join(self.dataset_id_or_path, "y_val_M1_M20_train_val_test_set_part_0.pkl")

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

        self.y_test = self.y_test.reshape(-1, 1)[:self.x1_test.shape[0]]

        prompt_template_data = f"""
        # Data Run 1
        - Initial concentration of catalyst (normalized to [S]0): {{run_1[initial_concentration_of_catalyst]}}
        - Initial concentration of substrate (normalized to [S]0): {{run_1[substrate_data][0]}}
        - Initial concentration of ES: 0.0
        - Initial concentration of product (normalized to [S]0): {{run_1[product_data][0]}}
        - Time_data (normalized, unitless): {{run_1[time_data]}}
        - Substrate_data (normalized to [S]0): {{run_1[substrate_data]}}
        - Product_data (normalized to [S]0): {{run_1[product_data]}}

        # Data Run 2
        - Initial concentration of catalyst (normalized to [S]0): {{run_2[initial_concentration_of_catalyst]}}
        - Initial concentration of substrate (normalized to [S]0): {{run_2[substrate_data][0]}}
        - Initial concentration of ES: 0.0
        - Initial concentration of product (normalized to [S]0): {{run_2[product_data][0]}}
        - Time_data (normalized, unitless): {{run_2[time_data]}}
        - Substrate_data (normalized to [S]0): {{run_2[substrate_data]}}
        - Product_data (normalized to [S]0): {{run_2[product_data]}}

        # Data Run 3
        - Initial concentration of catalyst (normalized to [S]0): {{run_3[initial_concentration_of_catalyst]}}
        - Initial concentration of substrate (normalized to [S]0): {{run_3[substrate_data][0]}}
        - Initial concentration of ES: 0.0
        - Initial concentration of product (normalized to [S]0): {{run_3[product_data][0]}}
        - Time_data (normalized, unitless): {{run_3[time_data]}}
        - Substrate_data (normalized to [S]0): {{run_3[substrate_data]}}
        - Product_data (normalized to [S]0): {{run_3[product_data]}}

        # Data Run 4
        - Initial concentration of catalyst (normalized to [S]0): {{run_4[initial_concentration_of_catalyst]}}
        - Initial concentration of substrate (normalized to [S]0): {{run_4[substrate_data][0]}}
        - Initial concentration of ES: 0.0
        - Initial concentration of product (normalized to [S]0): {{run_4[product_data][0]}}
        - Time_data (normalized, unitless): {{run_4[time_data]}}
        - Substrate_data (normalized to [S]0): {{run_4[substrate_data]}}
        - Product_data (normalized to [S]0): {{run_4[product_data]}}
        """

        train_dict = {
            "problem": [
                prompt_template_data.format(**self.generate_data_pass_to_prompt(i, is_test=False)) 
                for i in range(self.x1_train.shape[0])
            ],
            "solution": ["M" + str(int(y[0]) + 1) for y in self.y_train.tolist()]
        }

        test_dict = {
            "problem": [
                prompt_template_data.format(**self.generate_data_pass_to_prompt(i, is_test=True)) 
                for i in range(self.x1_test.shape[0])
            ],
            "solution": ["M" + str(int(y[0]) + 1) for y in self.y_test.tolist()]
        }

        self.dataset = DatasetDict({"train": Dataset.from_dict(train_dict), "test": Dataset.from_dict(test_dict)})
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
                "product_data": x2_data[index, :, 2].tolist()
            },
            "run_2": {
                "initial_concentration_of_catalyst": float(x1_data[index, 1]),
                "time_data": x2_data[index, :, 3].tolist(),
                "substrate_data": x2_data[index, :, 4].tolist(),
                "product_data": x2_data[index, :, 5].tolist()
            },
            "run_3": {
                "initial_concentration_of_catalyst": float(x1_data[index, 2]),
                "time_data": x2_data[index, :, 6].tolist(),
                "substrate_data": x2_data[index, :, 7].tolist(),
                "product_data": x2_data[index, :, 8].tolist()
            },
            "run_4": {
                "initial_concentration_of_catalyst": float(x1_data[index, 3]),
                "time_data": x2_data[index, :, 9].tolist(),
                "substrate_data": x2_data[index, :, 10].tolist(),
                "product_data": x2_data[index, :, 11].tolist()
            }
        }

    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - check that the answer is same as ground truth
        """
        answers = [self.preprocess_response(c) for c in completions]
        rewards = []

        for answer, sol in zip(answers, solution):
            if sol == answer:
                rewards.append(1)
            else:
                rewards.append(0)
        return rewards
    
    def generate_prompt(self, problem, tokenizer, **kwargs):
        """Generate prompt for the MCQA task."""
        r1_prefix = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.question_template.format(problem),
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True
            ),
            "problem": problem
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
            lambda x: self.generate_prompt(
                x["problem"], tokenizer
            )
        )
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


if __name__ == "__main__":
    task = KineticDataClassification(dataset_id_or_path="/work/liac/kinetic")
    task.load()
    print(task.dataset)