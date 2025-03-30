from open_r1.tasks.base import RLTask
from datasets import DatasetDict

class KineticDataClassification(RLTask):
    """
    Description of your new task.
    
    This task should [describe what the task does and its purpose].
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = """
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
        """
        
    def load(self) -> DatasetDict:
        """
        Load and prepare the dataset for the task.
        
        Returns:
            DatasetDict: Dataset with 'train' and 'test' splits
        """

        # Assume the dataset is in the same directory as the script
        x1_train_path = os.path.join(self.dataset_id_or_path, "x_train", "x1_train_M1_M20_train_val_test_set.pkl")
        x2_train_path = os.path.join(self.dataset_id_or_path, "x_train", "x2_train_M1_M20_train_val_test_set.pkl")
        y_train_path = os.path.join(self.dataset_id_or_path, "y_train", "y_train_M1_M20_train_val_test_set.pkl")

        x1_test_path = os.path.join(self.dataset_id_or_path, "x_val", "x1_val_M1_M20_train_val_test_set.pkl")
        x2_test_path = os.path.join(self.dataset_id_or_path, "x_val", "x2_val_M1_M20_train_val_test_set.pkl")
        y_test_path = os.path.join(self.dataset_id_or_path, "y_val", "y_val_M1_M20_train_val_test_set.pkl")

        # Implement dataset loading logic
        with open(x1_train_path, "rb") as f:
            x1_train = pickle.load(f)
        with open(x2_train_path, "rb") as f:
            x2_train = pickle.load(f)
        with open(y_train_path, "rb") as f:
            y_train = pickle.load(f)
        
        y_train = y_train.reshape(-1, 1)

        with open(x1_test_path, "rb") as f:
            x1_test = pickle.load(f)
        with open(x2_test_path, "rb") as f:
            x2_test = pickle.load(f)
        with open(y_test_path, "rb") as f:
            y_test = pickle.load(f)

        y_test = y_test.reshape(-1, 1)

        prompt_template_data = f"""
        # Data Run 1
        - Initial concentration of catalyst (normalized to [S]0): {run_1[initial_concentration_of_catalyst]}
        - Initial concentration of substrate (normalized to [S]0): {run_1[substrate_data][0]}
        - Initial concentration of ES: 0.0
        - Initial concentration of product (normalized to [S]0): {run_1[product_data][0]}
        - Time_data (normalized, unitless): {run_1[time_data]}
        - Substrate_data (normalized to [S]0): {run_1[substrate_data]}
        - Product_data (normalized to [S]0): {run_1[product_data]}

        # Data Run 2
        - Initial concentration of catalyst (normalized to [S]0): {run_2[initial_concentration_of_catalyst]}
        - Initial concentration of substrate (normalized to [S]0): {run_2[substrate_data][0]}
        - Initial concentration of ES: 0.0
        - Initial concentration of product (normalized to [S]0): {run_2[product_data][0]}
        - Time_data (normalized, unitless): {run_2[time_data]}
        - Substrate_data (normalized to [S]0): {run_2[substrate_data]}
        - Product_data (normalized to [S]0): {run_2[product_data]}

        # Data Run 3
        - Initial concentration of catalyst (normalized to [S]0): {run_3[initial_concentration_of_catalyst]}
        - Initial concentration of substrate (normalized to [S]0): {run_3[substrate_data][0]}
        - Initial concentration of ES: 0.0
        - Initial concentration of product (normalized to [S]0): {run_3[product_data][0]}
        - Time_data (normalized, unitless): {run_3[time_data]}
        - Substrate_data (normalized to [S]0): {run_3[substrate_data]}
        - Product_data (normalized to [S]0): {run_3[product_data]}

        # Data Run 4
        - Initial concentration of catalyst (normalized to [S]0): {run_4[initial_concentration_of_catalyst]}
        - Initial concentration of substrate (normalized to [S]0): {run_4[substrate_data][0]}
        - Initial concentration of ES: 0.0
        - Initial concentration of product (normalized to [S]0): {run_4[product_data][0]}
        - Time_data (normalized, unitless): {run_4[time_data]}
        - Substrate_data (normalized to [S]0): {run_4[substrate_data]}
        - Product_data (normalized to [S]0): {run_4[product_data]}
        """

        train_dict = {
            "prompt_data": [
                prompt_template_data.format(**self.generate_data_pass_to_prompt(i, is_test=False)) 
                for i in range(x_train.shape[0])
            ],
            "solution": y_train.tolist()
        }

        test_dict = {
            "prompt_data": [
                prompt_template_data.format(**self.generate_data_pass_to_prompt(i, is_test=True)) 
                for i in range(x_test.shape[0])
            ],
            "solution": y_test.tolist()
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
        """
        Calculate rewards for model completions.
        
        Args:
            completions (List[str]): Model generated responses
            solution (List[str]): Ground truth solutions
            
        Returns:
            List[float]: Rewards for each completion
        """
        rewards = []
        for completion, solution in zip(completions, solution):
            final_answer = self.extract_final_answer(completion)
            if completion == final_answer:
                rewards.append(1)
            else:
                rewards.append(0)
        return rewards

    def extract_final_answer(self, completion):
        prompt = """Extract the final reaction class and categorize other considered mechanisms into 'possible' (plausible alternatives) and 'rejected' (explicitly discussed and rejected). 

        Provide the answer in the format: 
        'Final: M1, Possible: M3 M4, Rejected: M2 M5'

        Rules:
        1. The final answer should be a single mechanism (e.g., M1, M2, etc.)
        2. 'Possible' mechanisms are those that were discussed as plausible alternatives but not chosen as final
        3. 'Rejected' mechanisms are those that were explicitly discussed and ruled out
        4. Format must be exactly: 'Final: Mx, Possible: My Mz, Rejected: Ma Mb'
        5. If no other mechanisms were considered in either category, omit that category
        e.g., 'Final: M1, Rejected: M2' (if no possible alternatives)
        e.g., 'Final: M1, Possible: M2' (if no rejected mechanisms)
        e.g., 'Final: M1' (if no other mechanisms were considered)

        Example outputs:
        - 'Final: M1, Possible: M3 M4, Rejected: M2 M5'
        - 'Final: M2, Possible: M1, Rejected: M3'
        - 'Final: M1, Rejected: M2 M3'
        - 'Final: M3, Possible: M1 M2'
        - 'Final: M1'
        """

        chat_completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # 決定論的な出力を得るため
                max_tokens=100  # 短い応答で十分
        )
        extracted_result = chat_completion.choices[0].message.content.strip()
        final_answer, possible_mechanisms, rejected_mechanisms = self._parse_evaluation_result(extracted_result)
        return final_answer            

    def _parse_evaluation_result(self, result: str) -> Tuple[str, List[str], List[str]]:
        """
        Parse the evaluation result string into final answer, possible mechanisms, and rejected mechanisms.
        
        Args:
            result: String in format 'Final: M1, Possible: M3 M4, Rejected: M2 M5'
            
        Returns:
            Tuple containing:
            - final_answer (str): The final mechanism (e.g. 'M1')
            - possible_mechanisms (List[str]): List of possible mechanisms
            - rejected_mechanisms (List[str]): List of rejected mechanisms
        """
        final_answer = ""
        possible_mechanisms = []
        rejected_mechanisms = []
        
        # Extract final answer
        if "Final:" in result:
            final_part = result.split("Final:")[1].split(",")[0].strip()
            final_answer = final_part
            
        # Extract possible mechanisms
        if "Possible:" in result:
            possible_part = result.split("Possible:")[1].split(",")[0].strip()
            possible_mechanisms = possible_part.split()
            
        # Extract rejected mechanisms
        if "Rejected:" in result:
            rejected_part = result.split("Rejected:")[1].split(",")[0].strip()
            rejected_mechanisms = rejected_part.split()
        
        return final_answer, possible_mechanisms, rejected_mechanisms


    def dataset_preprocess(self, tokenizer):
        return self.dataset