from ..base import RLTask
from ..task_utils import compute_lcs_length, compute_levenshtein_distance, compute_tanimoto_similarity
import re
import random
from datasets import Dataset, DatasetDict
from rdkit import RDLogger, Chem
import pandas as pd
import numpy as np
import json
from dataclasses import field
RDLogger.DisableLog('rdApp.*')


class SmilesHydrogen(RLTask):
    question_template_addH: str = ""
    question_template_removeH: str = ""
    log_custom_metrics: bool = True
    custom_metrics: dict = field(default_factory=dict)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define prompt guiding options
        with open('smiles_hydrogen_prompt_guiding.json.json', 'r') as f:
            prompt_modes = json.load(f)

        pg_addH = ""
        pg_removeH = ""
        if self.task_mode in prompt_modes:
            pg_addH = prompt_modes[self.task_mode]['addH']
            pg_removeH = prompt_modes[self.task_mode]['removeH']

        self.question_template_addH = (
            "What is the SMILES for this molecule after adding all the implicit hydrogens? Here is a SMILES without hydrogens: {} "
            f"The order of the atoms in your answer should be the same as in the input SMILES. {pg_addH}"
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer>CN1C=C...</answer>. Think step by step inside <think> tags.\n"
        )
        self.question_template_removeH = (
            "What is the SMILES for this molecule after removing all the implicit hydrogens? Here is a SMILES with hydrogens: {} "
            f"The order of the atoms in your answer should be the same as in the input SMILES. {pg_removeH}"
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer>CN1C=C...</answer>. Think step by step inside <think> tags.\n"
        )
        self.log_custom_metrics = True
        self.custom_metrics = {
            'n_samples': 0,
            'addH/accuracy_reward': [],
            'removeH/accuracy_reward': [],
            'addH/smiles_validity_reward': [],
            'removeH/smiles_validity_reward': [],
            'addH/tanimoto_accuracy_reward': [],
            'removeH/tanimoto_accuracy_reward': [],
            'addH/levenstein_accuracy_reward': [],
            'removeH/levenstein_accuracy_reward': [],
            'addH/sequential_reward': [],
            'removeH/sequential_reward': [],
        }
    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M_hydrogen.csv

        df = pd.read_csv(self.dataset_id_or_path)
        # Only keep the rows where SMILES_Hs only contains character insertions compared to SMILES_noHs
        df = df[df['levenshtein_distance'] == df['length_diff']]

        if self.task_mode in ["base", "PG1", "PG2", "PG3"]:
            train_dict = {
                'problem': df['SMILES_noHs'].tolist() + df['SMILES_Hs'].tolist(),
                'solution': df['SMILES_Hs'].tolist() + df['SMILES_noHs'].tolist(),
                'question_category': len(df) * ['addH'] + len(df) * ['removeH'],
            }
        elif self.task_mode == "addH":
            train_dict = {
                'problem': df['SMILES_noHs'].tolist(),
                'solution': df['SMILES_Hs'].tolist(),
                'question_category': len(df) * ['addH'],
            }
        elif self.task_mode == "removeH":
            train_dict = {
                'problem': df['SMILES_Hs'].tolist(),
                'solution': df['SMILES_noHs'].tolist(),
                'question_category': len(df) * ['removeH'],
            }
        else:
            raise ValueError(f"Task mode not recognized: {self.task_mode}")
        train_dataset = Dataset.from_dict(train_dict)
        train_test_split = train_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        # Combine into DatasetDict
        self.dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        return self.dataset
    def generate_prompt(self, problem, question_category, tokenizer, **kwargs):
        if question_category == 'addH':
            question_template = self.question_template_addH
        elif question_category == 'removeH':
            question_template = self.question_template_removeH
        else:
            raise ValueError(f"Dataset 'question_category' not recognized: {question_category}")
        r1_prefix = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question_template.format(problem)},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "problem": problem,
        }
    def dataset_preprocess(self, tokenizer):
        self.dataset["train"] = self.dataset["train"].shuffle(seed=42).select(range(min(100_000, len(self.dataset["train"]))))
        self.dataset["test"] = self.dataset["test"].shuffle(seed=42).select(range(min(20_000, len(self.dataset["test"]))))
        self.dataset = self.dataset.map(lambda x: self.generate_prompt(x["problem"], x["question_category"], tokenizer))
        return self.dataset
    def preprocess_answer(self, response):
        """ Preprocess the answer before checking for accuracy. """
        pattern = r"<answer>(.*)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            smi = m.groups()[0]
            # Maybe smiles contains [BEGIN_SMILES] and [END_SMILES]
            if "[BEGIN_SMILES]" in smi:
                if "[END_SMILES]" in smi[smi.index("[BEGIN_SMILES]"):]:
                    smi = smi[smi.index("[BEGIN_SMILES]") + len("[BEGIN_SMILES]"):smi.index("[END_SMILES]")]
            return smi
        else:
            return "NONE"
    def preprocess_think(self, response):
        """ Preprocess the think before checking for accuracy. """
        pattern = r"<think>(.*)<\/think>"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            return m.groups()[0]
        else:
            return "NONE"
    def format_reward(self, completions, problem, solution, question_category, **kwargs):
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

        for completion_id, completion in enumerate(completions):
            current_reward = 0.0
            try:
                # 0.2 reward if each tag is present once
                for tag_word in ["<think>", "</think>", "<answer>", "</answer>"]:
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
                if completion.count('</think>\n<answer>') == 1:
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

            # Randomly print a completion
            if random.random() < 0.01:  # 1% chance to print a completion
                # Compute the other metrics (while logging is disabled)
                self.log_custom_metrics = False
                kwargs = {
                    'completions': [completion],
                    'problem': [problem[completion_id]],
                    'solution': [solution[completion_id]],
                    'question_category': [question_category[completion_id]],
                }
                _format_reward = f"{current_reward:.2f}"
                _reasoning_steps_reward = f"{self.reasoning_steps_reward(**kwargs)[0]:.2f}"
                _accuracy_reward = f"{self.accuracy_reward(**kwargs)[0]:.2f}"
                _smiles_validity_reward = f"{self.smiles_validity_reward(**kwargs)[0]:.2f}"
                _tanimoto_accuracy_reward = f"{self.tanimoto_accuracy_reward(**kwargs)[0]:.2f}"
                _levenstein_accuracy_reward = f"{self.levenstein_accuracy_reward(**kwargs)[0]:.2f}"
                self.log_custom_metrics = True
                # Print
                print(
                    f"\n\n=======<RANDOM_RESPONSE>=======\n"
                    f"# Question type: {question_category[completion_id]}\n"
                    f"# Problem: {problem[completion_id]}\n"
                    f"# Solution: {solution[completion_id]}\n"
                    f"# Rewards:\n"
                    f"# - format [-1,1]:                {_format_reward.rjust(5)}\n"
                    f"# - reasoning_steps [0,1]:        {_reasoning_steps_reward.rjust(5)}\n"
                    f"# - accuracy [0,1]:               {_accuracy_reward.rjust(5)}\n"
                    f"# - smiles_validity [-0.5,0]:     {_smiles_validity_reward.rjust(5)}\n"
                    f"# - tanimoto_accuracy [-0.5,1]:   {_tanimoto_accuracy_reward.rjust(5)}\n"
                    f"# - levenstein_accuracy [-0.5,1]: {_levenstein_accuracy_reward.rjust(5)}\n"
                    f"{completion}"
                )

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
        # Reward goal: foster clear step-by-step reasoning (inside think tags only)
        # Reward range: 0 to 1

        thoughts = [self.preprocess_think(c) for c in completions]

        pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
        matches = [len(re.findall(pattern, content)) for content in thoughts]

        # Magic number 3 to encourage 3 steps and more, otherwise partial reward
        return [min(1.0, count / 3) for count in matches]
    def accuracy_reward(self, completions, solution, question_category, **kwargs):
        """ Reward function - check that answer is same as solution. """
        # Reward goal: if the answer is perfectly correct, reward 1.0
        # Reward range: 0 to 1

        answers = [self.preprocess_answer(c) for c in completions]
        rewards = []

        for content, sol in zip(answers, solution):
            if content == sol:
                rewards.append(1)
            else:
                rewards.append(0)

        # Logging custom metrics
        if self.log_custom_metrics:
            self.custom_metrics['n_samples'] += len(completions)
            self.custom_metrics['addH/accuracy_reward'].extend([r for r, c in zip(rewards, question_category) if c == 'addH'])
            self.custom_metrics['removeH/accuracy_reward'].extend([r for r, c in zip(rewards, question_category) if c == 'removeH'])

        return rewards
    def smiles_validity_reward(self, completions, question_category, **kwargs):
        """ Reward function - check that answer contain a valid SMILES. """
        # Reward goal: ensure that the answer is a valid SMILES
        # Reward range: -0.5 to 0

        answers = [self.preprocess_answer(c) for c in completions]

        rewards = []
        for sol in answers:
            try:
                mol = Chem.MolFromSmiles(sol)
                if mol is None:
                    rewards.append(-0.5)
                else:
                    rewards.append(0)
            except:
                rewards.append(-0.5)

        # Logging custom metrics
        if self.log_custom_metrics:
            self.custom_metrics['addH/smiles_validity_reward'].extend([r for r, c in zip(rewards, question_category) if c == 'addH'])
            self.custom_metrics['removeH/smiles_validity_reward'].extend([r for r, c in zip(rewards, question_category) if c == 'removeH'])

        return rewards
    def tanimoto_accuracy_reward(self, completions, solution, question_category, **kwargs):
        """ Reward function - compute Tanimoto similarity reward between answer and solution. """
        # Reward goal: foster answers with high Tanimoto similarity to the solution
        # Reward range: -0.5 to 1

        answers = [self.preprocess_answer(c) for c in completions]

        rewards = []
        for content, sol in zip(answers, solution):
            try:
                # Scale the reward:
                # 1.0 for perfect match
                # Proportional to similarity for partial matches (-0.5 to 0.5)
                tanimoto_similarity = compute_tanimoto_similarity(sol, content)
                if tanimoto_similarity is None:
                    reward = 0.0  # invalid smiles in the answer (already penalized in smiles_validity_reward)
                elif tanimoto_similarity == 1.0:
                    reward = 1.0
                else:
                    reward = tanimoto_similarity - 0.5  # Shifts the reward to be negative for very low similarities
                rewards.append(reward)
            except:
                rewards.append(0.0)

        # Logging custom metrics
        if self.log_custom_metrics:
            self.custom_metrics['addH/tanimoto_accuracy_reward'].extend([r for r, c in zip(rewards, question_category) if c == 'addH'])
            self.custom_metrics['removeH/tanimoto_accuracy_reward'].extend([r for r, c in zip(rewards, question_category) if c == 'removeH'])

        return rewards
    def levenstein_accuracy_reward(self, completions, problem, solution, question_category, **kwargs):
        """ Reward function - compute Levenshtein distance reward between answer and solution. """
        # Reward goal: foster answers that are adding or removing hydrogens correctly
        # Reward range: -0.5 to 1

        answers = [self.preprocess_answer(c) for c in completions]

        rewards = []
        for content, prob, sol, cat in zip(answers, problem, solution, question_category):
            # Compute Levenshtein distance
            levenshtein_distance_problem_solution = compute_levenshtein_distance(prob, sol)
            levenshtein_distance_input = compute_levenshtein_distance(prob, content)
            levenshtein_distance_solution = compute_levenshtein_distance(sol, content)

            # Step 1: set the reward in function of the levenshtein distances
            if levenshtein_distance_problem_solution == 0:
                # Case when input = solution
                if levenshtein_distance_solution == 0:
                    reward = 1.0
                else:
                    reward = -0.5
            elif levenshtein_distance_solution < levenshtein_distance_problem_solution:
                # Reward from 0.2 to 1.0 if the answer contains correctly inferred information
                reward = max(1 - levenshtein_distance_solution / levenshtein_distance_problem_solution, 0.2)
            elif levenshtein_distance_input == 0:
                # Small reward if the answer is a copy-paste of the input
                reward = 0.1
            else:
                # Penalize -0.5 if the answer is further to the solution than the input
                reward = -0.5
            # Step 2: penalize if the SMILES without the hydrogens is not a subsequence of the answer
            if cat == 'addH':
                _SMILES_noHs = prob
            elif cat == 'removeH':
                _SMILES_noHs = sol
            else:
                raise ValueError(f"Dataset 'question_category' not recognized: {cat}")
            lcs_length = compute_lcs_length(_SMILES_noHs, content)
            missing_characters = len(_SMILES_noHs) - lcs_length
            reward -= 0.1 * missing_characters  # Remove 0.1 for each missing character
            reward = max(reward, -0.5)  # Ensure the reward is not lower than -0.5

            rewards.append(reward)

        # Logging custom metrics
        if self.log_custom_metrics:
            self.custom_metrics['addH/levenstein_accuracy_reward'].extend([r for r, c in zip(rewards, question_category) if c == 'addH'])
            self.custom_metrics['removeH/levenstein_accuracy_reward'].extend([r for r, c in zip(rewards, question_category) if c == 'removeH'])

        return rewards
    def sequential_reward(self, completions, problem, solution, question_category, **kwargs):
        reward_funcs_ordered = [
            self.format_reward,
            self.reasoning_steps_reward,
            self.levenstein_accuracy_reward,
            self.smiles_validity_reward,
            self.tanimoto_accuracy_reward,
            self.accuracy_reward,
        ]
        reward_rescaling_funcs = [
            lambda x: (x+1)/2,  # format_reward [-1, 1] -> [0, 1]
            lambda x: x,  # reasoning_steps_reward [0, 1] -> [0, 1]
            lambda x: (x+0.5)/1.5,  # levenstein_accuracy_reward [-0.5, 1] -> [0, 1]
            lambda x: (x+0.5)/0.5,  # smiles_validity_reward [-0.5, 0] -> [0, 1]
            lambda x: np.maximum(x, 0),  # tanimoto_accuracy_reward [-0.5, 1] -> [0, 1]
            lambda x: x,  # accuracy_reward [0, 1] -> [0, 1]
        ]

        # Compute rewards
        reward_kwargs = {
            'completions': completions,
            'problem': problem,
            'solution': solution,
            'question_category': question_category,
            **kwargs,
        }
        rewards = [np.array(reward_func(**reward_kwargs)) for reward_func in reward_funcs_ordered]

        # Compute rewards_rescaled (rescaled to [0, 1])
        rewards_rescaled = [reward_rescaling_func(_rewards) for reward_rescaling_func, _rewards in zip(reward_rescaling_funcs, rewards)]

        # Sum rewards - compute the reward scaling factors
        n_rewards = len(reward_funcs_ordered)
        n_samples = len(completions)
        reward_scaling_factors = np.ones((n_rewards, n_samples), dtype=float)
        for i in range(n_rewards-1):
            reward_scaling_factors[i+1:] = reward_scaling_factors[i+1:] * rewards_rescaled[i].reshape(1, -1)
        # Sum rewards - compute final rewards
        sequential_rewards = np.sum(np.stack(rewards, axis=0) * reward_scaling_factors, axis=0)
        sequential_rewards = sequential_rewards.tolist()

        # Logging custom metrics
        if self.log_custom_metrics:
            self.custom_metrics['addH/sequential_reward'].extend([r for r, c in zip(sequential_rewards, question_category) if c == 'addH'])
            self.custom_metrics['removeH/sequential_reward'].extend([r for r, c in zip(sequential_rewards, question_category) if c == 'removeH'])

        return sequential_rewards
    def get_metrics(self) -> dict:
        """
        Get task metrics to log in WANDB.
        This function takes no arguments and returns a dictionary of metrics {key[str]: value[float]}.
        """
        metrics = dict()
        if self.log_custom_metrics:
            # Log the number of samples that have been processed
            if self.custom_metrics['n_samples'] > 0:
                metrics['n_samples'] = self.custom_metrics['n_samples']
            # Iterate over the custom metrics and compute the average
            for metric_name in [m for m in self.custom_metrics.keys() if m not in ['n_samples']]:
                if len(self.custom_metrics[metric_name]) > 0:
                    metrics[metric_name] = sum(self.custom_metrics[metric_name]) / len(self.custom_metrics[metric_name])
                    # Reset the metric
                    self.custom_metrics[metric_name] = []
        return metrics

