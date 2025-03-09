
from ..base import RLTask
import numpy as np
import re
import collections
import random
from datasets import Dataset, DatasetDict
import pandas as pd

class CanonicalizeSmilesMCQA(RLTask):
    question_template: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "What is the canonical SMILES for this molecule? Here is a non-canonical SMILES: {} "
            "Choose from the following options. Options: \nA. {}\nB. {}\nC. {}\nD. {}\n"
            "Respond with the option letter inside <answer> </answer> tags. (A, B, C, or D)."
        )
        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)

        if self.task_variant == 'base':
            shuffled = [np.random.permutation(row).tolist() for row in df[['SMILES_variant2', 'SMILES_variant3', 'SMILES_variant4', "SMILES"]].values]
        elif self.task_variant == 'options_sameformula_canon':
            # Drop duplicates (same SMILES)
            df = df.drop_duplicates(subset='SMILES', keep=False)
            # Drop compounds with rare molecular formulas (less than 4 occurrences)
            molecular_formula_dropped = {k for k, v in collections.Counter(df['MolecularFormula'].tolist()).items() if v < 4}
            df = df[~df['MolecularFormula'].isin(molecular_formula_dropped)]
            # Build the molecular formula mapping
            formula_mapping = df.groupby('MolecularFormula')['SMILES'].apply(list).to_dict()
            # Build the wrong SMILES (different from the correct one but with the same molecular formula and also canonicalized)
            df_smiles = df['SMILES'].tolist()
            df_formulas = df['MolecularFormula'].tolist()
            random.seed(42)
            df_smiles_wrong = [random.sample([sw for sw in formula_mapping[f] if sw != s], k=3) for s, f in zip(df_smiles, df_formulas)]
            df['SMILES_wrong1'] = [sw[0] for sw in df_smiles_wrong]
            df['SMILES_wrong2'] = [sw[1] for sw in df_smiles_wrong]
            df['SMILES_wrong3'] = [sw[2] for sw in df_smiles_wrong]
            # Create shuffled with the new values (wrong SMILES but canonicalized & with the same molecular formula)
            shuffled = [np.random.permutation(row).tolist() for row in df[['SMILES_wrong1', 'SMILES_wrong2', 'SMILES_wrong3', "SMILES"]].values]
        else:
            raise ValueError(f"task_variant {self.task_variant} not recognized")

        train_dict = {
            'problem': df['SMILES_variant1'].tolist(),
            'solution': df['SMILES'].tolist(),
            'options': shuffled
        }
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

    def accuracy_reward(self, completions, solution, options, **kwargs):
        """Reward function - check that completion is same as ground truth."""

        answers = [self.preprocess_response(c) for c in completions]
        rewards = []

        letters = "ABCD"
        for i, (ans, sol) in enumerate(zip(answers, solution)):
            try:
                idx = letters.index(ans)
                select = options[i][idx]
                if select == sol:
                    rewards.append(1)
                    self.log_correct(ans)
                else:
                    rewards.append(0)
            except:
                rewards.append(0)
        return rewards

    def generate_prompt(self, problem, tokenizer, **kwargs):
        """Generate prompt for the MCQA task."""
        options = kwargs.get("options", [])
        r1_prefix = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.question_template.format(problem, *options)},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "problem": problem,
            "options": options
        }

    def dataset_preprocess(self, tokenizer):
        self.dataset["train"] = self.dataset["train"].shuffle(seed=42).select(range(min(50000, len(self.dataset["train"]))))
        self.dataset["test"] = self.dataset["test"].shuffle(seed=42).select(range(min(10000, len(self.dataset["test"]))))

        self.dataset = self.dataset.map(lambda x: self.generate_prompt(x["problem"], tokenizer, options=x["options"]))
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
