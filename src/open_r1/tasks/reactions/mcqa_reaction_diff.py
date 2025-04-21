
from ..base import RLTask
import numpy as np
import re
from random import random
from datasets import Dataset, DatasetDict
import pandas as pd

class SmilesReplacement(RLTask):
    question_template: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "You are an expert in Chemsitry and you understand chemical reactions very well. Which chemical reaction is correct?" 
            "Choose from the following options and make sure to think before answering. Options: \nA. {}\nB. {}\nC. {}\nD. {}\n"
            "Respond with the option letter inside <answer> </answer> tags. (A, B, C, or D)."
        )
        # Dataset here: /data/david/dataset_swapped500k.csv

    def load(self) -> DatasetDict:

        "loading & preping the dataset"
        
        df = pd.read_csv(self.dataset_id_or_path)
        shuffled = [np.random.permutation(row).tolist() for row in df[['true_reaction', 'fake1', 'fake2', 'fake3']].values]
        train_dict = {
            'solution': df['true_reaction'].tolist(),
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
    
    def generate_prompt(self, tokenizer, **kwargs):
        """Prompt for the MC task"""
        options = kwargs.get("options", [])
        r1_prefix = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.question_template.format(*options)},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "options": options
        }

    def dataset_preprocess(self, tokenizer):
        self.dataset["train"] = self.dataset["train"].shuffle(seed=42).select(range(min(50000, len(self.dataset["train"]))))
        self.dataset["test"] = self.dataset["test"].shuffle(seed=42).select(range(min(10000, len(self.dataset["test"]))))

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


    def accuracy_reward(self, completions, solution, options, **kwargs):
        """Reward +1 for correct choice, 0 otherwise, with periodic example logging."""
        rewards = []
        letters = "ABCD"

        for i, (completion, sol) in enumerate(zip(completions, solution)):
            ans = self.preprocess_response(completion).strip()
            reward = 0.0

            if ans in letters:
                idx = letters.index(ans)
                select = options[i][idx]
                if select == sol:
                    reward = 1.0
                    self.good_print({
                        "choice": ans,
                        "selected_text": select,
                        "gold": sol
                    })
                else:
                    self.random_print({
                        "choice": ans,
                        "selected_text": select,
                        "gold": sol
                    })

            rewards.append(reward)

        return rewards

"""
    def accuracy_reward(self, completions, solution, options, **kwargs):
       
        answers = [self.preprocess_response(c) for c in completions]
        rewards = []

        format_rewards = self.format_reward(completions, **kwargs) 
        reasoning_rewards = self.reasoning_steps_reward(completions, **kwargs)

        letters = "ABCD"
        for i, (ans, sol) in enumerate(zip(answers, solution)):
            try:
                idx = letters.index(ans)
                select = options[i][idx]

                accuracy = 1 if select == sol else 0

                final_reward = (0.6 * accuracy) + (0.2 * format_rewards[i]) + (0.2 * reasoning_rewards[i])
                rewards.append(final_reward)

                if accuracy == 1:
                    self.log_correct(ans)

            except:
                rewards.append(0)  

        return rewards

"""