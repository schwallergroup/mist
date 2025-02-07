
from .base import CustomTask
from typing import Dict
import re
from rdkit import Chem


class ForwardReaction(CustomTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = "What is the product of the following reaction - [BEGIN_SMILES] {} [END_SMILES]?"

    def process_line(self, line: str) -> str:
        """Process a line from the source file."""
        rm_space = re.sub(" +", "", line)
        return rm_space

    def read_files(self, src_file: str, tgt_file: str) -> Dict:
        """Read source and target files and create dataset dictionary."""
        with open(src_file, 'r', encoding='utf-8') as f:
            problems = [self.question_template.format(self.process_line(line)) for line in f.readlines()]
            
        with open(tgt_file, 'r', encoding='utf-8') as f:
            solutions = ["[BEGIN_SMILES] "+self.process_line(line)+" [END_SMILES]" for line in f.readlines()]
            
        # Create empty messages list for each example
        messages = [[] for _ in range(len(problems))]
        
        return {
            'problem': problems,
            'solution': solutions,
            'messages': messages
        }

    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - chack that completion is same as ground truth."""

        contents = [completion[0]["content"] for completion in completions]
        answers = [self.preprocess_response(c) for c in contents]

        rewards = []
        for content, sol in zip(answers, solution):
            if content == "NONE":
                rewards.append(-1)
                continue
            try:
                gold_mol = Chem.MolToSmiles(Chem.MolFromSmiles(sol))
            except:
                # invalid target smiles
                rewards.append(-1)
                continue
            try:
                completion_mol = Chem.MolToSmiles(Chem.MolFromSmiles(content))
            except:
                # invalid generated smiles
                rewards.append(-1) # penalize if invalid smiles
                continue
            if gold_mol == completion_mol:
                rewards.append(1)  # reward if correct
            else:
                rewards.append(-0.5) # no reward if incorrect
        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        pattern = r"<answer>.*?</answer>"
        if re.match(pattern, response):
            smi = response.split("<answer>")[1].split("</answer>")[0]

            # Maybe smiles contains [BEGIN_SMILES] and [END_SMILES]
            if "[BEGIN_SMILES]" in smi:
                smi = smi.replace("[BEGIN_SMILES]", "")
            if "[END_SMILES]" in smi:
                smi = smi.replace("[END_SMILES]", "")
            print(smi)
            return smi
        else:
            return "NONE"
