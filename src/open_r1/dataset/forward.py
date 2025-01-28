
from .base import CustomDatasetLoader
from typing import Dict
import re


class ForwardReactionDataset(CustomDatasetLoader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = "What is the product of the following reaction - [BEGIN_SMILES] {} [END_SMILES]? Reason step-by-step."

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
