import random
from ..base import RLTask
from typing import Dict, Optional
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem
import pandas as pd
from Levenshtein import ratio as levenshtein_ratio

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs

class PermuteSmiles(RLTask):
    question_template: str = ""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "You are a student in Cheminformatics, who is very familiar with Simplified Molecular Input Line Entry System (SMILES) notation, and here's an exercise for you. Please permute the SMILES sequence for this molecule, in such a way that the SMILES sequence is different from the original one, but the original molecule does not change. Here is the original SMILES: [START_SMILES] {} [END_SMILES]. "
            "It is preferred that the resulted SMILES is different from the input SMILES as much as possible. "
            "For example, CC(=O)O can be permuted into O=C(O)C. "
	    "A reasoning pattern that you could follow is to visualize the molecule in your mind, describe it in details, and then find another starting atom for the SMILES sequence. "
            "Don't hesitate to start over if you get stuck. You can reasoning as much as you want. "
            # "A strategy you could try, but not obligatory to do, is to reverse the order of the atoms. "
            "Your reponse must strictly follow the format: <think> [REASONING] </think> <answer> [START_SMILES] [SMILES] [END_SMILES] </answer>.\n"
            "Show your reasoning step-by-step in <think> </think> tags. And return the final answer in <answer> </answer> tags as a single SMILES sequence, for example <answer> [START_SMILES] c1ccccc1 [END_SMILES] </answer>. "
            "Do not write anything else outside of the tags.\n"
            "Your response: <think> "
        )
    
    def load(self):
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        train_dict = {
            'problem': df['SMILES'].tolist(),
            'solution': df['SMILES'].tolist()
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
    
    def random_print(self, answer, ref, completion, out_rate = 0.01):
        if random.random() < out_rate:  # 1% chance to print a completion
            out = (
                "\n\n=======<RANDOM_RESPONSE>=======\n"
                f"*** ANSWER: {answer}\n"
                f"*** REFERENCE: {ref}\n"
                f"*** FULL RESPONSE: {completion}"
            )
            print(out)
    
    def good_print(self, answer, ref, completion, out_rate = 0.1):
        if random.random() < out_rate:  # 10% chance to print a completion
            # print(f"\n\n=======<RANDOM_RESPONSE>=======\n{completion}")
            out = (
                "\n\n=======<GOOD_RESPONSE>=======\n"
                f"*** ANSWER: {answer}\n"
                f"*** REFERENCE: {ref}\n"
                f"*** FULL RESPONSE: {completion}"
            )
            print(out)
    
    def accuracy_reward(self, completions, solution, **kwargs):
        """
        Reward function - check that completed SMILES refers to the same molecule as the original SMILES.
        Bonus if the output SMILES is different from the original SMILES.
        """

        # answers = [self.preprocess_response(c) for c in completions]
        # self.random_print(completions, solution, answers)
        
        def tanimoto_sim(mol1, mol2):
            mol1 = Chem.MolFromSmiles(mol1)
            mol2 = Chem.MolFromSmiles(mol2)
            
            fpgen = AllChem.GetRDKitFPGenerator(fpSize=1024)
            fp1 = fpgen.GetFingerprint(mol1)
            fp2 = fpgen.GetFingerprint(mol2)
            
            return DataStructs.FingerprintSimilarity(fp1, fp2)

        def calc_reward(mol1, mol2, beta=10):
            return (1-levenshtein_ratio(mol1, mol2))**(1+(1-tanimoto_sim(mol1, mol2))*beta)

        rewards = []

        for completion, ref in zip(completions, solution):
            answer = self.preprocess_response(completion)
            self.random_print(answer, ref, completion)
            
            if answer == "NONE":
                rewards.append(-1)
                continue
            
            response_mol = Chem.MolFromSmiles(answer)
            if response_mol is None:
                rewards.append(-1)
                continue
            
            reward = calc_reward(answer, ref)
            
            # if answer == ref:
            #     rewards.append(-0.6)
            #     continue
                        
            # canon_response = Chem.MolToSmiles(response_mol, canonical=True)
            # canon_reference = Chem.MolToSmiles(Chem.MolFromSmiles(ref), canonical=True)
            # if canon_response != canon_reference:
            #     rewards.append(-0.3)
            #     continue
            
            # self.good_print(answer, ref, completion)
              
            # edit_distance_reward = 1 - levenshtein_ratio(answer, ref) # range (0, 1)
            # # reward = 0.2 + 0.8*edit_distance_reward
            # reward = edit_distance_reward
            
            rewards.append(reward)
        
        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        if not response.startswith("<think>"):
            response = "<think>" + response
        pattern = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m and len(m.groups()) == 2:
            smi = m.groups()[1]

            # Maybe smiles contains [BEGIN_SMILES] and [END_SMILES]
            smi = smi.replace("[BEGIN_SMILES]", "")
            smi = smi.replace("[START_SMILES]", "")
            smi = smi.replace("[END_SMILES]", "")
            smi = smi.replace(' ', '')
            
            return smi
        else:
            return "NONE"
