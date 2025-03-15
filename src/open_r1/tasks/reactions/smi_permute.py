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
    system_prompt: str = ""
    custom_metrics: dict = {}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "Question: You are an expert in Cheminformatics, who is very familiar with Simplified Molecular Input Line Entry System (SMILES) notation, and here's an exercise for you. Please permute the given SMILES sequence of a molecule, in such a way that the resulted SMILES is different from the input SMILES as much as possible, but the original molecule is not changed. "
            # "It is preferred that the resulted SMILES is different from the input SMILES as much as possible. "
            # "Here is an example to help you understand the task:\n"
            # "Given an example input SMILES [START_SMILES] CCCC(C)C1CC1C [END_SMILES]. "
            # "<think> First, I visualize the molecule by numbering the atoms: [START_SMILES] [C:1][C:2][C:3][C:4]([C:5])[C:6]1[C:7][C:8]1[C:9] [END_SMILES]. "
            # "This structure is a cyclopropane ([START_SMILES] [C:6]1[C:7][C:8]1 [END_SMILES]) with a 1-methylbutyl ([START_SMILES] [C:1][C:2][C:3][C:4]([C:5])- [END_SMILES]) and a methyl ([START_SMILES] -[C:9] [END_SMILES]) substitutions. "
            # "I could try to permute this SMILES by starting from another atom, for example, the methyl group [START_SMILES] -[C:9] [END_SMILES]. "
            # "In that case, a possible permutation could be [START_SMILES] [C:9][C:8]1[C:7][C:6]1[C:4]([C:5])[C:3][C:2][C:1] [END_SMILES]. "
            # "Removing the atom numbering, this would give [START_SMILES] CC1CC1C(C)CCC [END_SMILES]. "
            # "This SMILES represents the same molecule as the input SMILES, with a cyclopropane substituted by a methyl group and a 1-methylbutyl group. "
            # "Therefore, I submit [START_SMILES] CC1CC1C(C)CCC [END_SMILES] as the final answer </think>. "
            # "<answer> [START_SMILES] CC1CC1C(C)CCC [END_SMILES] </answer>.\n"
            # "For example, [START_SMILES] CC(=O)O [END_SMILES] can be permuted into [START_SMILES] O=C(O)C [END_SMILES]. "
            # "Show your reasoning in <think> </think> tags. And return the final answer in <answer> </answer> tags as a single SMILES sequence, for example <answer> [START_SMILES] O=C(O)C [END_SMILES] </answer>. "
	        # "A reasoning pattern that you could follow is to visualize the molecule in your mind, describe it in details, and then find another starting atom for the SMILES sequence. "
            # "Don't hesitate to start over again if you get stuck or get the molecule wrong. "
            "Remember that your answer SMILES must satisfy two criteria: 1) it must be different from the input SMILES, and 2) it must represent the same molecule. "
            # "You can revise your reasoning as many times as you want. "
            # "A strategy you could try, but not obligatory to do, is to reverse the order of the atoms. "
            # "Your reponse must strictly follow the format: <think> [REASONING] </think> <answer> [START_SMILES] [SMILES] [END_SMILES] </answer>.\n"
            # "Do not write anything else outside of the tags.\n"
            "Here is the SMILES that you need to work on: [START_SMILES] {} [END_SMILES]. "
            "Your response: <think> "
        )
        self.question_template = self.question_template.replace("[START_SMILES] ", "")
        self.question_template = self.question_template.replace(" [END_SMILES]", "")
        
        self.custom_metrics = {
            'n_samples': 0,
            'n_waits': [],
            'reasoning_score': [],
            'answer_scores': [],
        }
        
    def generate_prompt(self, problem, tokenizer, **kwargs):
        return {
            'prompt': self.question_template.format(problem),
            'problem': problem
        }
    
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
    
    def random_print(self, print_data, out_rate = 0.01):
        if random.random() < out_rate:  # 1% chance to print a completion
            out = (
                "\n\n=======<RANDOM_RESPONSE>=======\n"
                # f"*** ANSWER: {answer}\n"
                # f"*** REFERENCE: {ref}\n"
                # f"*** FULL RESPONSE: {completion}"
            )
            for k, v in print_data.items():
                out += f"*** {k.upper()}: {v}\n"
            print(out)
    
    def good_print(self, print_data, out_rate = 0.1):
        if random.random() < out_rate:  # 10% chance to print a completion
            # print(f"\n\n=======<RANDOM_RESPONSE>=======\n{completion}")
            out = (
                "\n\n=======<GOOD_RESPONSE>=======\n"
                # f"*** ANSWER: {answer}\n"
                # f"*** REFERENCE: {ref}\n"
                # f"*** FULL RESPONSE: {completion}"
            )
            for k, v in print_data.items():
                out += f"*** {k.upper()}: {v}\n"
                
            print(out)
    
    def accuracy_reward(self, completions: list[str], solution, **kwargs):
        """
        Reward function - check that completed SMILES refers to the same molecule as the original SMILES.
        Bonus if the output SMILES is different from the original SMILES.
        """

        # answers = [self.preprocess_response(c) for c in completions]
        # self.random_print(completions, solution, answers)
        
        def _tanimoto_sim(mol1, mol2):
            mol1 = Chem.MolFromSmiles(mol1)
            mol2 = Chem.MolFromSmiles(mol2)
            
            fpgen = AllChem.GetRDKitFPGenerator(fpSize=1024)
            fp1 = fpgen.GetFingerprint(mol1)
            fp2 = fpgen.GetFingerprint(mol2)
            
            charge1 = Chem.GetFormalCharge(mol1)
            charge2 = Chem.GetFormalCharge(mol2)
            
            charge_penalty = 0.85 if charge1 != charge2 else 1.0
            
            return DataStructs.FingerprintSimilarity(fp1, fp2) * charge_penalty
        
        def _edit_distance_preprocess(smiles: str):
            # Remove stereo symbols
            smiles = re.sub(r'\/|\\|@', '', smiles)
            smiles = re.sub(r'\[CH\d?\]', 'C', smiles)
            smiles = re.sub(r'\[(Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)(?:H\d?)?(?:\:\d)?\]', lambda m: m.group(1), smiles)
            # Remove ring numbers
            smiles = re.sub(r'\d', '', smiles)
            # Remove redundant symbols
            smiles = re.sub(r'-', '', smiles)
            
            return smiles
        
        def _edit_distance(mol1, mol2):
            mol1 = _edit_distance_preprocess(mol1)
            mol2 = _edit_distance_preprocess(mol2)
            if mol1 in mol2 or mol2 in mol1:
                return 0.0
            return 1-levenshtein_ratio(mol1, mol2)

        def _calc_score(mol1: str, mol2: str, beta=30):
            if Chem.MolFromSmiles(mol1) is None or Chem.MolFromSmiles(mol2) is None:
                return 0.0
            edit_distance = _edit_distance(mol1, mol2)
            edit_distance = min(edit_distance, 0.3)
            return edit_distance**(1+(1-_tanimoto_sim(mol1, mol2))*beta) / 0.3
        
        def _extract_smiles(completion: str):
            def _post_process_smiles(smiles):
                smiles = re.sub(r'(?<=[A-Za-z]|\)|\])-(?=[A-Za-z]|\(|\[)', '', smiles)
                smiles = re.sub(r'\[CH\d?\]', 'C', smiles)
                smiles = re.sub(r'\[(?:Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)\]', lambda m: m.group(0).strip("[]"), smiles)
                smiles = smiles.split('.')[0]
                return smiles
            excluded_smiles = set(('I'))
            words = completion.split()
            words = [w.strip(' !"#$%&\'*+,-./:;<=>?@\\^_`{|}~') for w in words]
            # words_tkns = [smiles_tokenizer(w) for w in words]
            # smiles = [w for w, w_tokens in zip(words, words_tkns) if w_tokens.replace(' ', '') == w]
            smiles = words
            smiles = [s for s in smiles if s and s not in excluded_smiles]
            smiles = [_post_process_smiles(s) for s in smiles]
            smiles = [s for s in smiles if Chem.MolFromSmiles(s)]
            return smiles
        
        def _extract_smiles_from_answer(answer: str, input_smi: str):
            '''Extract the longest SMILES from the answer that is different from the input SMILES'''
            smiles = _extract_smiles(answer)
            smiles = [s for s in smiles if s != input_smi]
            smiles = max(smiles, key=len) if smiles else None
            return smiles
        
        def count_waits(completion: str):
            return completion.lower().count("wait")

        rewards = []

        for completion, ref in zip(completions, solution):
            reasoning = completion.rsplit('<answer>', maxsplit=1)[0]
            smiles = _extract_smiles(reasoning)
            scores = [_calc_score(smi, ref) for smi in smiles]
            reasoning_score = max(scores) if scores else -0.5
            
            answer = self.preprocess_response(completion)
            answer_smiles = _extract_smiles_from_answer(answer, ref)
            answer_score = _calc_score(answer_smiles, ref) if answer_smiles else 0
            
            reward = reasoning_score + answer_score
            
            report = {'answer': answer, 
                      'reference': ref, 
                      'reasoning_score [0, 1]': reasoning_score,
                      'answer_score [0, 1]': answer_score, 
                      'accuracy_reward [0, 2]': reward, 
                      'full_completion': completion}
            
            self.random_print(report)
            if reward > 0.3:
                self.good_print(report)
            
            rewards.append(reward)
            
            self.custom_metrics['n_samples'] += 1
            self.custom_metrics['n_waits'].append(count_waits(completion))
            self.custom_metrics['reasoning_score'].append(reasoning_score)
            self.custom_metrics['answer_scores'].append(answer_score)
        
        return rewards

    def reasoning_length_reward(self, completions, **kwargs):
        max_length = 2000
        rewards = []
        for completion in completions:
            if not completion.startswith("<think>"):
                completion = "<think>" + completion
            reasoning = re.search(r"<think>(.*?)<\/think>", completion, re.DOTALL)
            if reasoning is None:
                rewards.append(0.0)
                continue
            reasoning = reasoning.group(1)
            reasoning_length = len(reasoning.strip().split())
            reasoning_reward = min(1, reasoning_length / max_length)
            rewards.append(reasoning_reward)
        return rewards
    
    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        if not response.startswith("<think>"):
            response = "<think>" + response
        pattern = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m and len(m.groups()) == 2:
            # smi = m.groups()[1]

            # smi = self._post_process_smiles(smi)
            
            # return smi
            return m.groups()[1]
        else:
            return "NONE"

    
    # def _post_process_smiles(self, smiles):
    #     # Maybe smiles contains [BEGIN_SMILES] and [END_SMILES]
    #     smiles = smiles.replace("[BEGIN_SMILES]", "")
    #     smiles = smiles.replace("[START_SMILES]", "")
    #     smiles = smiles.replace("[END_SMILES]", "")
    #     smiles = smiles.replace(' ', '')
    #     smiles = smiles.strip(' !"#$%&\'*+,-./:;<=>?@\\^_`{|}~')
    #     return smiles
        
    def format_reward(self, completions, **kwargs):
        """
        Format: <think>...</think><answer>...</answer>
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
        
        Returns:
            list[float]: Reward scores
        """
        rewards = []

        for completion in completions:
            try:
                if not completion.startswith("<think>"):
                    completion = "<think>" + completion
                regex = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
                match = re.search(regex, completion, re.DOTALL) 
                # if the format is not correct, reward is 0
                if match is None or len(match.groups()) != 2:
                    rewards.append(0.0)
                else:
                    # The model tends to generate gibberish outside of the tags
                    reward = len(match.group()) / len(completion)
                    # smi = match.group(2)
                    # smi = self._post_process_smiles(smi)
                    # if Chem.MolFromSmiles(smi):
                    #     reward += 0.5
                    rewards.append(reward)
            except Exception:
                rewards.append(0.0)
        return rewards
    
    def get_metrics(self):
        metrics = {}
        if self.custom_metrics['n_samples'] > 0:
            metrics['n_samples'] = self.custom_metrics['n_samples']
            for k, v in self.custom_metrics.items():
                if k != 'n_samples':
                    metrics[k] = sum(v) / len(v)
                    self.custom_metrics[k] = []
        
        return metrics
        