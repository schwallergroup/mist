
from ..base import RLTask
from typing import Any, Dict
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from random import random

class Iupac2Smiles(RLTask):
    question_template: str = ""
    custom_metrics: Dict[str, Any] = {}
    printed_sample_prompt: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = (
            "Question: You are an expert in Cheminformatics, who is very familiar with Simplified Molecular Input Line Entry System (SMILES) notation, and here's a task for you. "
            "Given a molecule with the IUPAC name as below, please provide the corresponding SMILES notation.\n"
            # "What is the SMILES for this molecule? {}. "
            "Here is the IUPAC name: {}.\n"
            # "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> CN1C=C... </answer>. Think step by step inside <think> tags."
            "Your response: <think> "
        )
        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv
        self.custom_metrics = {
            'n_samples': 0,
            'n_waits': [],
            'reasoning_score': [],
            'answer_scores': [],
        }
        
    def generate_prompt(self, problem, tokenizer, **kwargs):
        prompt = {
            'prompt': self.question_template.format(problem),
            'problem': problem
        }
        if not self.printed_sample_prompt: # print sample prompt once
            print(f"***SAMPLE PROMPT:\n{prompt['prompt']}")
            self.printed_sample_prompt = True
        
        return prompt
        
    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        df = df.drop_duplicates(subset=['SMILES'])
        train_dict = {
            'problem': df['IUPAC'].tolist(),
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

    def accuracy_reward(self, completions, solution, **kwargs):
        """Reward function - check that completion is same as ground truth."""
        def _tanimoto_sim(mol1, mol2):
            mol1 = Chem.MolFromSmiles(mol1)
            mol2 = Chem.MolFromSmiles(mol2)
            
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, useChirality=True)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, useChirality=True)
                        
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        
        def _extract_smiles(completion: str):
            def _post_process_smiles(smiles):
                smiles = re.sub(r'(?<=[A-Za-z]|\)|\])-(?=[A-Za-z]|\(|\[)', '', smiles)
                smiles = re.sub(r'\[CH\d?\]', 'C', smiles)
                smiles = re.sub(r'\[(?:Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)\]', lambda m: m.group(0).strip("[]"), smiles)
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
        
        def _extract_smiles_from_answer(answer: str):
            '''Extract the longest SMILES from the answer '''
            smiles = _extract_smiles(answer)
            smiles = max(smiles, key=len) if smiles else None
            return smiles
        
        def _calc_score(mol1: str, mol2: str, beta=10):
            if Chem.MolFromSmiles(mol1) is None or Chem.MolFromSmiles(mol2) is None:
                return 0.0
            sim = _tanimoto_sim(mol1, mol2)
            return sim ** beta
        
        rewards = []

        for completion, ref in zip(completions, solution):
            reasoning = completion.rsplit('<answer>', maxsplit=1)[0]
            reasoning_smiles = _extract_smiles(reasoning)
            scores = [_calc_score(smi, ref) for smi in reasoning_smiles]
            max_score = max(scores) if scores else -0.5
            best_smiles_reasoning = reasoning_smiles[scores.index(max_score)] if max_score in scores else 'None'
            reasoning_score = max_score
            if reasoning_score == 1.0:
                reasoning_score += 1.0 # massive bonus for truly correct reasoning
            
            answer = self.preprocess_response(completion)
            answer_smiles = _extract_smiles_from_answer(answer)
            answer_score = _calc_score(answer_smiles, ref) if answer_smiles else 0
            if answer_score == 1.0:
                answer_score += 1.0 # massive bonus for truly correct answer
            
            reward = reasoning_score + answer_score
            
            answer_smiles = answer_smiles if answer_smiles else 'None'
            report = {'reference': ref,
                      'answer': answer_smiles,
                      'best_smiles_in_reasoning': best_smiles_reasoning,
                      'reasoning_score [0, 2]': reasoning_score,
                      'answer_score [0, 2]': answer_score, 
                      'accuracy_reward [0, 4]': reward, 
                      'full_completion': completion}
            
            self.random_print(report)
            if reward > 0.3:
                self.good_print(report)
                
            rewards.append(reward)
            
            self.custom_metrics['n_samples'] += 1
            self.custom_metrics['n_waits'].append(self.count_waits(completion))
            self.custom_metrics['reasoning_score'].append(reasoning_score)
            self.custom_metrics['answer_scores'].append(answer_score)
            
        return rewards

    def tanimoto_accuracy_reward(self, completions, solution, **kwargs):
        """Reward function using Tanimoto similarity between prediction and ground truth."""
        answers = [self.preprocess_response(c) for c in completions]
        
        rewards = []
        for content, sol in zip(answers, solution):
            if content == "NONE":
                rewards.append(-1)
                continue
                
            try:
                # Convert ground truth SMILES to molecule and fingerprint
                gold_mol = Chem.MolFromSmiles(sol)
                if gold_mol is None:
                    rewards.append(-1)
                    continue
                gold_fp = AllChem.GetMorganFingerprintAsBitVect(gold_mol, 2)
                
                # Convert prediction SMILES to molecule and fingerprint
                pred_mol = Chem.MolFromSmiles(content)
                if pred_mol is None:
                    rewards.append(-1)
                    continue
                pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2)
                
                # Calculate Tanimoto similarity
                tanimoto = DataStructs.TanimotoSimilarity(gold_fp, pred_fp)
                
                # Scale the reward: 
                # 1.0 for perfect match
                # Proportional to similarity for partial matches
                # Still penalize very poor predictions
                if tanimoto == 1.0:
                    reward = 1.0
                    self.log_correct(content)
                elif tanimoto < 0.3:  # You can adjust this threshold
                    reward = -0.5
                else:
                    reward = tanimoto - 0.3  # Shifts the reward to be negative for very low similarities
                    
                rewards.append(reward)
                
            except Exception as e:
                rewards.append(-1)
                continue
                
        return rewards

    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        if not response.startswith("<think>"):
            response = "<think>" + response
        pattern = r"<think>(.*?)<\/think>\s*<answer>(.*?)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m and len(m.groups()) == 2:
            return m.groups()[1]
        else:
            return "NONE"
        
    def get_metrics(self):
        return super().get_metrics()