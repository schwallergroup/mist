
from ..base import RLTask
from typing import Dict
import re
import os
from datasets import Dataset, DatasetDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from random import random
import difflib

class Smiles2Name(RLTask):
    question_template: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_template = ("""
            "What is the name of this reaction? {}. "
            "You have the choice between the following reaction names : ['Formation of Sulfonic Esters',
            'Reduction of nitro groups to amines',
            'Hydrogenation (double to single)', 'Bromination',
            'Aminolysis of esters', 'Reduction of tertiary amides to amines',
            'Ester saponification (alkyl deprotection)',
            'Halide to nitrile conversion',
            'Carboxylic acid with primary amine to amide',
            'Esterification of Carboxylic Acids', 'Decarboxylation',
            'C-methylation', 'Friedel-Crafts acylation', 'S-methylation',
            'Aromatic nitration with HNO3', 'Methylation of OH with DMS',
            'N-alkylation of secondary amines with alkyl halides',
            'Williamson Ether Synthesis',
            'N-alkylation of primary amines with alkyl halides',
            'Hydrolysis or Hydrogenolysis of Carboxylic Esters or Thioesters',
            'Hydroxyl benzyl deprotection', 'Dehalogenation',
            'Reduction of nitrile to amine',
            'Reduction of ketone to secondary alcohol', 'Aldol condensation',
            'Wohl-Ziegler bromination benzyl primary',
            'Oxidation or Dehydrogenation of Alcohols to Aldehydes and Ketones',
            'Urea synthesis via isocyanate and primary amine',
            'Ullmann-Goldberg Substitution thiol', 'Aromatic dehalogenation',
            'Ullmann-Goldberg Substitution amine',
            'Displacement of ethoxy group by primary amine',
            'Sulfonamide synthesis (Schotten-Baumann) primary amine',
            'N-methylation', 'Chlorination', 'S-alkylation of thiols ',
            'Cleavage of methoxy ethers to alcohols', 'thia-Michael addition',
            'Sulfonamide synthesis (Schotten-Baumann) secondary amine',
            'Schotten-Baumann to ester',
            'Acylation of Nitrogen Nucleophiles by Carboxylic Acids',
            'Ester saponification (methyl deprotection)',
            'O-alkylation of carboxylic acids with diazo compounds',
            'Hydrogenation (triple to double)', 'Negishi',
            'Clemmensen ketone reduction',
            'Eschweiler-Clarke Secondary Amine Methylation',
            'oxa-Michael addition', 'aza-Michael addition primary',
            'Reduction of carboxylic acid to primary alcohol',
            'Urea synthesis via isocyanate and secondary amine',
            'Alcohol to ether', 'Ether cleavage to primary alcohol',
            'thiourea', 'O-methylation', 'Reductive amination with ketone',
            'Hydrogenolysis of tertiary amines',
            'Azide to amine reduction (Staudinger)', 'Hurtley reaction',
            'Arene hydrogenation', 'Wohl-Ziegler bromination allyl secondary',
            'Aromatic nitration with NO3 salt', 'Mitsunobu esterification',
            'Friedel-Crafts alkylation with halide', 'thiazole',
            'Goldberg coupling aryl amine-aryl chloride',
            'Azide-nitrile click cycloaddition to tetrazole',
            'Dehydration of amides to nitriles',
            'Boc amine protection of primary amine', 'Mitsunobu aryl ether',
            'Wittig reaction with triphenylphosphorane',
            'Reduction of secondary amides to amines',
            'Friedel-Crafts alkylation',
            'Williamson Ether Synthesis (intra to epoxy)',
            'Non-aromatic nitration with HNO3', 'Transesterification',
            'Reduction of ester to primary alcohol', 'Wittig with Phosphonium',
            'Wohl-Ziegler bromination benzyl secondary',
            'Reductive amination with aldehyde',
            'Deprotection of carboxylic acid',
            'Grignard from ketone to alcohol',
            'Cleavage of alkoxy ethers to alcohols', 'reductive amination',
            'Alcohol protection with silyl ethers', 'Alkylation of amines',
            'Formation of Azides from halogens', 'Goldberg coupling',
            'Iodination', 'Ketonization by decarboxylation of carbonic acids',
            'Reduction of aldehydes and ketones to alcohols',
            'Alcohol to bromide with HBr',
            'Ester with secondary amine to amide',
            'Eschweiler-Clarke Primary Amine Methylation',
            'Oxidation of alcohol and aldehyde to ester',
            'Acetal hydrolysis to aldehyde', 'Clemmensen aldehyde reduction',
            'Reductive amination with alcohol',
            'aza-Michael addition secondary',
            'Grignard from aldehyde to alcohol', 'Ketal hydrolysis to ketone',
            'Reduction of primary amides to amines', 'Wittig', 'Diels-Alder',
            'Oxidation of amide to carboxylic acid',
            'Reductive methylation of primary amine with formaldehyde',
            'DMS Amine methylation', 'Michael addition', 'Pyrazole formation',
            'Oxidation of ketone to carboxylic acid', 'Methylation with DMS',
            'Boc amine protection of secondary amine',
            'Ketonization by decarboxylation of acid halides',
            'Benzoxazole formation from acyl halide',
            'Urea synthesis via isocyanate and diazo',
            'Nef reaction (nitro to ketone)',
            'Benzoxazole formation (intramolecular)',
            'Phenol with formaldehyde (ortho)',
            'Benzothiazole formation from aldehyde',
            'Displacement of ethoxy group by secondary amine',
            'Alcohol to iodide', 'Wohl-Ziegler bromination allyl primary',
            'Paal-Knorr pyrrole synthesis',
            'Goldberg coupling aryl amide-aryl chloride',
            'Benzoxazole formation from aldehyde',
            'Reaction of alkyl halides with organometallic coumpounds',
            'Henry Reaction', 'Alcohol deprotection from silyl ethers',
            'Pictet-Spengler', 'Alcohol to azide', 'Knoevenagel Condensation',
            'Ester and halide to ketone',
            'Mitsunobu aryl ether (intramolecular)',
            'Acylation of olefines by aldehydes',
            'Intramolecular amination of azidobiphenyls (heterocycle formation)',
            'Michael addition methyl', 'Boc amine protection (ethyl Boc)',
            'Oxidative esterification of primary alcohols',
            'Wohl-Ziegler bromination carbonyl secondary', 'urea',
            'Alcohol to bromide with NBS', 'beta C(sp3) arylation',
            'Directed ortho metalation of arenes', 'Boc amine deprotection',
            'Acetal hydrolysis to diol', 'Alkyl bromides from alcohols',
            'Hydrazone oxidation to diazoalkane',
            'Oxidation of alcohol to carboxylic acid',
            'Ester with ammonia to amide',
            'S-alkylation of thiols with alcohols',
            'Oxidation of alkene to carboxylic acid',
            'Carboxyl benzyl deprotection', 'Primary amine to chloride',
            'Protection of carboxylic acid', 'Methylation',
            'Azide-nitrile click cycloaddition to triazole',
            'Boc amine protection with Boc anhydride', 'P-cleavage',
            'Ketone from Li and CO2',
            'Reductive carboxylation of aryl bromides', 'Heck terminal vinyl',
            'Wohl-Ziegler bromination carbonyl tertiary', 'Minisci (ortho)',
            'Minisci (para)', 'Ullmann condensation',
            'Acyl chloride with ammonia to amide', 'Benzimidazole aldehyde',
            'Cleavage of sulfons and sulfoxides',
            'Olefination of ketones with Grignard reagents',
            'Formation of NOS Heterocycles',
            'Wohl-Ziegler bromination benzyl tertiary',
            'Wohl-Ziegler bromination carbonyl primary',
            'Huisgen alkyne-azide 1,3 dipolar cycloaddition',
            'Kumada cross-coupling', 'Aromatic chlorination',
            'Alkyl chlorides from alcohols',
            'Oxidation of aldehydes to carboxylic acids',
            'Wohl-Ziegler bromination allyl tertiary',
            'Phenol with formaldehyde (para)',
            'O-alkylation of amides with diazo compounds',
            'Preparation of boronic acids',
            'Suzuki coupling with boronic acids',
            'Aromatic nitration with NO2 salt',
            'S-alkylation of thiols (ethyl)',
            'Suzuki coupling with boronic acids OTf',
            'Aromatic nitration with alkyl NO2', 'Ketone from Weinreb amide',
            'Deselenization', 'spiro-chromanone',
            'Heck reaction with vinyl ester and amine',
            'Suzuki coupling with boronic esters', 'Methylation with DMC',
            'Asymmetric ketones from N,N-dimethylamides', 'Stille', 'Suzuki',
            'Suzuki coupling with sulfonic esters',
            'Alkene oxidation to aldehyde',
            'Formation of Sulfonic Esters on TMS protected alcohol',
            'Diels-Alder (ON bond)',
            'Nucleophilic substitution OH - alkyl silane', 'Quinone formation',
            'Benzothiazole formation from acyl halide', 'Fluorination',
            'Preparation of boronic ethers',
            'Boc amine deprotection to NH-NH2',
            'Boc amine deprotection of guanidine',
            'Alcohol deprotection from silyl ethers (diol)',
            'Alkyl iodides from alcohols',
            'Grignard with CO2 to carboxylic acid', 'Julia Olefination',
            'Preparation of boronic ethers with bis(pinacolato)diboron',
            'Petasis reaction with amines aldehydes and boronic acids',
            'Petasis reaction with amines and boronic acids', 'Ugi reaction',
            'Carboxylic acid from Li and CO2',
            'Suzuki coupling with boronic esters OTf',
            'Acyl chlorides from alcohols', 'Negishi coupling',
            'Aromatic substitution of bromine by chlorine',
            'Huisgen 1,3,4-oxadiazoles from COOH and tetrazole',
            'Aromatic bromination', 'Chan-Lam amine', 'A3 coupling',
            'Boc amine protection explicit',
            'Sulfamoylarylamides from carboxylic acids and amines',
            'Oxirane functionalization with alkyl iodide',
            'Petasis reaction with amines and boronic esters',
            'Schmidt aldehyde amide',
            'Alcohol deprotection from silyl ethers (double)',
            'Oxidation of boronic esters', 'Oxidative Heck reaction',
            'Deboronation of boronic esters',
            'Carbonylation with aryl formates', 'Oxidation of boronic acids]"
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags in SMILES notation, for example <answer> ... </answer>. Think step by step inside <think> tags."
            """
        )
        # Dataset here: /iopsstor/store/cscs/swissai/a05/chem/CRLLM-PubChem-compounds1M.csv

    def load(self) -> DatasetDict:
        """Load and return the complete dataset."""
        df = pd.read_csv(self.dataset_id_or_path)
        train_dict = {
            'problem': df['REACTION'].tolist(),
            'solution': df['NAME'].tolist()
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
        """Updated reward function awarding full reward (+1) to close matches."""
        rewards = []
        for content, sol in zip(completions, solution):
            ans = self.preprocess_response(content)

            norm_ans = ans.strip().lower()
            norm_sol = sol.strip().lower()

            if norm_ans in ("none", ""):
                rewards.append(-1)
                continue

            similarity = difflib.SequenceMatcher(None, norm_ans, norm_sol).ratio()

            if similarity > 0.9:  
                rewards.append(1)
                self.log_correct(content)
            elif similarity > 0.8:
                rewards.append(0.2)
            else:
                rewards.append(-0.5)
        return rewards


    def preprocess_response(self, response):
        """Preprocess the response before checking for accuracy."""
        pattern = r"<answer>(.*)<\/answer>"
        m = re.search(pattern, response, re.DOTALL)
        if m:
            smi = m.groups()[0]

            # Maybe smiles contains [BEGIN_SMILES] and [END_SMILES]
            if "[BEGIN_SMILES]" in smi:
                smi = smi.replace("[BEGIN_SMILES]", "")
            if "[END_SMILES]" in smi:
                smi = smi.replace("[END_SMILES]", "")

            return smi
        else:
            return "NONE"





