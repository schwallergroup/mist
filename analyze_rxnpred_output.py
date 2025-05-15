import os
import re
import pandas as pd
from rdkit import Chem

from collections import defaultdict

from tqdm import tqdm
from open_r1.tasks import ForwardReaction
from transformers import AutoTokenizer

INPUT_TXT = 'eval_results_500/rxnpred_ShaiV6/no_thinking/output.at1.txt'
PER_SAMPLE_EVAL_CSV = 'per_sample_results.at1.csv'
RESULT_JSON = 'eval_results.at1.txt'

task = ForwardReaction(dataset_id_or_path='/data/share/USPTO_480k_clean_no_sft/', task_mode='tagged')
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

def split_responses(content: str):
    patterns = re.compile(r"\[\[EXAMPLE \d+\]\].*?\[Extracted SMILES\]:.*?\n", re.DOTALL)
    matches = patterns.findall(content)
    return matches

def extract_smiles(text, prompt="", relax=False):
    # match = re.search(r'<answer>\s*\[START_SMILES\]\s*(.*?)\s*\[END_SMILES\]\s*</answer>', text)
    # return match.group(1).strip() if match else None
    res = task.extract_smiles_from_answer(task.preprocess_response(text), prompt)
    if res is None and relax: # if relax is True, extract the longest SMILES in the reponse that is different from those in the prompt
        res = task.extract_smiles_from_answer(text, prompt)
    return res
    

def extract_info_response(response: str):
    example_num = re.search(r"\[\[EXAMPLE (\d+)\]\]", response).group(1).strip()
    prompt = re.search(r"(?<=\[Prompt\]: ).*?\n(?=\[Reference SMILES\])", response, re.DOTALL).group(0).strip()
    reaction = extract_smiles(prompt, relax=True)
    reference_smiles = re.search(r"(?<=\[Reference SMILES\]: ).*?\n", response, re.DOTALL).group(0).strip()
    response_content = re.search(r"(?<=\[Predicted Output\]: ).*?\n(?=\[Extracted SMILES\])", response, re.DOTALL).group(0).strip()
    extracted_smiles_exact = extract_smiles(response_content, prompt)
    extracted_smiles_relax = extract_smiles(response_content, prompt, relax=True)
    
    return {"example_num": example_num, "prompt": prompt, "reaction": reaction, "reference_smiles": reference_smiles, "response_content": response_content, "extracted_smiles_exact": extracted_smiles_exact, "extract_smiles_relax": extracted_smiles_relax}

def eval(response: str):
    def same_molecule(smiles1, smiles2):
        if smiles1 is None or smiles2 is None:
            return False
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if not mol1 or not mol2:
            return False
        return Chem.MolToSmiles(mol1, canonical=True) == Chem.MolToSmiles(mol2, canonical=True)
    response_info = extract_info_response(response)
    ref_smiles = response_info["reference_smiles"]
    extracted_smiles_exact = response_info["extracted_smiles_exact"]
    extracted_smiles_relax = response_info["extract_smiles_relax"]
    
    correct = same_molecule(ref_smiles, extracted_smiles_exact)
    correct_relax = same_molecule(ref_smiles, extracted_smiles_relax)
    
    response_length = len(tokenizer.tokenize(response_info["response_content"]))
    
    return {"example_id": response_info["example_num"], 
            "reaction": response_info["reaction"],
            "reference_smiles": ref_smiles,
            "extracted_smiles_exact": extracted_smiles_exact,
            "extracted_smiles_relax": extracted_smiles_relax,
            "correct": int(correct), 
            "correct_relax": int(correct_relax), 
            "response_length": response_length}

def summary_eval(per_response_eval_csv: str):
    eval_data = pd.read_csv(per_response_eval_csv)
    per_example_correct = defaultdict(list)
    per_example_correct_relax = defaultdict(list)
    for i, each in eval_data.iterrows():
        each = each.to_dict()
        per_example_correct[each["example_id"]].append(each["correct"])
        per_example_correct_relax[each["example_id"]].append(each["correct_relax"])
        
    for k, v in per_example_correct.items():
        per_example_correct[k] = int(sum(v) > 0)
    for k, v in per_example_correct_relax.items():
        per_example_correct_relax[k] = int(sum(v) > 0)
        
    return {"accuracy": sum(per_example_correct.values()) / len(per_example_correct), 
            "accuracy_relax": sum(per_example_correct_relax.values()) / len(per_example_correct_relax),
            "lengths": (eval_data["response_length"].mean(), eval_data["response_length"].std())}
    
    
def main():
    output_dir = os.path.dirname(INPUT_TXT)
    per_sample_output_csv = os.path.join(output_dir, PER_SAMPLE_EVAL_CSV)
    results_json = os.path.join(output_dir, RESULT_JSON)
    with open(INPUT_TXT, 'r') as f:
        content = f.read()
    
    responses = split_responses(content)
    res = []
    for r in tqdm(responses):
        try:
            result = eval(r)
            res.append(result)
        except Exception as e:
            # print(f"Error processing response: {e}")
            print(r)
            raise e
    
    res = pd.DataFrame(res)
    res.to_csv(per_sample_output_csv, index=False)
    
    final_res = summary_eval(per_sample_output_csv)
    with open(results_json, 'w') as f:
        f.write(f"Final Accuracy: {final_res['accuracy']:.4f}\n")
        f.write(f"Final Accuracy Relax: {final_res['accuracy_relax']:.4f}\n")
        f.write(f"Response Lengths: {final_res['lengths'][0]:.2f} ± {final_res['lengths'][1]:.2f}\n")
    
if __name__ == "__main__":
    main()