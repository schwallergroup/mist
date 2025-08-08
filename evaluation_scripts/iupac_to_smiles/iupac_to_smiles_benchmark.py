import os
import argparse

from vllm import LLM, SamplingParams
from rdkit import Chem
import re
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="IUPAC to SMILES Benchmarking Script")
    parser.add_argument("--model", type=str, required=True, help="Path to the vLLM model checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file (pickle format).")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save the output results.")
    return parser.parse_args()

def _evaluate(prompts: list[str], test_data, llm, sampling_params, outpath: str, reasoning=True):
    def extract_smiles(text):
        if not reasoning:
            return extract_normal_smiles(text)
        match = re.search(r'<answer>.*\[START_SMILES\]\s*(.*?)\s*\[END_SMILES\].*</answer>', text)
        return match.group(1).strip() if match else None

    def extract_normal_smiles(text):
        match = re.search(r'\[START_SMILES\](.*?)\[END_SMILES\]', text)
        return match.group(1).strip() if match else None


    def same_molecule(smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if not mol1 or not mol2:
            return False
        return Chem.MolToSmiles(mol1, canonical=True) == Chem.MolToSmiles(mol2, canonical=True)
    
    def print_to_file(content, mode="a"):
        file_path=outpath
        with open(file_path, mode) as f:
            f.write(content + "\n")
        # print(content)
    
    if os.path.exists(outpath):
        return
    
    correct = 0
    total = len(prompts)
        
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate each example
    for idx, output in enumerate(outputs):
        for j in range(1):
            predicted = output.outputs[j].text
            predicted_smiles = extract_smiles(predicted)
            reference_smiles = extract_normal_smiles(test_data[idx][0]['answer'])

            print_to_file(f"Example {idx+1}:")
            print_to_file(f"Prompt: {prompts[idx]}")
            print_to_file(f"Reference SMILES: {reference_smiles}")
            print_to_file(f"Predicted Output: {predicted}")
            print_to_file(f"Extracted SMILES: {predicted_smiles}")

            if predicted_smiles is None:
                print_to_file("=" * 60)
                continue
            elif ';' not in predicted_smiles:
                if predicted_smiles and same_molecule(reference_smiles, predicted_smiles):
                    print_to_file("✔️ Match")
                    correct += 1
                    break
                else:
                    print_to_file("❌ No Match")
            else:
                split_smile = predicted_smiles.split(";")
                flag=0
                if ";" not in reference_smiles:
                    continue
                if len(split_smile)!=len(reference_smiles.split(";")):
                    continue
                for smile_num in range(len(split_smile)):
                    if split_smile[smile_num] and same_molecule(reference_smiles.split(";")[smile_num], split_smile[smile_num]):
                        print_to_file("✔️ Match")
                    else:
                        print_to_file("❌ No Match")
                        flag=1
                        break
                if flag==0:
                    correct+=1
                    break

            print_to_file("=" * 60)

    accuracy = 100.0 * correct / total
    print_to_file(f"Final Accuracy: {accuracy:.2f}% ({correct}/{total})")

def main(args):
    # Sample data list
    # data = pickle.load(open("/data/shai/benchmark_i2s.pkl", "rb"))
    data = pickle.load(open(args.data_path, "rb"))

    # Load vLLM model
    # llm = LLM(model="/data/shai/checkpoint-100")  # replace with your actual model
    llm = LLM(model=args.model)  # replace with your actual model
    
    sampling_params = SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.00, temperature=0.8, top_p=0.80, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643, 151644, 151645], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=2048, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None)
    #sampling_params = SamplingParams(n=5, max_tokens=4096, stop_token_ids=[151643, 151644, 151645])



    # Prepare all prompts
    direct_prompts = ['<|im_start|>assistant\You are a useful chemistry assistant and answer the SMILES generation based question below. Just give your answer alone inside the <answer>...</answer> tags.<|im_end|>\n<|im_start|>user\\'+item[0]['input'].replace("<answer>","")+'<|im_end|>\n<|im_start|>assistant\Your response:\n<answer>' for item in data]
    
    reasoning_prompts = ['<|im_start|>assistant\You are a useful chemistry assistant and answer the SMILES generation based question below. Think your answer in steps in terms of molecule substituent postion and SMILES structures inside the <think>...</think> tags and then give your final answer inside <answer>...</answer> tags.<|im_end|>\n<|im_start|>user\\'+item[0]['input'].replace(" <answer>","")+'<|im_end|>\n<|im_start|>assistant\Your response:\n<think>' for item in data]
    
    # reasoning_prompts= ["<|im_start|>assistant\You are a useful chemistry assistant and answer the question to change IUPAC to SMILES. Reason out your answer inside <think> tags and give your confident final answer inside the answer tags.<|im_end|>\n<|im_start|>user\\"+ item[0]['input'].replace("<answer>","")  +"\nDo only the necessary reasoning and backtracking to get to the final answer<|im_end|>\n<|im_start|>assitant\<think>" for item in data]
    
    _evaluate(direct_prompts, data, llm, sampling_params, os.path.join(args.out_dir, "direct_output.txt"), reasoning=False)
    _evaluate(reasoning_prompts, data, llm, sampling_params, os.path.join(args.out_dir, "reasoning_output.txt"), reasoning=True)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)