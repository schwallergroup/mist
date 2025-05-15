import json
import os
import shutil
import random
from vllm import LLM, SamplingParams
from rdkit import Chem
import re
import pickle
from open_r1.tasks import ForwardReaction
from argparse import ArgumentParser


def arg_parse():
    parser = ArgumentParser(description="Evaluate the model on the USPTO dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--task_mode", type=str, required=True, help="Path to the evaluation data.")
    parser.add_argument("--datapath", type=str, default="/data/USPTO/USPTO_480k_clean_no_sft/", help="Path to the evaluation data.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples generated.")
    parser.add_argument("--maxlength", type=int, default=4096, help="Maximum length of the output sequence.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling.")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save the evaluation results.")
    return parser.parse_args()

def save_args(args, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(vars(args), f)

def main():
    args = arg_parse()
    os.makedirs(args.save_dir, exist_ok=True)

    task = ForwardReaction(task_mode=args.task_mode, dataset_id_or_path=args.datapath)
    # Sample data list
    data = task.load()['test']
    # randomly select 100 samples
    # random.seed(42)
    print("Seed set to 42")
    # data = random.sample(data, 100)
    
    data = data.shuffle(seed=42).select(range(500))
    # Load vLLM model
    llm = LLM(model=args.model)  # replace with your actual model

    sampling_params = SamplingParams(n=args.n_samples, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.00, temperature=args.temperature, top_p=0.80, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643, 151644, 151645], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=args.maxlength, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None)
    #sampling_params = SamplingParams(n=5, max_tokens=4096, stop_token_ids=[151643, 151644, 151645])

    def extract_smiles(text, prompt="", relax=False):
        # match = re.search(r'<answer>\s*\[START_SMILES\]\s*(.*?)\s*\[END_SMILES\]\s*</answer>', text)
        # return match.group(1).strip() if match else None
        res = task.extract_smiles_from_answer(task.preprocess_response(text), prompt)
        if res is None and relax: # if relax is True, extract the longest SMILES in the reponse that is different from those in the prompt
            res = task.extract_smiles_from_answer(text, prompt)
        return res

    def extract_normal_smiles(text):
        match = re.search(r'\[START_SMILES\](.*?)\[END_SMILES\]', text)
        return match.group(1).strip() if match else None


    def same_molecule(smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if not mol1 or not mol2:
            return False
        return Chem.MolToSmiles(mol1, canonical=True) == Chem.MolToSmiles(mol2, canonical=True)

    correct = 0
    total = len(data)
    
    outpath = os.path.join(args.save_dir, f"output.txt")
    if os.path.exists(outpath):
        shutil.copyfile(outpath, outpath.replace(".txt", ".txt.old"))
        os.remove(outpath)
        
    arg_outpath = os.path.join(args.save_dir, f"args.json")
    save_args(args, arg_outpath)

    # prompts= ["<|im_start|>assistant\You are a useful chemistry assistant and answer the question to change IUPAC to SMILES. Reason out your answer inside <think> tags and give your confident final answer inside the answer tags.<|im_end|>\n<|im_start|>user\\"+ item[0]['input'].replace("<answer>","")  +"\nDo only the necessary reasoning and backtracking to get to the final answer<|im_end|>\n<|im_start|>assitant\<think>" for item in data]
    prompts = [each['problem'] for each in data]
    outputs = llm.generate(prompts, sampling_params)
    
        
    def print_to_file(content, mode="a"):
        file_path=outpath
        with open(file_path, mode) as f:
            f.write(content + "\n")
        print(content)

    # with open(os.path.join(args.save_dir, "output.txt"), ":
    # Evaluate each example
    for idx, (output, prompt) in enumerate(zip(outputs, prompts)):
        for j in range(args.n_samples):
            predicted = output.outputs[j].text
            predicted_smiles = extract_smiles(predicted, prompt)
            # predicted_smiles_relaxed = extract_smiles(predicted, prompt, relax=True)
            # reference_smiles = extract_normal_smiles(data[idx][0]['answer'])
            reference_smiles = data[idx]['solution']

            print_to_file(f"[[EXAMPLE {idx+1}]]")
            print_to_file(f"[Prompt]: {prompts[idx]}")
            print_to_file(f"[Reference SMILES]: {reference_smiles}")
            print_to_file(f"[Predicted Output]: {predicted}")
            print_to_file(f"[Extracted SMILES]: {predicted_smiles}")

            if predicted_smiles is None:
                continue
            elif ';' not in predicted_smiles:
                if predicted_smiles and same_molecule(reference_smiles, predicted_smiles):
                    print_to_file("[✔️ Match]")
                    correct += 1
                    break
                else:
                    print_to_file("[❌ No Match]")
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
    print_to_file(f"[[Final Accuracy: {accuracy:.2f}% ({correct}/{total})]]")
    
if __name__ == "__main__":
    main()
