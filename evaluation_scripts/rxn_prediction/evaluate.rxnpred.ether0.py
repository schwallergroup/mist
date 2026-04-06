import json
import os
import pickle
import random
import re
import shutil
from argparse import ArgumentParser

from rdkit import Chem

from open_r1.tasks import ForwardReaction
from vllm import LLM, SamplingParams


def arg_parse():
    parser = ArgumentParser(description="Evaluate the model on the USPTO dataset.")
    parser.add_argument("--task_mode", type=str, required=True, help="Path to the evaluation data.")
    parser.add_argument(
        "--datapath", type=str, default="/data/USPTO/USPTO_480k_clean_no_sft/", help="Path to the evaluation data."
    )
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples generated.")
    # parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling.")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save the evaluation results.")
    return parser.parse_args()


def save_args(args, save_path: str):
    with open(save_path, "w") as f:
        json.dump(vars(args), f)


def main():
    args = arg_parse()
    os.makedirs(args.save_dir, exist_ok=True)

    task = ForwardReaction(task_mode=args.task_mode, dataset_id_or_path=args.datapath)
    # Sample data list
    data = task.load()["test"]
    # randomly select 100 samples
    # random.seed(42)
    # print("Seed set to 42")
    # data = random.sample(data, 100)

    if len(data) > 500:
        data = data.shuffle(seed=42).select(range(500))
    # Load vLLM model
    llm = LLM(
        model="futurehouse/ether0",
        generation_config="auto",
        download_dir="/iopsstor/scratch/cscs/nnguyenx/models/",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.7,
    )  # replace with your actual model

    sampling_params = llm.get_default_sampling_params()
    sampling_params.n = args.n_samples
    sampling_params.max_tokens = 2048
    # sampling_params = SamplingParams(n=5, max_tokens=4096, stop_token_ids=[151643, 151644, 151645])

    def extract_smiles(text, prompt="", relax=False):
        """Example answer
        <|think_start|>...<|think_end|><|answer_start|>c1cc2c(c(cc(c2)OC)OC)c(n1)Oc3ccc(OC)c(F)c3<|answer_end|>
        """
        match = re.search(r"<\|answer_start\|>(.*?)<\|answer_end\|>", text)
        if match is None:
            return None
        return match.group(1).strip()

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
    prompts = [each["problem"] for each in data]
    outputs = llm.generate(prompts, sampling_params)

    def print_to_file(content, mode="a"):
        file_path = outpath
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
            reference_smiles = data[idx]["solution"]

            print_to_file(f"[[EXAMPLE {idx+1}]]")
            print_to_file(f"[Prompt]: {prompts[idx]}")
            print_to_file(f"[Reference SMILES]: {reference_smiles}")
            print_to_file(f"[Predicted Output]: {predicted}")
            print_to_file(f"[Extracted SMILES]: {predicted_smiles}")

            if predicted_smiles is None:
                continue
            elif ";" not in predicted_smiles:
                if predicted_smiles and same_molecule(reference_smiles, predicted_smiles):
                    print_to_file("[✔️ Match]")
                    correct += 1
                    break
                else:
                    print_to_file("[❌ No Match]")
            else:
                split_smile = predicted_smiles.split(";")
                flag = 0
                if ";" not in reference_smiles:
                    continue
                if len(split_smile) != len(reference_smiles.split(";")):
                    continue
                for smile_num in range(len(split_smile)):
                    if split_smile[smile_num] and same_molecule(
                        reference_smiles.split(";")[smile_num], split_smile[smile_num]
                    ):
                        print_to_file("✔️ Match")
                    else:
                        print_to_file("❌ No Match")
                        flag = 1
                        break
                if flag == 0:
                    correct += 1
                    break

            print_to_file("=" * 60)

    accuracy = 100.0 * correct / total
    print_to_file(f"[[Final Accuracy: {accuracy:.2f}% ({correct}/{total})]]")


if __name__ == "__main__":
    main()
