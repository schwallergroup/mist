import argparse
import os
from vllm import LLM, SamplingParams
from rdkit import Chem
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_answer(text):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL|re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"([A-Za-z \-]+?)\s*</answer>", text, re.IGNORECASE)
    return m2.group(1).strip() if m2 else None

def generate_with_retry(prompt, sampling_params, llm, max_retries=1):
    """
    Generate a completion for `prompt`, retrying up to `max_retries` times.
    On each retry, if no <answer>…</answer> tag was found in the previous
    completion, we feed back prompt + last_completion + "\n<answer>".
    Returns the raw text of the first well‐formed completion (or the last attempt).
    """
    last_text = ""
    
    result  = llm.generate([prompt], sampling_params)
    outputs = result[0].outputs
    last_text = outputs[0].text
    
    if extract_answer(last_text) is not None:
        return last_text
    
    for attempt in range(1, max_retries+1):
        print(f"Retry #{attempt}: feeding back previous output + <answer>…")
        
        to_send = prompt + last_text + "\n<answer>"
        
        result  = llm.generate([to_send], sampling_params)
        outputs = result[0].outputs
        last_text = outputs[0].text
        
        if extract_answer(last_text) is not None:
            return last_text

    return last_text

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Naming-2 with retries")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the vLLM model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--data_path", type=str, required=True, default="/data/david/final_tasks_prompts/reaction_class_prompts_600k.csv", help="Path to the input data CSV file")

    return parser.parse_args()

def main(args):
    # df = pd.read_csv("/data/david/final_tasks_prompts/reaction_class_prompts_600k.csv")
    df = pd.read_csv(args.data_path)

    df = df.iloc[10_000:].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=3000,     
        random_state=42,   
        shuffle=True
    )

    test_df = test_df.drop(columns=["Unnamed: 0", "REACTION"])

    test_df = test_df.rename(columns={
        "REACTION_PROMPT": "input",
        "CLASS":           "answer"
    })

    test_data  = test_df.to_dict(orient="records")

    all_keys = set().union(*(rec.keys() for rec in test_data))
    print(all_keys)

    # Load vLLM model
    # llm = LLM(model="/data/share/jgoumaz_grpo_models/checkpoints/rxn_naming_409465/checkpoint-200/")  
    # llm = LLM(model="Qwen/Qwen2.5-3B", trust_remote_code=True)
    # llm = LLM(model="/data/share/qwen_pretranined_v6")  
    llm = LLM(args.model_path)
    
    sampling_params = SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.00, temperature=0.8, top_p=0.80, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643, 151644, 151645], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=4096, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None)
    #sampling_params = SamplingParams(n=5, max_tokens=4096, stop_token_ids=[151643, 151644, 151645])

    classes = [
        "Acylation",
        "Aromatic Heterocycle Formation",
        "C-C Coupling",
        "Deprotection",
        "Functional Group Addition",
        "Functional Group Interconversion",
        "Heteroatom Alkylation and Arylation",
        "Miscellaneous",
        "Protection",
        "Reduction",
    ]


    prompts = []
    for ex in test_data:
        inp = ex["input"].replace("<answer>", "")
        opts = "\n".join(f"- {cls}" for cls in classes)

        # prompt = (
        #     "<|im_start|>assistant\n"
        #     "You are a useful Chemistry assistant.\n"
        #     "Think about your choice and output your reasoning within <think> ... </think> tags.\n"
        #     "Make sure to put your final choice inside <answer>…</answer>.\n"
        #     "<|im_end|>\n"
        #     "<|im_start|>user\n"
        #     f"Question: What is the name of this chemical reaction?\n{inp}\n"
        #     "Choose **only** from the following list:\n"
        #     f"{opts}\n"
        #     "<|im_end|>\n"
        #     "<|im_start|>assistant\n"
        #     "<think>"
        # )
        prompt = f"""<|im_start|>assistant
            You are a useful Chemistry assistant and you will answer the following class prediction question. Give your reasoning inside the <think>...</think> tags and then respond inside <answer>...</answer> tags, think and reason for all the options before giving your answer. Structure your reasoning such that you think through all options before giving the answer.<|im_end|>

            <|im_start|>user
            Question: What is the name of this chemical reaction? {inp}

            Choose ONLY from the following options and write your response choice inside <answer>...</answer>:
            {opts}

            Do not provide final answer different than what it provided in this list. 
            <|im_end|>

            <|im_start|>assistant
            <think>"""

        prompts.append(prompt)

    classes_lc = [c.lower() for c in classes]

    first_results = llm.generate(prompts, sampling_params)

    completions = [res.outputs[0].text for res in first_results]

    needs_retry = [
        i for i, text in enumerate(completions)
        if extract_answer(text) is None
    ]

    if needs_retry:
        retry_prompts = [
            prompts[i] + completions[i] + "\n<answer>"
            for i in needs_retry
        ]

        print(f"Retrying {len(needs_retry)} examples: {needs_retry}")
        retry_results = llm.generate(retry_prompts, sampling_params)

        for idx, res in zip(needs_retry, retry_results):
            old_text = completions[idx]
            new_text = res.outputs[0].text

            new_text = old_text + new_text

            print(f"\nExample #{idx} -- before vs. after:")
            print("-" * 40)
            print("OLD completion:\n", old_text)
            print("\nNEW completion:\n", new_text)

            if extract_answer(new_text) is not None:
                print(f"Replacing completion for example #{idx} with retry result")
                print(f"The new completion becomes:\n{new_text}")
                completions[idx] = new_text
            else:
                print(f"Retrying for example #{idx} still has no <answer> tag; keeping old completion")


    records = []
    correct = 0
    for i, (ex, comp) in enumerate(zip(test_data, completions), start=1):
        gold = ex["answer"].strip().lower()
        pred = extract_answer(comp) or ""
        hit  = gold in pred.lower()
        if hit:
            print(f"\nExample #{i}")
            print("Gold class:   ", ex["answer"].strip().lower())
            print("Full completion:\n" + comp)
            print("-" * 80)
            correct += 1
            records.append({
                "example_idx":      i,
                "gold_answer":      gold,
                "predicted_answer": pred,
                "full_completion":  comp
            })


    out_df = pd.DataFrame(records)
    # out_df.to_csv("/data/david/benchmark_save_completion_runs/naming_2_correct_completions_think_retry_pt.csv", index=False)
    samplewise_csv = os.path.join(args.output_dir, "samplewise_results.csv")
    out_df.to_csv(samplewise_csv, index=False)
        
    acc = 100 * correct / len(test_data)
    print(f"\nFinal Any-of-5 Accuracy: {acc:.2f}% ({correct}/{len(test_data)})")

    metrics = pd.DataFrame([{
        "accuracy_pct":   round(acc, 2),
        "correct":        correct,
        "total":          len(test_data)
    }])

    # metrics.to_csv("/data/david/benchmark_save_completion_runs/naming_2_metrics_think_retry_pt.csv", index=False)
    eval_res_csv = os.path.join(args.output_dir, "evaluation_metrics.csv")
    metrics.to_csv(eval_res_csv, index=False)

if __name__ == "__main__":
    main(args)
