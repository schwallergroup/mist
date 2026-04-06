import pickle
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

from vllm import LLM, SamplingParams

df = pd.read_csv("/data/david/final_tasks_prompts/dataset_swapped500k_prompt.csv")

df = df.iloc[10_000:].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=3000, random_state=42, shuffle=True)
print(test_df.columns)

test_data = test_df.to_dict(orient="records")

all_keys = set().union(*(rec.keys() for rec in test_data))
print(all_keys)

# Load vLLM model
llm = LLM(model="/data/share/sft_hf_3/")
sampling_params = SamplingParams(
    n=1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repetition_penalty=1.00,
    temperature=0.8,
    top_p=0.80,
    top_k=20,
    min_p=0.0,
    seed=None,
    stop=[],
    stop_token_ids=[151643, 151644, 151645],
    bad_words=[],
    include_stop_str_in_output=False,
    ignore_eos=False,
    max_tokens=4096,
    min_tokens=0,
    logprobs=None,
    prompt_logprobs=None,
    skip_special_tokens=True,
    spaces_between_special_tokens=True,
    truncate_prompt_tokens=None,
    guided_decoding=None,
)
# sampling_params = SamplingParams(n=5, max_tokens=4096, stop_token_ids=[151643, 151644, 151645])


def extract_answer(text):
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"([ABCD])\s*</answer>", text, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    return None


letters = ["A", "B", "C", "D"]


prompts = []
gold_letters = []
for ex in test_data:
    true_rx = ex["true_reaction"]
    fakes = [ex["fake1"], ex["fake2"], ex["fake3"]]

    opts = np.random.permutation([true_rx] + fakes).tolist()

    gold_letters.append(letters[opts.index(true_rx)])

    prompt = (
        "<|im_start|>assistant\n"
        "You are a useful Chemistry assistant and will answer the following MCQ.\n"
        "Give your reasoning inside <think>...</think> tags, then respond with the option letter\n"
        "inside <answer>...</answer> tags. Think through all four options (A, B, C, D)\n"
        "before choosing your final answer.\n"
        "<|im_end|>\n\n"
        "<|im_start|>user\n"
        "Question: Which chemical reaction is correct? Choose from the following options:\n"
        f"A. {opts[0]}\n"
        f"B. {opts[1]}\n"
        f"C. {opts[2]}\n"
        f"D. {opts[3]}\n"
        "Make sure to give your choice A, B, C, or D inside the <answer>...</answer> tags.\n"
        "<|im_end|>\n\n"
        "<|im_start|>assistant\n"
        "<think>"
    )
    prompts.append(prompt)

if needs_retry:
    retry_prompts = [prompts[i] + completions[i] + "\n<answer>" for i in needs_retry]
    print(f"Retrying {len(needs_retry)} examples: {needs_retry}")
    retry_results = llm.generate(retry_prompts, sampling_params)

    for idx, res in zip(needs_retry, retry_results):
        old = completions[idx]
        new = res.outputs[0].text

        print(f"\nExample #{idx} – before vs. after")
        print("OLD:", old)
        print("NEW:", new)

        if extract_answer(new) is not None:
            merged = old + new
            print("Merged retry for index", idx)
            completions[idx] = merged
        else:
            print("Still no tag at index", idx)

correct = 0
records = []
for i, (gold, comp) in enumerate(zip(gold_letters, completions), start=1):
    pred = extract_answer(comp)
    hit = pred == gold
    if hit:
        correct += 1
        records.append({"example_idx": i, "gold_letter": gold, "predicted": pred, "completion": comp})
    print(f"Example #{i}: gold={gold} pred={pred} → {'OK' if hit else 'WRONG'}")

acc = 100 * correct / len(test_data)
print(f"\nFinal any-of-5 accuracy: {acc:.2f}% ({correct}/{len(test_data)})")

out_df = pd.DataFrame(records)
out_df.to_csv("/data/david/benchmark_save_completion_runs/inversion_2_correct_completions_think.csv", index=False)
print(f"Saved {len(out_df)} correct completions to inversion_2_correct_completions_think.csv")

metrics = pd.DataFrame([{"accuracy_pct": round(acc, 2), "correct": correct, "total": len(test_data)}])

metrics.to_csv("/data/david/benchmark_save_completion_runs/inversion_2_metrics_think.csv", index=False)
