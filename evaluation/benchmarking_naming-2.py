import pickle
import re

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

from vllm import LLM, SamplingParams

df = pd.read_csv("/data/david/final_tasks_prompts/reaction_class_prompts_600k.csv")

df = df.iloc[10_000:].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=3000, random_state=42, shuffle=True)

test_df = test_df.drop(columns=["Unnamed: 0", "REACTION"])

test_df = test_df.rename(columns={"REACTION_PROMPT": "input", "CLASS": "answer"})

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
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"([A-Za-z \-]+?)\s*</answer>", text, re.IGNORECASE)
    return m2.group(1).strip() if m2 else None


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

    prompt = (
        "<|im_start|>assistant\n"
        "You are a useful Chemistry assistant.\n"
        "Think about your choice and output your reasoning within <think> ... </think> tags.\n"
        "Make sure to put your final choice inside <answer>…</answer>.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: What is the name of this chemical reaction?\n{inp}\n"
        "Choose **only** from the following list:\n"
        f"{opts}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>"
    )

    prompts.append(prompt)

outputs = llm.generate(prompts, sampling_params)


classes_lc = [c.lower() for c in classes]

correct = 0
records = []

for idx, (ex, out) in enumerate(zip(test_data, outputs), 1):
    gold = ex["answer"].strip().lower()
    hit = False

    print(f"\n--- Example #{idx} ---")
    print("Question:", ex["input"])
    print("Gold class:", gold)
    print()

    for j, sample in enumerate(out.outputs, 1):
        comp = sample.text
        pred = extract_answer(comp)
        if pred:
            ok = gold.lower() in pred.lower()
        else:
            ok = False

        print(f"--- Sample {j} ---")
        print(f"Extracted completion:{comp}")
        print(f"Extracted <answer>: {pred!r}")
        print(f"Count this sample as correct? {'OK' if ok else 'WRONG'}\n")

        if ok:
            hit = True
            correct += 1
            records.append(
                {"example_idx": idx, "gold_answer": gold, "predicted_answer": pred, "full_completion": comp}
            )
            break

    print("Any-of-5 correct overall?", "OK" if hit else "WRONG")

out_df = pd.DataFrame(records)
out_df.to_csv("/data/david/benchmark_save_completion_runs/naming_2_correct_completions_think.csv", index=False)

acc = 100 * correct / len(test_data)
print(f"\nFinal Any-of-5 Accuracy: {acc:.2f}% ({correct}/{len(test_data)})")

metrics = pd.DataFrame([{"accuracy_pct": round(acc, 2), "correct": correct, "total": len(test_data)}])

metrics.to_csv("/data/david/benchmark_save_completion_runs/naming_2_metrics_think.csv", index=False)
