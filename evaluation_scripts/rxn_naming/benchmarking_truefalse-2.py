from vllm import LLM, SamplingParams
from rdkit import Chem
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("/data/david/final_tasks_prompts/is_it_correct_reaction_1M.csv")

df = df.iloc[10_000:].reset_index(drop=True)

train_df, test_df = train_test_split(
    df,
    test_size=3000,     
    random_state=42,   
    shuffle=True
)

print(test_df.columns)

test_data  = test_df.to_dict(orient="records")

all_keys = set().union(*(rec.keys() for rec in test_data))
print(all_keys)

# Load vLLM model
llm = LLM(model="/data/share/sft_hf_3/")  
sampling_params = SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.00, temperature=0.8, top_p=0.80, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643, 151644, 151645], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=4096, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None)
#sampling_params = SamplingParams(n=5, max_tokens=4096, stop_token_ids=[151643, 151644, 151645])

TEMPLATE = """<|im_start|>assistant
You are a useful Chemistry assistant and will answer whether the reaction is True or False.
You are allowed to think about the question and you must output your reasoning within <think> ... </think> tags. Your will need to respond with True or False within <answer>True</answer> or <answer>False</answer>.
<|im_end|>

<|im_start|>user
Question: Is this chemical reaction correct?
{reaction}

Make sure to reason about the mechanism in <think> tags.
<|im_end|>

<|im_start|>assistant
<think>"""

def extract_answer_tag(text: str):

    """
    1) Try to pull 'True' or 'False' out of a well-formed <answer>…</answer>.
    2) Otherwise, grab the last occurrence of true/false/correct/incorrect
       before </answer> and normalize to 'true'/'false'.
    """
    m = re.search(r"<answer>\s*(true|false)\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    idx = text.lower().rfind("</answer>")
    if idx == -1:
        return None

    snippet = text[:idx]
    toks = re.findall(r"\b(true|false|correct|incorrect)\b", snippet, re.IGNORECASE)
    if not toks:
        return None

    last = toks[-1].lower()
    if last == "correct":
        return "true"
    if last == "incorrect":
        return "false"
    return last

for sample in [
    " The reaction is correct. </answer>",
    " The reaction is True. </answer>",
    " False </answer>",
    "<answer>false</answer>",
]:
    print(sample, "->", extract_answer_tag(sample))


prompts     = [TEMPLATE.format(reaction=rec["reaction"]) for rec in test_data]
gold_labels = [str(rec["label"]).strip().lower()    for rec in test_data]


outputs = llm.generate(prompts, sampling_params)


correct = 0
records = []

for idx, (rec, out, gold) in enumerate(zip(test_data, outputs, gold_labels), 1):
    print(f"\n=== Example #{idx} ===\nPrompt: {rec['reaction']}\nGold: {gold}\n")
    hit = False

    for j, sample in enumerate(out.outputs, 1):
        comp = sample.text           
        pred = extract_answer_tag(comp) 
        print(f" Sample {j} raw: {comp!r} -> extracted: {pred!r}")
        if pred == gold:
            print("Correct!")
            correct += 1
            hit = True
            records.append({
                "example_idx":   idx,
                "prompt":        rec["reaction"],
                "gold_label":    gold,
                "predicted":     pred,
                "completion":    comp
            })
            break
        else:
            print("Incorrect")

    print("Any-of-5 correct?", "OK" if hit else "WRONG")


out_df = pd.DataFrame(records)
out_df.to_csv("/data/david/benchmark_save_completion_runs/true_false_2_correct_completions_think.csv", index=False)
print(f"\nSaved {len(records)} correct completions to true_false_2_correct_completions_think.csv")

accuracy = 100 * correct / len(test_data)
print(f"\nFinal any-of-5 accuracy: {accuracy:.2f}% ({correct}/{len(test_data)})")

metrics = pd.DataFrame([{
    "accuracy_pct":   round(accuracy, 2),
    "correct":        correct,
    "total":          len(test_data)
}])

metrics.to_csv("/data/david/benchmark_save_completion_runs/true_false_2_metrics_think.csv", index=False)