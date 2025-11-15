# Save this as evaluate_naturelm.py
import requests
import pandas as pd
import random
from datasets import Dataset
import re
import os
from datetime import datetime
import json

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


def extract_answer(text):
    m = re.search(r"ANSWER:\s*([A-Za-z \-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    m = re.search(r"(?:final answer is|the answer is)\s+([A-Za-z \-]+)", text[-200:], re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    last_part = text[-50:].lower()
    matching_classes = [cls for cls in classes if cls.lower() in last_part]
    if len(matching_classes) == 1:
        return matching_classes[0]
    
    return None

print("Using NatureLM server...")
BASE_URL = "http://nid006786:8001"

print("Loading data...")
df = pd.read_csv("/capstor/store/cscs/swissai/a131/dsegura/reaction_class_prompts_600k.csv")
df["orig_idx"] = df.index

ds = Dataset.from_pandas(df)

test42 = ds.train_test_split(test_size=0.1, seed=42)["test"]
test43 = ds.train_test_split(test_size=0.1, seed=43)["test"]

idx42 = set(test42["orig_idx"])
idx43 = set(test43["orig_idx"])

common_idx = sorted(idx42 & idx43)
print(f"Common item check {len(common_idx)} examples in both test42 and test43")

n = 3000
random.seed(42)
selected_idx = random.sample(common_idx, min(n, len(common_idx)))

common_ds = ds.select(selected_idx)

common_df = common_ds.to_pandas().reset_index(drop=True)

print(common_df.shape)    
print(common_df.columns) 

test_df = common_df.drop(columns=["Unnamed: 0", "REACTION", "orig_idx"])  
test_df = test_df.rename(columns={
    "REACTION_PROMPT": "input",
    "CLASS":           "answer"
})

test_data = test_df.to_dict(orient="records")

all_keys = set().union(*(rec.keys() for rec in test_data))
print(all_keys)

prompts = []
for ex in test_data:
    inp = ex["input"].replace("<answer>", "")
    opts = "\n".join(f"- {cls}" for cls in classes)
    
    prompt = f"""You are a chemistry expert. Classify this reaction and respond with your final answer after "ANSWER:".

Question: What is the name of this chemical reaction?
{inp}

Choose ONLY from:
{opts}

Think through each option, then write: ANSWER: [your choice]

Let me think:"""
    
    prompts.append(prompt)

classes_lc = [c.lower() for c in classes]

print("Generating first completions...")
completions = []

for i, prompt in enumerate(prompts):
    try:
        response = requests.post(
            f"{BASE_URL}/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "/capstor/store/cscs/swissai/a131/jmeng/megatron/models/NatureLM-8x7B-Inst/",
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.8,
                "top_p": 0.8,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "stop": ["</answer>"]
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        completion = result["choices"][0]["text"]
        completions.append(completion)
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{len(prompts)} completions...")
    except Exception as e:
        print(f"Error on example {i}: {e}")
        completions.append(f"ERROR: {e}")

needs_retry = [
    i for i, text in enumerate(completions)
    if extract_answer(text) is None
]

if needs_retry:
    print(f"Retrying {len(needs_retry)} examples: {needs_retry}")
    
    for idx in needs_retry:
        retry_prompt = prompts[idx] + completions[idx] + "\n<answer>"
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "/capstor/store/cscs/swissai/a131/jmeng/megatron/models/NatureLM-8x7B-Inst/",
                    "prompt": prompt,
                    "max_tokens": 4096,
                    "temperature": 0.8,
                    "top_p": 0.8,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    # "stop": ["</answer>"]  
                    "stop": ["<|im_end|>", "\n\n\n"] 
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            new_text = result["choices"][0]["text"]
            
            old_text = completions[idx]
            combined_text = old_text + new_text
            
            print(f"\nExample #{idx} -- before vs. after:")
            print("-" * 40)
            print("OLD completion:\n", old_text)
            print("\nNEW completion:\n", combined_text)
            
            if extract_answer(combined_text) is not None:
                print(f"Replacing completion for example #{idx} with retry result")
                completions[idx] = combined_text
            else:
                print(f"Retrying for example #{idx} still has no <answer> tag; keeping old completion")
        except Exception as e:
            print(f"Retry failed for example {idx}: {e}")

records = []
correct = 0
for i, (ex, comp) in enumerate(zip(test_data, completions), start=1):
    gold = ex["answer"].strip().lower()
    pred = extract_answer(comp) or ""
    hit = gold in pred.lower()
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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/iopsstor/scratch/cscs/davidsegura/git/sink/evaluation/NatureLM-8x7B_{timestamp}"
os.makedirs(out_dir, exist_ok=True)

out_df = pd.DataFrame(records)
samplewise_csv = os.path.join(out_dir, "samplewise_results.csv")
out_df.to_csv(samplewise_csv, index=False)

acc = 100 * correct / len(test_data)
print(f"\nFinal Accuracy: {acc:.2f}% ({correct}/{len(test_data)})")

metrics = pd.DataFrame([{
    "accuracy_pct":   round(acc, 2),
    "correct":        correct,
    "total":          len(test_data)
}])

eval_res_csv = os.path.join(out_dir, "evaluation_metrics.csv")
metrics.to_csv(eval_res_csv, index=False)

print(f"Results saved to: {out_dir}")