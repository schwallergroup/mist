import requests
import pandas as pd
import random
from datasets import Dataset
import re
import os
from datetime import datetime

BASE_URL = "http://nid006786:8001"
MODEL_PATH = "/capstor/store/cscs/swissai/a131/jmeng/megatron/models/NatureLM-8x7B-Inst/"
DATA_PATH = "/capstor/store/cscs/swissai/a131/dsegura/reaction_class_prompts_600k.csv"
OUTPUT_BASE = "/iopsstor/scratch/cscs/davidsegura/git/sink/evaluation"

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
    # The prompt ends with "The answer is:", so the model completion is the answer.
    # We just take the first line of the completion and strip whitespace.
    first_line = text.split('\n')[0].strip()
    return first_line

print("Using NatureLM server (Direct Answer Mode)...")

df = pd.read_csv(DATA_PATH)
df["orig_idx"] = df.index

ds = Dataset.from_pandas(df)

test42 = ds.train_test_split(test_size=0.1, seed=42)["test"]
test43 = ds.train_test_split(test_size=0.1, seed=43)["test"]

idx42 = set(test42["orig_idx"])
idx43 = set(test43["orig_idx"])

common_idx = sorted(idx42 & idx43)
print(f"Found {len(common_idx)} examples in both test42 and test43")

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

prompts = []
opts_text = "\n".join(f"- {cls}" for cls in classes)

for ex in test_data:
    inp = ex["input"].replace("<answer>", "")
    # *** FIXED BUG HERE: Changed {opts} to {opts_text} ***
    prompt = (
        f"""You are a chemistry expert. Classify this reaction and respond with your final answer after "ANSWER:".

        Question: What is the name of this chemical reaction?
        {inp}

        Choose ONLY from:
        {opts_text}

        Think through each option, then write: ANSWER: [your choice]

        The answer is:"""
        )
    prompts.append(prompt)

completions = []
records = []
correct = 0

print("Generating completions...")

for i, (prompt, ex) in enumerate(zip(prompts, test_data), start=1):
    try:
        response = requests.post(
            f"{BASE_URL}/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_PATH,
                "prompt": prompt,
                "max_tokens": 50,  
                "temperature": 0.8,
                "top_p": 0.8,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "stop": ["\n", "\n\n", "."] 
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        completion = result["choices"][0]["text"]
    except Exception as e:
        print(f"Error on example {i}: {e}")
        completion = ""

    gold = ex["answer"].strip().lower()
    pred = extract_answer(completion)
    
    hit = False
    if pred and gold in pred.lower():
        hit = True
        correct += 1
    
    records.append({
        "example_idx":      i,
        "gold_answer":      gold,
        "predicted_answer": pred,
        "full_completion":  completion,
        "is_correct":       hit
    })

    if i % 100 == 0:
        print(f"Processed {i}/{len(prompts)}... Current Acc: {100 * correct / i:.2f}%")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"{OUTPUT_BASE}/NatureLM-8x7B_Direct_{timestamp}"
os.makedirs(out_dir, exist_ok=True)

out_df = pd.DataFrame(records)
out_df.to_csv(os.path.join(out_dir, "samplewise_results.csv"), index=False)

final_acc = 100 * correct / len(test_data)
print(f"\nFinal Accuracy: {final_acc:.2f}% ({correct}/{len(test_data)})")

metrics = pd.DataFrame([{
    "accuracy_pct":   round(final_acc, 2),
    "correct":        correct,
    "total":          len(test_data)
}])

metrics.to_csv(os.path.join(out_dir, "evaluation_metrics.csv"), index=False)

print(f"Results saved to: {out_dir}")