import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import Dataset
import random
import re
import os
from datetime import datetime

def extract_answer(text):
    m = re.search(r"ANSWER:\s*([A-Za-z \-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL|re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')
    
    m = re.search(r"(?:^|\n)Answer:\s*\*?\*?([A-Za-z \-]+)", text[-200:], re.IGNORECASE | re.MULTILINE)
    if m:
        answer = m.group(1).strip().rstrip('.')
        for cls in classes:
            if cls.lower() == answer.lower():
                return cls
    
    m = re.search(r"(?:final answer is|the answer is)\s+([A-Za-z \-]+)", text[-200:], re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    last_part = text[-50:].lower()
    matching_classes = [cls for cls in classes if cls.lower() in last_part]
    if len(matching_classes) == 1:
        return matching_classes[0]
    
    return None

model_path = "/capstor/store/cscs/swissai/a131/jmeng/megatron/models/SciReasoner-8B/"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"  
)
print("Model loaded!")

df = pd.read_csv("/capstor/store/cscs/swissai/a131/dsegura/reaction_class_prompts_600k.csv")
df["orig_idx"] = df.index

ds = Dataset.from_pandas(df)
test42 = ds.train_test_split(test_size=0.1, seed=42)["test"]
test43 = ds.train_test_split(test_size=0.1, seed=43)["test"]

idx42 = set(test42["orig_idx"])
idx43 = set(test43["orig_idx"])
common_idx = sorted(idx42 & idx43)

n = 3000
random.seed(42)
selected_idx = random.sample(common_idx, min(n, len(common_idx)))

common_ds = ds.select(selected_idx)
common_df = common_ds.to_pandas().reset_index(drop=True)

test_df = common_df.drop(columns=["Unnamed: 0", "REACTION", "orig_idx"])  
test_df = test_df.rename(columns={"REACTION_PROMPT": "input", "CLASS": "answer"})
test_data = test_df.to_dict(orient="records")

classes = [
    "Acylation", "Aromatic Heterocycle Formation", "C-C Coupling",
    "Deprotection", "Functional Group Addition", "Functional Group Interconversion",
    "Heteroatom Alkylation and Arylation", "Miscellaneous", "Protection", "Reduction"
]

records = []
correct = 0

for i, ex in enumerate(test_data, start=1):
    inp = ex["input"].replace("<answer>", "")
    opts = "\n".join(f"- {cls}" for cls in classes)
    
    prompt = f"""<|im_start|>assistant
You are a useful Chemistry assistant and you will answer the following class prediction question. Give an direct answer inside <answer>...</answer> tags, do not reason and give directly the answer.<|im_end|>

<|im_start|>user
Question: What is the name of this chemical reaction? {inp}

Choose ONLY from the following options and write your response choice inside <answer>...</answer>:
{opts}

Do not provide final answer different than what it provided in this list. 
<|im_end|>

<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.8,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    completion_full = tokenizer.decode(outputs[0], skip_special_tokens=False)

    matches = list(re.finditer(r'<\|im_start\|>assistant\n?(.+?)(?:<\|im_end\|>|$)', completion_full, re.DOTALL))
    if matches:
        completion = matches[-1].group(1).strip() 
    else:
        completion = ""

    if i <= 3:
        print(f"\n{'='*80}")
        print(f"Example {i} - Gold: {ex['answer']}")
        print(f"Number of assistant blocks found: {len(matches)}")
        print(f"Extracted completion: '{completion}'")
        print(f"{'='*80}")
        
    gold = ex["answer"].strip().lower()
    pred = extract_answer(completion) or ""
    hit = gold in pred.lower()
    
    if hit:
        correct += 1
    
    records.append({
        "example_idx": i,
        "gold_answer": gold,
        "predicted_answer": pred,
        "hit": hit,
        "full_completion": completion
    })
    
    print(f"Processed {i}/{len(test_data)} - Acc so far: {100*correct/i:.1f}%")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/iopsstor/scratch/cscs/davidsegura/git/sink/evaluation/SciReasoner-8B_{timestamp}"
os.makedirs(out_dir, exist_ok=True)

out_df = pd.DataFrame(records)
out_df.to_csv(os.path.join(out_dir, "samplewise_results.csv"), index=False)

acc = 100 * correct / len(test_data)
print(f"\nFinal Accuracy: {acc:.2f}% ({correct}/{len(test_data)})")

metrics = pd.DataFrame([{
    "model": "SciReasoner-8B",
    "accuracy_pct": round(acc, 2),
    "correct": correct,
    "total": len(test_data)
}])
metrics.to_csv(os.path.join(out_dir, "evaluation_metrics.csv"), index=False)