import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import Dataset
import random
import re
import os
from datetime import datetime

classes = [
    "Acylation", "Aromatic Heterocycle Formation", "C-C Coupling",
    "Deprotection", "Functional Group Addition", "Functional Group Interconversion",
    "Heteroatom Alkylation and Arylation", "Miscellaneous", "Protection", "Reduction"
]

def extract_answer(text):
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

prompts = []
for ex in test_data:
    inp = ex["input"].replace("<answer>", "")
    opts = "\n".join(f"- {cls}" for cls in classes)
    
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

print("Generating first completions...")
completions = []

for i, prompt in enumerate(prompts):
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
        completion = completion.replace('<|im_end|>', '').strip()
    else:
        completion = ""
    
    completions.append(completion)
    
    if (i + 1) % 50 == 0:
        print(f"Generated {i + 1}/{len(prompts)} completions...")

needs_retry = [i for i, text in enumerate(completions) if extract_answer(text) is None]

if needs_retry:
    print(f"Retrying {len(needs_retry)} examples")
    
    for idx in needs_retry:
        retry_prompt = prompts[idx] + completions[idx] + "\n<answer>"
        
        inputs = tokenizer(retry_prompt, return_tensors="pt").to(model.device)
        
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
            new_text = matches[-1].group(1).strip()
            new_text = new_text.replace('<|im_end|>', '').strip()
        else:
            new_text = ""
        
        old_text = completions[idx]
        combined_text = old_text + new_text
        
        if extract_answer(combined_text) is not None:
            print(f"Replacing completion for example #{idx} with retry result")
            completions[idx] = combined_text

records = []
correct = 0
for i, (ex, comp) in enumerate(zip(test_data, completions), start=1):
    gold = ex["answer"].strip().lower()
    pred = extract_answer(comp) or ""
    hit = gold in pred.lower()
    if hit:
        correct += 1
        records.append({
            "example_idx": i,
            "gold_answer": gold,
            "predicted_answer": pred,
            "full_completion": comp
        })

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"/iopsstor/scratch/cscs/davidsegura/git/sink/evaluation/SciReasoner-8B_retry_{timestamp}"
os.makedirs(out_dir, exist_ok=True)

out_df = pd.DataFrame(records)
out_df.to_csv(os.path.join(out_dir, "samplewise_results.csv"), index=False)

acc = 100 * correct / len(test_data)
print(f"\nFinal Accuracy: {acc:.2f}% ({correct}/{len(test_data)})")

metrics = pd.DataFrame([{
    "accuracy_pct": round(acc, 2),
    "correct": correct,
    "total": len(test_data)
}])
metrics.to_csv(os.path.join(out_dir, "evaluation_metrics.csv"), index=False)

print(f"Results saved to: {out_dir}")