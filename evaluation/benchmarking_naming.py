#!/usr/bin/env python
import os
import re
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Naming-2 without reasoning")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or Hugging Face repo ID for the vLLM model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input data CSV file")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save output files")
    return parser.parse_args()


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"([A-Za-z \-]+?)\s*</answer>", text, re.IGNORECASE)
    return m2.group(1).strip() if m2 else None


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df = df.iloc[10_000:].reset_index(drop=True)

    _, test_df = train_test_split(
        df,
        test_size=3000,
        random_state=42,
        shuffle=True
    )

    test_df = test_df.drop(columns=[c for c in ["Unnamed: 0", "REACTION"] if c in test_df.columns])
    test_df = test_df.rename(columns={
        "REACTION_PROMPT": "input",
        "CLASS":           "answer"
    })

    test_data = test_df.to_dict(orient="records")

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

    llm = LLM(model=args.model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        temperature=0.8,
        top_p=0.80,
        top_k=20,
        max_tokens=4096,
        stop_token_ids=[151643, 151644, 151645],
        skip_special_tokens=True,
    )

    prompts = []
    opts_text = "\n".join(f"- {cls}" for cls in classes)
    for ex in test_data:
        inp = ex["input"].replace("<answer>", "")
        prompt = (
            "<|im_start|>assistant\n"
            "You are a useful Chemistry assistant.\n"
            "Just put your final choice inside <answer>…</answer>.\n"
            "<|im_end|>\n\n"
            "<|im_start|>user\n"
            f"Question: What is the name of this chemical reaction?\n{inp}\n"
            "Choose **only** from the following list:\n"
            f"{opts_text}\n"
            "<|im_end|>\n\n"
            "<|im_start|>assistant\n"
            "<answer>"
        )
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    records = []
    for idx, (ex, out) in enumerate(zip(test_data, outputs), start=1):
        gold = ex["answer"].strip().lower()
        hit = False

        for sample in out.outputs:
            comp = sample.text
            pred = extract_answer(comp)
            ok = pred and (gold in pred.lower())
            if ok:
                correct += 1
                hit = True
                records.append({
                    "example_idx":      idx,
                    "gold_answer":      gold,
                    "predicted_answer": pred,
                    "full_completion":  comp
                })
                break

    total = len(test_data)
    acc = 100 * correct / total
    print(f"Final Any-of-5 Accuracy: {acc:.2f}% ({correct}/{total})")

    out_df = pd.DataFrame(records)
    out_df.to_csv(os.path.join(args.out_dir, "naming_correct_completions_direct.csv"), index=False)

    metrics = pd.DataFrame([{
        "accuracy_pct": round(acc, 2),
        "correct":      correct,
        "total":        total
    }])
    metrics.to_csv(os.path.join(args.out_dir, "naming_metrics_direct.csv"), index=False)


if __name__ == "__main__":
    main()
