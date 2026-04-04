import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel


DEFAULT_DATA_PATH = "${MIST_DATA_DIR}/CRLLM-PubChem-compounds1M.csv"
DEFAULT_OUTPUT_DIR = "${MIST_OUTPUT_DIR}/scs"


def load_llm(model, max_num_seqs, max_model_len, tensor_parallel_size, dtype):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    )
    params = SamplingParams(
        temperature=0.0,
        prompt_logprobs=1,
        max_tokens=1,
    )
    return llm, params


def prompt_template(smiles):
    return f"The molecule represented with the SMILES [BEGIN_SMILES] {smiles} [END_SMILES]"


def corrupt_smi(smiles, rng, corruption_rate=0.2):
    """Randomly delete grammar characters from a SMILES string."""
    grammar_elements = set("()[]0123456789")
    indices = [i for i, char in enumerate(smiles) if char in grammar_elements]

    n_remove = max(1, int(len(indices) * corruption_rate)) if indices else 0
    if n_remove == 0:
        return smiles

    remove_indices = set(rng.sample(indices, n_remove))
    return "".join(char for i, char in enumerate(smiles) if i not in remove_indices)


def load_dataset(data_path, num_rows, corruption_rate, seed):
    df = pd.read_csv(data_path, nrows=num_rows)

    required_columns = {"SMILES", "SMILES_variant1"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {data_path}: {', '.join(sorted(missing))}"
        )

    rng = random.Random(seed)
    df["prompt_canon"] = df["SMILES"].apply(prompt_template)
    df["prompt_random"] = df["SMILES_variant1"].apply(prompt_template)
    df["corrupt"] = df["SMILES"].apply(
        lambda smiles: corrupt_smi(smiles, rng, corruption_rate=corruption_rate)
    )
    df["prompt_corrupt"] = df["corrupt"].apply(prompt_template)
    return df


class LogprobStat(BaseModel):
    mean_logprob: float
    mean_rank: float
    n_tokens: int
    smiles: str


def process_generation_output(out):
    try:
        smiles_tokens = out.prompt_token_ids[12:-5]
        logprobs = out.prompt_logprobs[12:-5]

        token_logprobs = [values[token_id].logprob for values, token_id in zip(logprobs, smiles_tokens)]
        token_ranks = [values[token_id].rank for values, token_id in zip(logprobs, smiles_tokens)]
        smiles = "".join(
            values[token_id].decoded_token for values, token_id in zip(logprobs, smiles_tokens)
        )
        smiles = re.sub("Ġ", "", smiles)

        return LogprobStat(
            mean_logprob=float(np.mean(token_logprobs)),
            mean_rank=float(np.mean(token_ranks)),
            n_tokens=len(smiles_tokens),
            smiles=smiles,
        ).model_dump()
    except Exception:
        return LogprobStat(
            mean_logprob=0.0,
            mean_rank=0.0,
            n_tokens=1000,
            smiles="",
        ).model_dump()


def run_column(column_name, llm, params, data, batch_size):
    prompts = data[column_name].tolist()
    total = len(prompts)
    rows = []
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]
        outputs = llm.generate(batch_prompts, params)
        rows.extend(process_generation_output(output) for output in outputs)

        elapsed = time.time() - start_time
        print(
            f"[{column_name}] processed {batch_end}/{total} prompts "
            f"(batch_size={batch_size}, elapsed={elapsed:.1f}s)",
            flush=True,
        )

    return pd.DataFrame(rows)


def cohen_d(x, y):
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled = np.sqrt(((nx - 1) * std_x**2 + (ny - 1) * std_y**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled


def summarize_results(out_canon, out_random, out_corrupt):
    means = {
        "canonical": out_canon[["mean_logprob", "mean_rank"]].mean().to_dict(),
        "randomized": out_random[["mean_logprob", "mean_rank"]].mean().to_dict(),
        "corrupted": out_corrupt[["mean_logprob", "mean_rank"]].mean().to_dict(),
    }

    return {
        "means": means,
        "scs_logprob": float(cohen_d(out_canon["mean_logprob"], out_corrupt["mean_logprob"])),
        "scs_rank": float(cohen_d(out_canon["mean_rank"], out_corrupt["mean_rank"])),
        "n_examples": int(len(out_canon)),
    }


def ensure_output_dir(path):
    output_dir = expand_path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def expand_path(path):
    return Path(os.path.expandvars(str(path))).expanduser()


def run_scs(
    model,
    data_path,
    output_dir,
    num_rows,
    tensor_parallel_size,
    max_num_seqs,
    max_model_len,
    batch_size,
    dtype,
    corruption_rate,
    seed,
):
    llm, params = load_llm(
        model=model,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
    )
    data = load_dataset(
        data_path=data_path,
        num_rows=num_rows,
        corruption_rate=corruption_rate,
        seed=seed,
    )

    print(f"Loaded {len(data)} examples from {data_path}", flush=True)
    print("Running canonical prompts", flush=True)
    out_canon = run_column("prompt_canon", llm, params, data, batch_size=batch_size)
    print("Running randomized prompts", flush=True)
    out_random = run_column("prompt_random", llm, params, data, batch_size=batch_size)
    print("Running corrupted prompts", flush=True)
    out_corrupt = run_column("prompt_corrupt", llm, params, data, batch_size=batch_size)

    write_dir = ensure_output_dir(output_dir)
    out_canon.to_csv(write_dir / "lps_canonical.csv", index=False)
    out_random.to_csv(write_dir / "lps_random.csv", index=False)
    out_corrupt.to_csv(write_dir / "lps_corrupt.csv", index=False)

    summary = summarize_results(out_canon, out_random, out_corrupt)
    (write_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("Means:", flush=True)
    print("CANON", out_canon[["mean_logprob", "mean_rank"]].mean(), flush=True)
    print("RANDOM", out_random[["mean_logprob", "mean_rank"]].mean(), flush=True)
    print("CORRUPT", out_corrupt[["mean_logprob", "mean_rank"]].mean(), flush=True)
    print("----------", flush=True)
    print("SCS logprobs", summary["scs_logprob"], flush=True)
    print("SCS rank", summary["scs_rank"], flush=True)
    print(f"Wrote outputs to {write_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MiST SCS diagnostic on a SMILES dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model checkpoint, Hugging Face model id, or local path to use with vLLM.",
    )
    parser.add_argument(
        "--data-path",
        "--data_dir",
        dest="data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV used for SCS evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-condition CSVs and summary.json will be written.",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=10000,
        help="Number of rows to evaluate from the input CSV.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size. Use 1 for a single-GPU reviewer run.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=5,
        help="vLLM max_num_seqs value.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=256,
        help="vLLM max_model_len value.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of prompts evaluated per vLLM generate call.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="half",
        help="vLLM dtype value, for example half, bfloat16, or float32.",
    )
    parser.add_argument(
        "--corruption-rate",
        type=float,
        default=0.2,
        help="Fraction of grammar characters removed when building corrupted SMILES prompts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for corrupted SMILES generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_scs(
        model=args.model,
        data_path=expand_path(args.data_path),
        output_dir=args.output_dir,
        num_rows=args.num_rows,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        dtype=args.dtype,
        corruption_rate=args.corruption_rate,
        seed=args.seed,
    )
