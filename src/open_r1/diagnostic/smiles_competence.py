
import os
import re
import random
import pandas as pd
from vllm import LLM, SamplingParams
import numpy as np

def load_llm(model):
    llm = LLM(
        model=model,
        max_num_seqs=5,
        max_model_len=256,
        tensor_parallel_size=4,
    )
    params = SamplingParams(
        temperature=0.0,
        prompt_logprobs=1,
        max_tokens=1,
    )
    return llm, params


def prompt_template(smiles):
    tmp = f"""The molecule represented with the SMILES [BEGIN_SMILES] {smiles} [END_SMILES]"""
    return tmp


def corrupt_smi(smiles, corruption_rate=0.2):
    """
    Randomly deletes grammar elements from a SMILES string.
    
    Args:
        smiles (str): The original SMILES string.
        corruption_rate (float): Proportion of grammar elements to remove (0 to 1).
        
    Returns:
        str: The corrupted SMILES string.
    """
    # Define grammar elements to target
    grammar_elements = set('()[]0123456789')
    indices = [i for i, c in enumerate(smiles) if c in grammar_elements]
    
    # Determine how many to remove
    n_remove = max(1, int(len(indices) * corruption_rate)) if indices else 0
    if n_remove == 0:
        return smiles  # Nothing to remove
    
    # Randomly select indices to remove
    remove_indices = set(random.sample(indices, n_remove))
    
    # Build new string
    corrupted = ''.join(c for i, c in enumerate(smiles) if i not in remove_indices)
    return corrupted


def load_dataset(data_dir):
    df = pd.read_csv(data_dir, nrows=10000)
    df['prompt_canon'] = df['SMILES'].apply(prompt_template)
    df['prompt_random'] = df['SMILES_variant1'].apply(prompt_template)
    
    df['corrupt'] = df['SMILES'].apply(corrupt_smi)
    df['prompt_corrupt'] = df['corrupt'].apply(prompt_template)
    return df


from pydantic import BaseModel

class LogprobStat(BaseModel):
    mean_logprob: float
    mean_rank: float
    n_tokens: int
    smiles: str


def run_column(col, llm, params, data):
    def process_out(out):
        try:
            smiles_tokens = out.prompt_token_ids[12:][:-5]
            lps = out.prompt_logprobs[12:][:-5]

            # vllm gives you logprobs for your token, and for rank1 token
            smi_lps = [v[x].logprob for v,x in zip(lps, smiles_tokens)]
            smi_rank = [v[x].rank for v,x in zip(lps, smiles_tokens)]

            smiles = "".join([v[x].decoded_token for v,x in zip(lps, smiles_tokens)])
            smiles = re.sub('Ġ',"","".join(smiles))

            return LogprobStat(
                mean_logprob=np.mean(smi_lps),
                mean_rank=np.mean(smi_rank),
                n_tokens=len(smiles_tokens),
                smiles=smiles
            ).__dict__

        except:
            return LogprobStat(
                mean_logprob=0,
                mean_rank=0,
                n_tokens=1000,
                smiles=""
            ).__dict__

    out = llm.generate(data[col], params)
    lps = [process_out(o) for o in out]

    return pd.DataFrame(lps)

def main(model, data_dir):
    llm, params = load_llm(model)
    data = load_dataset(data_dir)

    out_canon = run_column('prompt_canon', llm, params, data)
    out_random = run_column('prompt_random', llm, params, data)
    out_corrupt = run_column('prompt_corrupt', llm, params, data)

    write_dir = os.path.dirname(data_dir)

    out_canon.to_csv(os.path.join(write_dir, "lps_canonical.csv"), index=False)
    out_random.to_csv(os.path.join(write_dir, "lps_random.csv"), index=False)
    out_corrupt.to_csv(os.path.join(write_dir, "lps_corrup.csv"), index=False)

    # Report some metrics here (e.g. avg difference, diff of averages, relative to canon)
    print("Means:")
    print("CANON", out_canon[['mean_logprob','mean_rank']].mean())
    print("RANDOM", out_random[['mean_logprob','mean_rank']].mean())
    print("CORRUPT", out_corrupt[['mean_logprob','mean_rank']].mean())
    print("----------")

    from numpy import std, mean, sqrt

    #correct if the population S.D. is expected to be equal for the two groups.
    def cohen_d(x,y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

    lp_canon = out_canon['mean_logprob']
    rk_canon = out_canon['mean_rank']
    lp_corrupt = out_corrupt['mean_logprob']
    rk_corrupt = out_corrupt['mean_rank']
    lp_sc = cohen_d(lp_canon, lp_corrupt)
    rk_sc = cohen_d(rk_canon, rk_corrupt)

    print("SCC logprobs", lp_sc)
    print("SCC rank", rk_sc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM evaluation on SMILES data")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model checkpoint or path to use with vllm"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/iopsstor/store/cscs/swissai/a05/LIAC/data/CRLLM-PubChem/CRLLM-PubChem-compounds1M.csv",
        help="Path to the data CSV"
    )
    args = parser.parse_args()

    main(args.model, args.data_dir)