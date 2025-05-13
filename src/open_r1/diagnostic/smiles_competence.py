
import os
import re
import random
import pandas as pd
from vllm import LLM, SamplingParams
import numpy as np

def load_llm(model):
    llm = LLM(
        model=model,
        max_num_seqs=12,
        max_model_len=512,
        tensor_parallel_size=1,
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
    df = pd.read_csv(data_dir, nrows=100)
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

    print("Mean distribution diff:")
    print("Canon to Random", (out_canon[['mean_logprob','mean_rank']]- out_random[['mean_logprob','mean_rank']]).mean())
    print("Canon to Corrupt", (out_canon[['mean_logprob','mean_rank']]- out_corrupt[['mean_logprob','mean_rank']]).mean())
    print("----------")

    print("Diff of means:")
    print("Canon to Random", out_canon[['mean_logprob','mean_rank']].mean() - out_random[['mean_logprob','mean_rank']].mean())
    print("Canon to Corrupt", out_canon[['mean_logprob','mean_rank']].mean() - out_corrupt[['mean_logprob','mean_rank']].mean())


if __name__=="__main__":
    model = "Qwen/Qwen2.5-0.5B"
    data_dir = "src/open_r1/diagnostic/data.csv"
    main(model, data_dir)