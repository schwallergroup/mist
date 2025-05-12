
import random
import pandas as pd
from vllm import LLM, SamplingParams

def load_llm(model):
    llm = LLM(
        model=model,
        max_num_seqs=12,
        max_model_len=512,
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
    df = pd.read_csv(data_dir, nrows=100)
    df['prompt_canon'] = df['SMILES'].apply(prompt_template)
    df['prompt_random'] = df['SMILES_variant1'].apply(prompt_template)
    
    df['corrupt'] = df['SMILES'].apply(corrupt_smi)
    df['prompt_corrupt'] = df['corrupt'].apply(prompt_template)
    return df

def run_column(col, llm, params, data):
    def process_out(out):
        lps = out.prompt_logprobs[12:][:-5]
        lps = [v.logprob for v in list(lps.items())]
        return lps

    out = llm.generate(data[col], params)
    lps = [process_out(o) for o in out]

    return lps

def main(model, data_dir):
    llm, params = load_llm(model)
    data = load_dataset(data_dir)

    out_canon = run_column('prompt_canon', llm, params, data)
    out_random = run_column('prompt_random', llm, params, data)
    out_corrupt = run_column('prompt_corrupt', llm, params, data)

    print(pd.DataFrame(out_canon))
    print(pd.DataFrame(out_random))


if __name__=="__main__":
    model = "/LLM_models/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
    data_dir = "/iopsstor/store/cscs/swissai/a05/LIAC/data/CRLLM-PubChem/CRLLM-PubChem-compounds1M.csv"
    main(model, data_dir)