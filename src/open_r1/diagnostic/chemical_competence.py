
import argparse
import os
import re
import random
import pandas as pd
from vllm import LLM, SamplingParams
import numpy as np

from open_r1.diagnostic.utils import cohen_d

def load_llm(model):
    llm = LLM(
        model=model,
        max_num_seqs=1,
        max_model_len=1024,
        tensor_parallel_size=2,
    )
    params = SamplingParams(
        temperature=0.0,
        prompt_logprobs=1,
        max_tokens=1,
    )
    return llm, params

def parse_args():
    parser = argparse.ArgumentParser(description="Chemical competence evaluation")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--data_path", type=str, help="Path to the data directory")
    parser.add_argument("--outdir", type=str, help="Output directory")
    return parser.parse_args()

# def prompt_template(smiles):
#     tmp = f"""The molecule represented with the SMILES [BEGIN_SMILES] {smiles} [END_SMILES]"""
#     return tmp


def load_dataset(data_path):
    df = pd.read_csv(data_path, nrows=1000, sep='\t')
    return df


from pydantic import BaseModel

class LogprobStat(BaseModel):
    mean_logprob: float
    mean_rank: float
    n_tokens: int
    prompt: str


def run_column(col, llm, params, data):
    def process_out(out):
        try:
            token_ids = out.prompt_token_ids[1:]
            lps = out.prompt_logprobs[1:]

            # vllm gives you logprobs for your token, and for rank1 token
            token_lps = [v[x].logprob for v,x in zip(lps, token_ids)]
            token_rank = [v[x].rank for v,x in zip(lps, token_ids)]

            prompt = "".join([v[x].decoded_token for v,x in zip(lps, token_ids)])

            return LogprobStat(
                mean_logprob=np.mean(token_lps),
                mean_rank=np.mean(token_rank),
                n_tokens=len(token_ids),
                prompt=prompt
            ).__dict__
        except:
            # print(out)
            return LogprobStat(
                mean_logprob=0,
                mean_rank=0,
                n_tokens=1000,
                prompt=""
            ).__dict__

    out = llm.generate(data[col], params)
    lps = [process_out(o) for o in out]

    return pd.DataFrame(lps)

def main():
    args = parse_args()
    write_dir = args.outdir
    os.makedirs(write_dir, exist_ok=True)
    
    with open(os.path.join(write_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    llm, params = load_llm(args.model)
    data = load_dataset(args.data_path)

    out_origin = run_column('output', llm, params, data)
    out_corrupt = run_column('corrupted_output', llm, params, data)


    out_origin.to_csv(os.path.join(write_dir, "lps_origin.csv"), index=False, sep='\t')
    out_corrupt.to_csv(os.path.join(write_dir, "lps_corrupted.csv"), index=False, sep='\t')

    # Report some metrics here (e.g. avg difference, diff of averages, relative to canon)
    print("Means:")
    print("ORIGIN", out_origin[['mean_logprob','mean_rank']].mean())
    print("CORRUPT", out_corrupt[['mean_logprob','mean_rank']].mean())
    print("----------")

    print("Mean distribution diff:")
    # print("Canon to Random", (out_origin[['mean_logprob','mean_rank']]- out_random[['mean_logprob','mean_rank']]).mean())
    print("Origin to Corrupt", (out_origin[['mean_logprob','mean_rank']]- out_corrupt[['mean_logprob','mean_rank']]).mean())
    print("----------")

    print("Diff of means:")
    # print("Canon to Random", out_origin[['mean_logprob','mean_rank']].mean() - out_random[['mean_logprob','mean_rank']].mean())
    print("Origin to Corrupt", out_origin[['mean_logprob','mean_rank']].mean() - out_corrupt[['mean_logprob','mean_rank']].mean())
    
    
    lp_origin = out_origin['mean_logprob']
    rk_origin = out_origin['mean_rank']
    lp_corrupt = out_corrupt['mean_logprob']
    rk_corrupt = out_corrupt['mean_rank']
    lp_sc = cohen_d(lp_origin, lp_corrupt)
    rk_sc = cohen_d(rk_origin, rk_corrupt)
    
    print("SCC logprobs", lp_sc)
    print("SCC rank", rk_sc)
    
    res_outpath = os.path.join(write_dir, 'final_results.txt')
    with open(res_outpath, 'w') as f:
        f.write(f"Means:\n")
        f.write(f"ORIGIN {out_origin[['mean_logprob','mean_rank']].mean()}\n")
        f.write(f"CORRUPT {out_corrupt[['mean_logprob','mean_rank']].mean()}\n\n")
        f.write(f"SCC logprobs: {lp_sc}\n")
        f.write(f"SCC rank: {rk_sc}\n")

if __name__=="__main__":
    main()