from vllm import LLM, SamplingParams
from numpy import std, mean, sqrt

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

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)