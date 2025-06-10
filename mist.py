import json
import os
import shutil
import random
from vllm import LLM, SamplingParams, EngineArgs
from rdkit import Chem
import re
import pickle
from open_r1.tasks import ForwardReaction
from argparse import ArgumentParser

def main():
    
    llm = LLM(
        model="/capstor/store/cscs/swissai/a05/LIAC/checkpoints/rxnpred_431710/checkpoint-88/",
    )
    # llm = LLM(model="futurehouse/ether0", download_dir="/iopsstor/scratch/cscs/nnguyenx/models/")
    sampling_params = SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.00, temperature=0.01, top_p=0.80, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643, 151644, 151645], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=10000, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None)
    
    with open("mist/prompt.txt", "r") as f:
        prompt = f.read()
    
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    
    output = outputs[0].outputs[0].text
    
    model_prompt = outputs[0].prompt
    
    response = {
        "model": "/capstor/store/cscs/swissai/a05/LIAC/checkpoints/rxnpred_431710/checkpoint-88/",
        "prompt": model_prompt,
        "sampling_params": str(sampling_params),
        "output": output
    }
    response_print = json.dumps(response, indent=2)
    
    with open("mist/output.json", "w") as f:
        f.write(response_print)

if __name__ == "__main__":
    main()
    