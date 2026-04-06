import json
import os
import pickle
import random
import re
import shutil
from argparse import ArgumentParser

from rdkit import Chem

from open_r1.tasks import ForwardReaction
from vllm import LLM, EngineArgs, SamplingParams


def main():

    llm = LLM(
        model="futurehouse/ether0", generation_config="auto", download_dir="/iopsstor/scratch/cscs/nnguyenx/models/"
    )
    # llm = LLM(model="futurehouse/ether0", download_dir="/iopsstor/scratch/cscs/nnguyenx/models/")
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 2048

    with open("ether0/prompt.txt", "r") as f:
        prompt = f.read()

    conversation = [{"role": "user", "content": prompt}]

    outputs = llm.chat(conversation, sampling_params=sampling_params)

    output = outputs[0].outputs[0].text

    model_prompt = outputs[0].prompt

    response = {
        "model": "futurehouse/ether0",
        "prompt": model_prompt,
        "sampling_params": str(sampling_params),
        "output": output,
    }
    response_print = json.dumps(response, indent=2)

    with open("ether0/output.json", "w") as f:
        f.write(response_print)


if __name__ == "__main__":
    main()
