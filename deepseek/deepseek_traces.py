import asyncio
import json
import os
import re

import aiohttp
import requests
from dotenv import load_dotenv

load_dotenv()


async def call_deepseek(question, answer):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": "Bearer ", "Content-Type": "application/json"}  # ENTER YOUR KEY HERE
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [{"role": "user", "content": question}],
        "include_reasoning": True,
        "temperature": 0.8,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.9,
        "top_k": 0,
        "top_a": 1,
        "min_p": 0,
        "repetition_penalty": 1,
        "provider": {"sort": "latency"},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                text = await response.json()
                # checks if the content is not None from API and returns them
                if text["choices"][0]["message"]["content"] != None:
                    output = (
                        "<think> "
                        + text["choices"][0]["message"]["content"]
                        + " </think>"
                        + "<answer> "
                        + answer
                        + " </answer>"
                    )
                else:
                    return []
                return [output, question]
    except Exception as e:  # Sometimes the API returns exceptions/blank messages
        print(f"Error with prompt {question}:\n{e}")
        return f"Error: {e}"


async def main():
    file = "./train.txt"  # Replace with your own file containing the questions and answer as a prompt
    question = open(file, "r").read()
    question_buffer = question.split("\n")

    tasks = []
    for question in question_buffer[:-1]:
        match = re.search(r"Answer:\s*(.*)", question)
        ans = match.group(1)
        entity = asyncio.create_task(
            call_deepseek(
                question
                + "\nGive the final reasoning and the final answer in terms of explanations in SMILES by parsing it and explaining it in the final answer correctly.",
                ans,
            )
        )
        tasks.append(entity)
    # Then gather all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    path = "./reactions/"  # Replace the folder in which you want to save the results to.

    for i, result in enumerate(results):
        if result[:5] == "Error":  # Sometimes error gets propagated in the results. See above for more deets
            continue
        f1 = open(path + str(i) + ".txt", "w")
        f1.write(str(result[1]))
        f1.write("\n")
        f1.write(str(result[0]))
        f1.close()
