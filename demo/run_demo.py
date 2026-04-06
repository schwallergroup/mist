import json
from pathlib import Path

from open_r1.tasks.reactions.forward import ForwardReaction


def main():
    dataset_dir = Path(__file__).resolve().parent / "rxnpred_tiny"
    task = ForwardReaction(dataset_id_or_path=str(dataset_dir))
    dataset = task.load()

    prompts = [
        dataset["test"][0]["problem"],
        dataset["test"][1]["problem"],
    ]
    solutions = [
        dataset["test"][0]["solution"],
        dataset["test"][1]["solution"],
    ]
    completions = [
        "<think>A plausible product is COC based on the reagents.</think><answer>COC</answer>",
        "<think>A plausible product is CCNC based on the reagents.</think><answer>CCNC</answer>",
    ]

    rewards = task.accuracy_reward(completions, solutions, prompts)
    summary = {
        "train_examples": len(dataset["train"]),
        "test_examples": len(dataset["test"]),
        "solutions": solutions,
        "rewards": rewards,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
