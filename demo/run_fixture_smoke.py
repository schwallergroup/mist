import json
from pathlib import Path

from make_kinetic_tiny import ensure_kinetic_tiny
from open_r1.tasks import CHEMTASKS


def summarize_dataset(task_name, task):
    dataset = task.load()
    summary = {
        "train_examples": len(dataset["train"]),
        "test_examples": len(dataset["test"]),
    }
    if len(dataset["train"]):
        summary["train_columns"] = sorted(dataset["train"].column_names)
        summary["train_solution_example"] = dataset["train"][0]["solution"]
    if len(dataset["test"]):
        summary["test_solution_example"] = dataset["test"][0]["solution"]
    return summary


def main():
    demo_dir = Path(__file__).resolve().parent
    datasets_dir = demo_dir / "datasets"
    kinetic_dir = ensure_kinetic_tiny(demo_dir / "kinetic_tiny")

    task_configs = {
        "rxnpred": demo_dir / "rxnpred_tiny",
        "iupacsm": datasets_dir / "CRLLM-PubChem-compounds1M.sample.csv",
        "iupacsm_with_tags": datasets_dir / "CRLLM-PubChem-compounds1M-simple.sample.csv",
        "canonic": datasets_dir / "CRLLM-PubChem-compounds1M.sample.csv",
        "canonmc": datasets_dir / "CRLLM-PubChem-compounds1M.sample.csv",
        "smi_permute": datasets_dir / "CRLLM-PubChem-compounds1M-very_very_simple.sample.csv",
        "smhydrogen": datasets_dir / "CRLLM-PubChem-compounds1M_hydrogen.sample.csv",
        "kinetic": kinetic_dir,
    }

    summary = {}
    for task_name, dataset_path in task_configs.items():
        task_class = CHEMTASKS[task_name]
        task = task_class(dataset_id_or_path=str(dataset_path))
        summary[task_name] = {
            "dataset_path": str(dataset_path),
            **summarize_dataset(task_name, task),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
