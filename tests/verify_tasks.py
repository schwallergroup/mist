import inspect
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_r1 import tasks
from open_r1.tasks import CHEMTASKS
from open_r1.tasks.base import RLTask


def verify_tasks():
    # Get all RLTask subclasses
    task_classes = [
        name
        for name, obj in inspect.getmembers(tasks)
        if inspect.isclass(obj) and issubclass(obj, RLTask) and obj is not RLTask
    ]

    # Base paths
    recipes_path = "recipes"
    docs_path = "docs/source/tasks"

    missing_files = []

    for task_key, task_class in CHEMTASKS.items():
        if task_class.__name__ not in task_classes:
            missing_files.append(f"Task `{task_key}` is not a subclass of RLTask")
            continue

        # Check for recipe file using task_key
        recipe_file = os.path.join(recipes_path, f"{task_key}.yaml")
        if not os.path.isfile(recipe_file):
            missing_files.append(f"Missing recipe for task `{task_key}`")

        # Check for documentation entry using task_key
        docs_file = os.path.join(docs_path, f"{task_key}.rst")
        if not os.path.isfile(docs_file):
            missing_files.append(f"Missing documentation for task `{task_key}`")

    return missing_files


if __name__ == "__main__":
    missing_files = verify_tasks()
    if missing_files:
        print("Task verification failed:")
        for msg in missing_files:
            print("- " + msg)
        exit(1)
    else:
        print("All tasks have associated recipes and documentation.")
        exit(0)
