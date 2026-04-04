import pickle
from pathlib import Path

import numpy as np


TRAIN_SIZE = 40
VAL_SIZE = 10
TIME_STEPS = 4
CHANNELS = 12


def build_x1(num_examples: int, offset: float) -> np.ndarray:
    rows = []
    for i in range(num_examples):
        base = offset + i * 0.01
        rows.append([base + 0.10, base + 0.20, base + 0.30, base + 0.40])
    return np.array(rows, dtype=float)


def build_x2(num_examples: int, offset: float) -> np.ndarray:
    tensor = np.zeros((num_examples, TIME_STEPS, CHANNELS), dtype=float)
    for i in range(num_examples):
        scale = offset + i * 0.02
        for t in range(TIME_STEPS):
            time_value = round(t / (TIME_STEPS - 1), 3)
            tensor[i, t, 0] = time_value
            tensor[i, t, 3] = time_value
            tensor[i, t, 6] = time_value
            tensor[i, t, 9] = time_value

            for run in range(4):
                substrate_col = run * 3 + 1
                product_col = run * 3 + 2
                start = max(0.15, 1.0 - 0.05 * run - 0.02 * i)
                decay = min(0.75, 0.16 * t + 0.015 * i + 0.02 * run + scale)
                substrate = max(0.0, round(start - decay, 3))
                product = min(1.0, round(1.0 - substrate, 3))
                tensor[i, t, substrate_col] = substrate
                tensor[i, t, product_col] = product
    return tensor


def build_y(num_examples: int, start_class: int) -> np.ndarray:
    return np.array(
        [[(start_class + i) % 20] for i in range(num_examples)],
        dtype=int,
    )


def build_kinetic_files():
    return {
        "x1_train_M1_M20_train_val_test_set_part_0.pkl": build_x1(
            TRAIN_SIZE, 0.00
        ),
        "x2_train_M1_M20_train_val_test_set_part_0.pkl": build_x2(
            TRAIN_SIZE, 0.00
        ),
        "y_train_M1_M20_train_val_test_set_part_0.pkl": build_y(
            TRAIN_SIZE, 0
        ),
        "x1_val_M1_M20_train_val_test_set_part_0.pkl": build_x1(
            VAL_SIZE, 0.40
        ),
        "x2_val_M1_M20_train_val_test_set_part_0.pkl": build_x2(
            VAL_SIZE, 0.10
        ),
        "y_val_M1_M20_train_val_test_set_part_0.pkl": build_y(VAL_SIZE, 4),
    }


def ensure_kinetic_tiny(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, array in build_kinetic_files().items():
        path = output_dir / filename
        with path.open("wb") as handle:
            pickle.dump(array, handle)
    return output_dir


if __name__ == "__main__":
    ensure_kinetic_tiny(Path(__file__).resolve().parent / "kinetic_tiny")
