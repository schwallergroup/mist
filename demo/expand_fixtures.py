#!/usr/bin/env python3
"""Expand all demo fixture files to exactly 50 rows by cycling existing data.

Run from repo root:
    python demo/expand_fixtures.py
"""

import csv
import os
from pathlib import Path

TARGET_ROWS = 50
DEMO_DIR = Path(__file__).resolve().parent


def expand_csv(path: Path, target: int = TARGET_ROWS) -> None:
    """Read a CSV, cycle rows to reach target count, rewrite."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if len(rows) >= target:
        print(f"  {path.name}: already {len(rows)} rows, skipping")
        return

    original_count = len(rows)
    while len(rows) < target:
        rows.append(rows[len(rows) % original_count])

    rows = rows[:target]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {path.name}: {original_count} -> {len(rows)} rows")


def expand_txt_pairs(directory: Path, target: int = TARGET_ROWS) -> None:
    """Expand paired src/tgt text files (blank-line-separated records) to target count."""
    src_train = directory / "src-train.txt"
    tgt_train = directory / "tgt-train.txt"
    src_test = directory / "src-test.txt"
    tgt_test = directory / "tgt-test.txt"

    for src_file, tgt_file, split_target in [
        (src_train, tgt_train, 40),
        (src_test, tgt_test, 10),
    ]:
        if not src_file.exists():
            continue

        src_records = read_records(src_file)
        tgt_records = read_records(tgt_file)

        if len(src_records) >= split_target:
            print(f"  {src_file.name}: already {len(src_records)} records, skipping")
            continue

        original = len(src_records)
        while len(src_records) < split_target:
            idx = len(src_records) % original
            src_records.append(src_records[idx])
            tgt_records.append(tgt_records[idx])

        src_records = src_records[:split_target]
        tgt_records = tgt_records[:split_target]

        write_records(src_file, src_records)
        write_records(tgt_file, tgt_records)
        print(f"  {src_file.name}/{tgt_file.name}: {original} -> {len(src_records)} records")


def read_records(path: Path) -> list[str]:
    """Read multi-line records separated by blank lines."""
    text = path.read_text()
    records = []
    current = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                records.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        records.append("\n".join(current))
    return records


def write_records(path: Path, records: list[str]) -> None:
    """Write records separated by blank lines."""
    with open(path, "w") as f:
        for i, record in enumerate(records):
            f.write(record + "\n")
            if i < len(records) - 1:
                f.write("\n")


def main():
    datasets_dir = DEMO_DIR / "datasets"

    print("Expanding CSV fixtures to 50 rows:")
    for csv_file in [
        datasets_dir / "rxn_inversion_sample.csv",
        datasets_dir / "rxn_naming_sample.csv",
        datasets_dir / "rxn_truefalse_sample.csv",
    ]:
        if csv_file.exists():
            expand_csv(csv_file)

    print("\nExpanding text-file fixtures to 40 train + 10 test = 50:")
    crystalrelax_dir = DEMO_DIR / "crystalrelax_tiny"
    if crystalrelax_dir.exists():
        expand_txt_pairs(crystalrelax_dir)

    print("\nDone. All fixtures now have 50 rows.")


if __name__ == "__main__":
    main()
