#!/usr/bin/env python3
"""Download and set up MiST datasets from Figshare.

Usage:
    python scripts/setup_data.py [--data-dir DIR] [--skip-models]

This script:
  1. Downloads datasets.zip (and optionally models.zip) from Figshare
  2. Extracts them into the target directory
  3. Creates the directory layout expected by ${MIST_DATA_DIR}
  4. Writes a .env.local file for convenience

After running, set MIST_DATA_DIR to the data directory, or source .env.local.
"""

import argparse
import hashlib
import os
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

FIGSHARE_FILES = {
    "datasets": {
        "url": "https://ndownloader.figshare.com/files/55028555",
        "filename": "datasets.zip",
        "size": 2359373145,
        "md5": "774bae0908abb81a419adaa0cce83b2d",
    },
    "models": {
        "url": "https://ndownloader.figshare.com/files/55032923",
        "filename": "models.zip",
        "size": 5419112931,
        "md5": "6b8702e8a52a3b30463a4101dfff28b5",
    },
}

# Mapping from Figshare archive paths to the MIST_DATA_DIR layout expected by recipes
DATASET_MAP = {
    # RxP
    "datasets/rl/ReactionPrediction": "rxnpred/USPTO_480k_clean",
    # I2S
    "datasets/rl/iupac2smiles/CRLLM-PubChem-compounds1M.csv": "CRLLM-PubChem-compounds1M.csv",
    "datasets/rl/iupac2smiles/CRLLM-PubChem-compounds1M-very_very_simple.csv": "CRLLM-PubChem-compounds1M-very_very_simple.csv",
    "datasets/rl/iupac2smiles/CRLLM-PubChem-compounds1M_hydrogen.csv": "CRLLM-PubChem-compounds1M_hydrogen.csv",
    "datasets/rl/iupac2smiles/CRLLM-PubChem-compounds1M.no_sft.csv": "CRLLM-PubChem-compounds1M.no_sft.csv",
    "datasets/rl/iupac2smiles/CRLLM-PubChem-compounds_001000001_002000000.csv": "CRLLM-PubChem-compounds_001000001_002000000.csv",
    "datasets/rl/iupac2smiles/CRLLM-PubChem-compounds_002000001_003000000.csv": "CRLLM-PubChem-compounds_002000001_003000000.csv",
    "datasets/rl/iupac2smiles/CRLLM-PubChem-compounds_003000001_004000000.csv": "CRLLM-PubChem-compounds_003000001_004000000.csv",
    # RxN
    "datasets/rl/ReactionName/reaction_class_prompts_600k.csv": "rxn_naming/reaction_class_prompts_600k.csv",
    # RxR
    "datasets/rl/ReactionReplacement/mcqa_modified_reactions_1M_prompts.csv": "rxn_replacement/mcqa_modified_reactions_1M_prompts.csv",
    # RxI
    "datasets/rl/ReactionInversion/dataset_swapped500k_prompt.csv": "rxn_inversion/dataset_swapped500k_prompt.csv",
    # RxTF
    "datasets/rl/RxnTrueFalse/is_it_correct_reaction_1M.csv": "rxn_truefalse/is_it_correct_reaction.csv",
    # CMG
    "datasets/rl/CondMatGen": "condmatgen",
    # CrR
    "datasets/rl/binary_compound_relax": "binary_compound_relaxing",
    # MiST pretraining data
    "datasets/mist": "mist_pretrain",
    # Diagnostic
    "datasets/diagnostic": "diagnostic",
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, expected_size: int, label: str) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  {label}: already downloaded ({expected_size / 1e9:.1f} GB)")
        return

    print(f"  {label}: downloading ({expected_size / 1e9:.1f} GB)...", flush=True)

    # Create an SSL context that doesn't verify certificates (some systems lack root certs)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    downloaded = 0
    with urllib.request.urlopen(req, context=ctx) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / expected_size * 100
            print(
                f"\r  {label}: {downloaded / 1e9:.2f} / {expected_size / 1e9:.2f} GB ({pct:.0f}%)", end="", flush=True
            )
    print()


def verify_md5(path: Path, expected: str, label: str) -> bool:
    print(f"  {label}: verifying MD5...", end="", flush=True)
    actual = md5sum(path)
    if actual == expected:
        print(" OK")
        return True
    else:
        print(f" MISMATCH (expected {expected}, got {actual})")
        return False


def extract_and_map(zip_path: Path, data_dir: Path) -> None:
    print(f"  Extracting {zip_path.name}...", flush=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract everything to a temp location first
        tmp_dir = data_dir / "_figshare_extract"
        zf.extractall(tmp_dir)

    # Now create the mapped directory structure
    for src_pattern, dest_rel in DATASET_MAP.items():
        src = tmp_dir / src_pattern
        dest = data_dir / dest_rel

        if not src.exists():
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            print(f"    {dest_rel}/ (directory)")
        else:
            shutil.copy2(src, dest)
            print(f"    {dest_rel}")

    # Clean up temp extraction
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  Done.")


def write_env_local(repo_root: Path, data_dir: Path, models_dir: Path | None) -> None:
    env_path = repo_root / ".env.local"
    lines = [
        f'MIST_DATA_DIR="{data_dir}"\n',
    ]
    if models_dir:
        lines.append(f'MIST_MODELS_DIR="{models_dir}"\n')

    env_path.write_text("".join(lines))
    print(f"\nWrote {env_path}")
    print(f"  Source it with: source .env.local && export MIST_DATA_DIR MIST_MODELS_DIR")


def main():
    parser = argparse.ArgumentParser(description="Download and set up MiST data from Figshare.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Target directory for datasets. Defaults to ./data/",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Directory to store downloaded zip files. Defaults to data-dir.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip downloading models.zip (5 GB).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip MD5 verification.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).resolve() if args.data_dir else repo_root / "data"
    download_dir = Path(args.download_dir).resolve() if args.download_dir else data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Download directory: {download_dir}")
    print()

    # Download datasets
    ds = FIGSHARE_FILES["datasets"]
    ds_zip = download_dir / ds["filename"]
    download_file(ds["url"], ds_zip, ds["size"], "datasets.zip")
    if not args.skip_verify:
        verify_md5(ds_zip, ds["md5"], "datasets.zip")

    # Extract and map datasets
    extract_and_map(ds_zip, data_dir)

    # Optionally download models
    models_dir = None
    if not args.skip_models:
        ms = FIGSHARE_FILES["models"]
        ms_zip = download_dir / ms["filename"]
        download_file(ms["url"], ms_zip, ms["size"], "models.zip")
        if not args.skip_verify:
            verify_md5(ms_zip, ms["md5"], "models.zip")

        models_dir = data_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting models.zip...")
        with zipfile.ZipFile(ms_zip, "r") as zf:
            zf.extractall(models_dir)
        print("  Done.")

    # Write .env.local
    write_env_local(repo_root, data_dir, models_dir)

    print("\nSetup complete. To use:")
    print(f"  export MIST_DATA_DIR={data_dir}")
    if models_dir:
        print(f"  export MIST_MODELS_DIR={models_dir}")
    print()
    print("Or source .env.local:")
    print("  source .env.local && export MIST_DATA_DIR MIST_MODELS_DIR")


if __name__ == "__main__":
    main()
