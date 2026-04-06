# Reviewer Run Guide

This file summarizes the run paths that are currently practical for reviewers
and editors.

## 1. Local smoke tests

These require only the repository, the bundled fixtures under `demo/`, and the
Python dependencies listed in `README.md`.

```bash
PYTHONPATH=src python demo/run_demo.py
PYTHONPATH=src python demo/run_fixture_smoke.py
```

These commands validate:

- reaction-prediction fixture loading
- reward computation on a tiny bundled example
- multi-task dataset loading for the task families covered by the local 50-row
  slices

## 2. SCS metric

The SCS implementation lives in `src/open_r1/diagnostic/smiles_competence.py`.
It requires `vllm` and a model that can be loaded by vLLM.

Install the full project dependencies before running SCS:

```bash
pip install -e .
```

### Small reviewer smoke run

Use the 50-row bundled fixture to verify that the diagnostic pipeline runs end
to end without downloading the full PubChem release:

```bash
PYTHONPATH=src python src/open_r1/diagnostic/smiles_competence.py \
  --model /path/to/model \
  --data-path demo/datasets/CRLLM-PubChem-compounds1M.sample.csv \
  --output-dir output/scs-smoke \
  --num-rows 50 \
  --tensor-parallel-size 1
```

Expected outputs:

- `output/scs-smoke/lps_canonical.csv`
- `output/scs-smoke/lps_random.csv`
- `output/scs-smoke/lps_corrupt.csv`
- `output/scs-smoke/summary.json`

### Full diagnostic run

When the full dataset bundle is available locally, run:

```bash
PYTHONPATH=src python src/open_r1/diagnostic/smiles_competence.py \
  --model /path/to/model \
  --data-path "${MIST_DATA_DIR}/CRLLM-PubChem-compounds1M.csv" \
  --output-dir "${MIST_OUTPUT_DIR}/scs/full" \
  --num-rows 10000 \
  --tensor-parallel-size 1
```

For a multi-GPU cluster run, use `launch_diagnostics.slurm`.

### Verified reference run

The SCS path above was validated on April 4, 2026 on a GCP `a2-highgpu-1g`
instance using `Qwen/Qwen2.5-3B`, `10000` rows from
`CRLLM-PubChem-compounds1M.csv`, `tensor_parallel_size=1`, and the current
`smiles_competence.py` batching implementation.

Observed outputs:

- `SCS logprobs = 0.9553193281550367`
- `SCS rank = -0.07295051975114555`

The run wrote `lps_canonical.csv`, `lps_random.csv`, `lps_corrupt.csv`, and
`summary.json` under `/var/log/mist-scs` on the evaluation VM.

## 3. GRPO training

This repository contains the task implementations, recipes, and SLURM launchers
used for the MiST GRPO experiments.

### Single-GPU smoke run

The quickest end-to-end GRPO check uses the bundled `rxnpred` fixture with a
single compatible base model checkpoint:

```bash
accelerate launch --config_file configs/smoke_single_gpu.yaml \
  src/open_r1/run_r1_grpo.py \
  --config recipes/rxnpred.smoke.yaml \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --output_dir output/rxnpred-smoke \
  --run_name rxnpred-smoke-qwen25-3b \
  --base_model_name Qwen/Qwen2.5-3B \
  --base_model_id Qwen/Qwen2.5-3B
```

This path uses `demo/rxnpred_tiny/`, which contains 50 total examples
(`40 train / 10 test`), and it does not require a MiST checkpoint.

Example cluster launch:

```bash
sbatch launch_CSCS.slurm Qwen2.5-3B rxnpred
```

Full reproduction of the GRPO results in the manuscript requires:

- the released task datasets and derived splits
- a supported Linux multi-GPU environment
- model paths or checkpoints compatible with `model_paths.txt`

## 4. Mid-training / pretraining

This repository is not yet a fully self-contained mid-training release. The
current release plan tracks the missing pieces under:

- `release/figshare_upload_manifest.csv`
- `release/dataset_components.csv`
- `release/submission_package_checklist.md`

In particular, the public release still needs the preprocessing scripts,
manifests, and split definitions for the MiST mid-training mixtures.
