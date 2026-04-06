# Demo Fixtures

This directory contains lightweight data fixtures for reviewer-friendly smoke
tests.

The CSV fixtures under `datasets/` were extracted from the published Figshare
`datasets.zip` bundle for article `29132657`. The committed demo files
now contain 50 rows each, built from that extracted slice so they stay small
enough for the repository while remaining faithful to the real dataset schema.
The fixture-to-task mapping is recorded in `fixture_manifest.csv`.

## Included fixtures

- `rxnpred_tiny/`
  - 40 training examples and 10 test examples in the text format expected by
    the reaction-prediction loader
  - also used by `recipes/rxnpred.smoke.yaml` for the GRPO smoke run
- `datasets/CRLLM-PubChem-compounds1M.sample.csv`
  - 50-row fixture based on `datasets/rl/iupac2smiles/CRLLM-PubChem-compounds1M.csv`
- `datasets/CRLLM-PubChem-compounds1M_hydrogen.sample.csv`
  - 50-row fixture based on `datasets/rl/iupac2smiles/CRLLM-PubChem-compounds1M_hydrogen.csv`
- `datasets/CRLLM-PubChem-compounds1M-very_very_simple.sample.csv`
  - 50-row fixture based on `datasets/rl/iupac2smiles/CRLLM-PubChem-compounds1M-very_very_simple.csv`
- `datasets/CRLLM-PubChem-compounds1M-simple.sample.csv`
  - 50-row convenience alias derived from the `compounds1M` fixture for
    loaders that only require the `IUPAC` and `SMILES` columns

## Not found in the current Figshare `datasets.zip`

- `rxnpred/USPTO_480k_clean` text splits
- kinetic pickle bundle expected by `recipes/kinetic.yaml`

For those two cases, the repository provides:

- `rxnpred_tiny/` as a minimal local reaction fixture
- `make_kinetic_tiny.py` to generate a tiny synthetic kinetic bundle in the
  expected on-disk format with 40 training examples and 10 validation examples

## Smoke tests

Run the original reaction-prediction demo:

```bash
PYTHONPATH=src python demo/run_demo.py
```

Run the multi-task loader smoke test:

```bash
PYTHONPATH=src python demo/run_fixture_smoke.py
```

## RL smoke run

The repository includes a minimal GRPO smoke recipe for reaction prediction:

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

This smoke run uses the bundled 50-example `rxnpred_tiny/` fixture and is
designed for a single GPU without vLLM.
