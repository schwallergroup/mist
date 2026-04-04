# Demo Fixtures

This directory contains lightweight data fixtures for reviewer-friendly smoke
tests.

The CSV fixtures under `datasets/` were extracted on GCP from the published
Figshare `datasets.zip` bundle for article `29132657`. The committed demo files
now contain 50 rows each, built from that extracted slice so they stay small
enough for the repository while remaining faithful to the real dataset schema.

## Included fixtures

- `rxnpred_tiny/`
  - 40 training examples and 10 test examples in the text format expected by
    the reaction-prediction loader
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
