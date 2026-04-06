# Release Scope

This folder tracks the planned public release contents for MiST.

## Planned for GitHub release

- RL / post-training code in this repository
- Task definitions and reward functions
- Training and evaluation recipes
- Sampling parameter configs
- Cluster launcher examples for CSCS and Kuma
- Dataset manifests and preprocessing metadata
- Lightweight demo dataset and walkthrough
- Release planning files for submission and Figshare packaging

## Planned model release

- Selected MiST checkpoints and intermediate snapshots

## Planned for Figshare release

- Release bundles for task datasets and derived splits
- Mid-training data manifests and preprocessing scripts
- Selected MiST model snapshots
- Supplementary release metadata describing dataset provenance and split rules

## Not planned for public release here

- Task-specific RL checkpoints produced for individual experiments

## Related assets to release separately

- Mid-training code and configs
- Mid-training data manifests and preprocessing scripts

Use `release/dataset_components.csv` as the working index for all datasets and
derived components referenced by the MiST codebase, and
`release/figshare_upload_manifest.csv` as the upload-facing inventory.
