# Submission Package Checklist

This checklist maps the Nature software submission form to concrete repository
files and release artifacts.

## Already covered in this repository

- Source code:
  - `src/`
  - `recipes/`
  - `sampling_params/`
  - cluster launchers under the repository root and `cluster/`
- Small demo dataset:
  - `demo/rxnpred_tiny/`
  - `demo/datasets/`
  - `demo/make_kinetic_tiny.py`
- README with system requirements, installation, demo, and reproduction notes:
  - `README.md`
- Reviewer-facing run paths for smoke tests, SCS, and cluster GRPO launches:
  - `release/reviewer_runs.md`
- Open-source repository metadata:
  - `setup.py`
  - `LICENSE`
  - `PROVENANCE.md`
- Pseudocode / algorithmic description in the manuscript:
  - Appendix / supplementary experimental details
  - `preprint/main-new-st.tex`
  - `ICLR2026/iclr2026_conference.tex`

## Must be exported before submission / public release

- Task datasets and derived splits referenced by the GRPO recipes:
  - RxP / USPTO-480K cleaned split
  - PubChem-derived GRPO datasets
  - kinetic mechanism dataset
- Additional task datasets described in the manuscript appendix but not present
  in this repository snapshot:
  - RxN
  - RxR
  - RxI
  - RxTF
  - CeB
  - CMG
  - CrR
- MiST mid-training artifacts:
  - preprocessing scripts
  - data manifests
  - split definitions
  - any releasable derived datasets
- Selected MiST model snapshots intended for release

## Explicitly not planned for release

- Task-specific RL checkpoints produced for individual experiments
- Secrets or local credentials:
  - `wandb_api_key.txt`
  - `.env.local`
  - cluster-local environment files

## Figshare planning notes

- Use `release/figshare_upload_manifest.csv` as the authoritative upload plan.
- For datasets with restrictive upstream licensing or impractical sizes,
  release the preprocessing scripts, manifests, and split definitions rather
  than mirrored raw source dumps.
- For lightweight reproducibility checks on cloud infrastructure, use:
  - `release/gcp_demo_startup.sh`
  - `release/gcp_create_demo_vm.sh`
