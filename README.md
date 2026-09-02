# MRDS for recording-level milling-chatter data reduction

This repository contains the implementation, processed feature matrices, fixed
condition-disjoint splits, and numerical results for Meta-Rényi Distributional
Sampling (MRDS).

Each acoustic recording is represented by an empirical distribution of 40
window-level descriptors. The reduction stage compresses every training
recording to eight support atoms, constructs exact Wasserstein-2 distances, and
updates synthetic distribution-valued prototypes under a finite Rényi
objective with `alpha = 2`. A minimum-cost one-to-one Wasserstein assignment
maps the prototypes to distinct observed recordings. The selected recordings
have equal subset mass and are passed to an unweighted GaussianNB classifier
through a separate 200-dimensional recording representation.

## Repository contents

- `code/`: MRDS, the structured comparison methods, the evaluation runner, and
  the MATLAB feature-extraction source.
- `data/processed_feature_matrices/`: the 160 processed MAT files used as
  direct computational inputs.
- `reproducibility/`: recording metadata, descriptor names, fixed folds, and
  the numerical configuration.
- `results/primary_10pct/`: the primary 10% comparison, including Random.
- `results/structured_retention_10_50/`: MRDS, Wasserstein k-medoids, and
  Facility Location across 10%, 20%, 30%, 40%, and 50% retention.
- `docs/`: data provenance, reproduction instructions, and validation notes.
- `FILE_MAP.md`: a file-by-file guide.
- `PACKAGE_MANIFEST.csv` and `SHA256SUMS.txt`: file sizes and SHA-256 digests.

The manuscript and its LaTeX sources are not distributed in this repository.

## Installation

Python 3.11 or a compatible later version is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Reproduction command

The full computation is expensive because it repeatedly solves exact optimal
transport problems.

```bash
python code/run_mrds_evaluation.py
```

Use `--splits` to run selected outer folds and `--output` to choose a result
directory. See `docs/REPRODUCIBILITY.md` for the complete configuration.

## Data scope

The MAT files contain processed window-descriptor matrices. The original WAV
recordings are not included. See `data/README_DATA.md` and
`docs/DATA_AND_PROVENANCE.md`.

## Use and citation

The authors retain copyright in this review release. See `USAGE_NOTICE.md` and
`CITATION.cff`.
