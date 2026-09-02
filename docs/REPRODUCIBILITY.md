# Reproducibility

## Inputs

- 160 processed MAT feature files in `data/processed_feature_matrices/`.
- Twenty fixed splits in `reproducibility/folds_grouped.csv`.
- The numerical settings in `reproducibility/frozen_config.json`.

Each split contains 120 outer-training recordings and 40 held-out recordings.
Machining conditions do not overlap between the two partitions.

## Environment

Create a Python environment and install `requirements.txt`. Python Optimal
Transport (POT) is required for exact EMD.

## Run

From the repository root:

```bash
python code/run_mrds_evaluation.py
```

The default output directory is `outputs/reproduction/`. A selected set of
outer folds can be requested with, for example:

```bash
python code/run_mrds_evaluation.py --splits 0 1 2
```

The full computation evaluates MRDS, Wasserstein k-medoids, and Facility
Location at 10%, 20%, 30%, 40%, and 50% retention. Random is evaluated at the
primary 10% setting only. GaussianNB is fitted without sample weights.

The supplied `results/` directory contains the numerical outputs retained for
the reported study.

The supplied fold-wise results can be summarized without rerunning selection:

```bash
python code/build_summary_tables.py
```
