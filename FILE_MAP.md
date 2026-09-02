# File map

## Code

| Path | Purpose |
|---|---|
| `code/mrds_core.py` | Empirical measures, support compression, exact EMD, finite Rényi quantities, Wasserstein k-medoids, and Facility Location. |
| `code/synthetic_projection.py` | Synthetic Gauss-Seidel MRDS updates, objective-based stopping, and duplicate-free one-to-one projection. |
| `code/run_mrds_evaluation.py` | Condition-disjoint preprocessing, selection, unweighted GaussianNB evaluation, and result export. |
| `code/build_summary_tables.py` | Primary and retention tables from the supplied fold-wise outputs. |
| `code/extract_recording_features_40d.m` | MATLAB source used to extract the 40 window-level descriptors. |

## Data and reproducibility

| Path | Purpose |
|---|---|
| `data/processed_feature_matrices/*.mat` | 160 processed recording feature matrices used by the Python evaluation. |
| `data/README_DATA.md` | MAT structure and data scope. |
| `reproducibility/dataset_manifest_grouped.csv` | Recording identifiers, labels, conditions, window counts, and SHA-256 digests. |
| `reproducibility/feature_fields.json` | Names of the 40 window descriptors. |
| `reproducibility/folds_grouped.csv` | Twenty fixed condition-disjoint outer splits. |
| `reproducibility/frozen_config.json` | Numerical and classifier settings. |

## Results

| Path | Purpose |
|---|---|
| `results/primary_10pct/` | Fold-wise objectives and classification metrics for the primary 10% comparison. Random appears only here. |
| `results/structured_retention_10_50/` | Fold-wise and aggregate results for the three structured selectors across the reported retention grid. |

## Documentation and integrity

| Path | Purpose |
|---|---|
| `docs/METHOD.md` | Concise description of the implemented selection path. |
| `docs/REPRODUCIBILITY.md` | Environment, inputs, command, and output description. |
| `docs/DATA_AND_PROVENANCE.md` | Relationship between processed data, splits, and saved results. |
| `docs/VALIDATION_REPORT.md` | Static code and package-integrity checks. |
| `PACKAGE_MANIFEST.csv` | Relative path, category, byte size, and SHA-256 digest for every payload file. |
| `SHA256SUMS.txt` | SHA-256 digest list for command-line verification. |
