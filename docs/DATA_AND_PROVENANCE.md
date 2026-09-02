# Data and provenance

The direct Python inputs are the 160 processed MAT files distributed in
`data/processed_feature_matrices/`. Their names and SHA-256 digests match
`reproducibility/dataset_manifest_grouped.csv`.

The folds file defines five seeded repetitions of four condition-disjoint
outer folds. Training-side preprocessing, support compression, Wasserstein
geometry, bandwidth selection, and subset selection are repeated within each
outer training partition.

The saved results are divided by scope:

- `results/primary_10pct/` contains the primary comparison and is the only
  location containing Random results.
- `results/structured_retention_10_50/` contains the three structured
  selectors across the complete retention grid.

The repository does not include acoustic WAV files, manuscript files, or
unreported development experiments.

The Python release files consolidate the final execution path into standalone
modules. The supplied numerical outputs are unchanged from the final reported
runs.
