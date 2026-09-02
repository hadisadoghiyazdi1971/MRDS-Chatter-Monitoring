# Processed feature matrices

This directory contains 160 MAT files, one for each complete acoustic
recording. Each file contains a `SigData` structure array. Every structure
element corresponds to one analysis window and stores the 40 descriptors
listed in `reproducibility/feature_fields.json`.

The number of windows is recorded in
`reproducibility/dataset_manifest_grouped.csv`. There are 40 machining
conditions and four recordings per condition. The manifest also provides the
SHA-256 digest of every MAT file.

These files are processed feature matrices. The original acoustic WAV files
are not included.
