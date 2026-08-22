# Reproducibility Guide

This file should contain the exact environment and commands required to reproduce the results in the manuscript.

## 1. Software environment

Document:
- operating system;
- MATLAB and/or Python version;
- required toolboxes/packages and versions;
- external dependencies;
- random seeds.

## 2. Expected data layout

All scripts should use relative paths from the repository root.

## 3. Main reproduction pipeline

Document the exact commands/scripts for:
1. preprocessing;
2. feature extraction;
3. MRDS selection;
4. baseline selection;
5. downstream classification;
6. metric calculation;
7. table/figure generation.

## 4. Expected outputs

List the filenames that should appear in:
- `results/tables/`
- `results/figures/`

## 5. Verification

Provide one small smoke test that can be executed quickly and one command/script that reproduces the complete reported analysis.
