# Reproducibility Notes

## Direct computational inputs
The Python evaluation pipeline uses the 160 MAT archives in `chatterData/`.

## Raw-signal limitation
The complete original WAV corpus is not present in the archived research package. The supplied MATLAB feature-extraction script documents the extraction procedure, but the public package cannot recreate every MAT archive from raw WAV files end-to-end without the missing WAV corpus.

## Primary grouped evaluation
The current runner defaults to five seeds (`11, 23, 37, 53, 71`), four folds per seed, 10% retention, 40 descriptors, alpha=2, K=8 support atoms, one synthetic iteration, ten weight-optimization iterations, and one swap-refinement pass.

## Frozen retention evaluation
The 10–50% budget runner reuses `results/grouped_chatter_all40/folds_grouped.csv`.

## Downstream weighting
The reported GaussianNB results use uniform downstream training; optimized Rényi mixture weights are not passed to GaussianNB.

## Provenance limitations
1. The exact byte-identical historical snapshot of `run_chatter_grouped.py` used for the first 10% output is not preserved.
2. The generator for `grouped_chatter_primary.tex` is not preserved, although its numbers match the archived raw result files.
3. Editable sources are incomplete for some manuscript figures; final exported assets are retained.
4. `run_chatter_grouped.py` also writes a per-fold `refinement_attribution.csv`. That file is
   not part of the archived production snapshot and is therefore not shipped here; rerunning the
   grouped experiment regenerates it.
