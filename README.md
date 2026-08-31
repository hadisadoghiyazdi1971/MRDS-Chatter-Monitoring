# MRDS: recording-level reduction for milling-chatter monitoring

This repository snapshot accompanies the final manuscript, **“Meta-Rényi
Distributional Sampling for Recording-Level Data Reduction in Milling Chatter
Detection.”** Open [manuscript/Manuscript (1).pdf](manuscript/Manuscript%20(1).pdf)
first.

## Final method represented here

The final MRDS pipeline is projection-only after synthetic optimization:

1. represent each complete acoustic recording as a compressed empirical
   distribution of window-level descriptors;
2. optimize synthetic distribution-valued prototypes under the finite
   Meta-Rényi objective using Wasserstein geometry;
3. recover distinct complete observed recordings through a minimum-cost
   one-to-one Wasserstein assignment; and
4. use the resulting equal-mass subset for standard, unweighted GaussianNB
   evaluation.

Post-projection mixture-weight optimization and one-swap subset refinement
are **not** part of the final reported MRDS method.

## What is included

- `manuscript/`: the final 34-page manuscript PDF and a self-contained LaTeX
  source bundle.
- `code/`: the frozen projection-only evaluation runner, its required support
  modules, dependency list, and the MATLAB feature-extraction source.
- `results/primary_10pct/`: Table 4 values and fold-wise primary-budget
  results. The Random baseline is provided **only** for this 10% comparison.
- `results/structured_retention_10_50/`: MRDS, Wasserstein k-medoids, and
  Facility Location results supporting the retention curve reported in the
  manuscript. It contains no Random-baseline rows.
- `reproducibility/`: frozen condition-disjoint split metadata and feature
  field definitions.

## Deliberate scope exclusions

No Random-baseline results for 20%, 30%, 40%, or 50% retention are included.
No exploratory, sensitivity, stopping-policy, legacy refinement, cache, raw
audio, or MAT-archive output is included. Raw acquisition data are not
redistributed in this repository.

The code is supplied as a frozen audit/reproducibility snapshot. A complete
rerun requires the original raw archive and local data-access authorization;
those data are intentionally absent. The final execution path in
`run_simplified_retention_curve.py` invokes
`synthetic_projection_only` and duplicate-free projection, not the legacy
post-projection refinement routines retained in a support module for exact
snapshot compatibility.

## Manuscript build

From `manuscript/source/`:

```powershell
latexmk -pdf MRDS_Meccanica.tex
```

The shipped `manuscript/Manuscript (1).pdf` is the canonical review copy.

## License

No open-source license has been selected for this snapshot. Do not assume
permission to reuse the code, figures, manuscript, or data-related metadata
until the authors choose and add a license.
