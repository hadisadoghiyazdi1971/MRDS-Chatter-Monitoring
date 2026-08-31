# Package validation

Validated on 2026-08-31.

## Checks passed

- The canonical manuscript is `manuscript/Manuscript (1).pdf`, a 34-page A4
  PDF copied byte-for-byte from the final `Manuscript (1).pdf` review copy.
- The included LaTeX source bundle compiled successfully with `latexmk -pdf`
  into a 34-page A4 PDF. Its build products were written outside this
  repository and are not packaged.
- All four included Python source files passed `python -m py_compile`.
- Table 4's MRDS, Wasserstein k-medoids, Facility Location, and Random means
  were recomputed from the included 20-fold primary-budget CSV files and agree
  with `table_primary_10pct.csv` to floating-point precision.
- The primary Random CSV files contain 20 rows each, all at 10% retention.
- The structured retention CSV files contain only MRDS, Wasserstein k-medoids,
  Facility Location, and the full-training reference; they contain no Random
  rows.
- No raw audio/MAT archive, cache, Python bytecode, LaTeX build product,
  archive-within-archive, credentials, or editor/repository metadata is
  included.

## Intentional limits

The raw acoustic archive is not supplied, so the package supports code and
result auditing but not a complete rerun without authorized data access. The
canonical manuscript PDF is the reference presentation; its source bundle is
included for editorial traceability and independent compilation.
