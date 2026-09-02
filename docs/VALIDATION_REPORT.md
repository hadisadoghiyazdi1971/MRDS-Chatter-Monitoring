# Package validation report

- The 160 processed MAT files match the SHA-256 values in the data manifest.
- The 20 fixed splits each contain 120 training and 40 test recordings, with
  no machining condition shared between the partitions.
- The Python sources pass static syntax and local-import checks.
- The implemented path uses `alpha = 2`, eight support atoms, exact EMD,
  objective-stopped synthetic updates, duplicate-free one-to-one projection,
  equal final subset mass, and unweighted GaussianNB.
- Structured results contain 20 folds per method at each retention level from
  10% to 50%. Random results are included at 10% only.
- The primary and retention tables match the final manuscript values.
- Contents are limited to the reported primary and retention analyses; no
  development experiment, manuscript file, LaTeX source, PDF, or absolute
  local path is included.
