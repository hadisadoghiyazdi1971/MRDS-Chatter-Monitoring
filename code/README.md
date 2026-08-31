# Code scope

`run_simplified_retention_curve.py` is the frozen runner for the final
projection-only MRDS study. Its MRDS execution path calls
`synthetic_projection_only` from `candidate_synthetic_core.py`, which performs
objective-stopped synthetic Gauss--Seidel updates followed by duplicate-free
one-to-one projection.

The final configuration uses `alpha=2`, eight support atoms per recording,
and the synthetic stopping rule: normalized objective reduction below `1e-3`
for three consecutive complete sweeps, with a safety cap of 30 sweeps. The
selected subset has equal mass; GaussianNB receives selected recordings only,
without mixture/sample weights.

`meta_renyi_reduction.py` and
`mrds_projection_refinement_integrated.py` are retained because they are direct
imports of the frozen runner/core snapshot. The final runner does not execute
post-projection mixture-weight optimization or one-swap refinement. They are
not separate reported methods in this repository.

The runner depends on the original raw acoustic/MAT archive, which is excluded
from this public snapshot. It is provided for transparent code review rather
than as a one-command data download.
