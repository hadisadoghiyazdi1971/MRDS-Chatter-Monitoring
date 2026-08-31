# Simplified MRDS projection-only retention curve

The evaluated MRDS candidate consists only of objective-stopped synthetic
meta-Renyi optimization followed by duplicate-free projection to complete
observed recordings. Mixture-weight optimization and one-swap refinement are
not part of this candidate.

The candidate is evaluated at 10%, 20%, 30%, 40%, and 50% retention on the
frozen 20 condition-disjoint outer folds. All three reduced methods use uniform
observed-subset weights for the finite objective. Downstream evaluation uses
the prespecified standard unweighted GaussianNB and the training-fitted 200-D
recording representation. No inner validation, classifier tuning, or
significance testing is performed.

The validated 10% projected subsets, objectives, and classifier results are
reused only after exact fold-hash, subset-ID, objective-reconstruction, and
protocol checks. The archived 20--50% `-R` curve is not reused because its
selected-recording file does not separately certify initializer-stage subsets;
W2-k-medoids and Facility Location initializers are therefore recomputed.

The runner is resumable at every completed budget x fold x method checkpoint.
A binding synthetic cap is recorded as `SYNTHETIC_CAP_BINDING`; its trace is
preserved and the numerical policy is not changed automatically.
