# Implemented selection path

For each outer training fold, preprocessing is fitted using training
recordings only. Every recording is represented by an empirical distribution
of 40-dimensional window descriptors and compressed to eight support atoms by
seeded K-means with five initializations and at most 200 iterations.

The implementation computes exact pairwise Wasserstein-2 distances with earth
mover's distance and applies the median positive distance as the kernel
bandwidth. MRDS initializes synthetic distribution-valued prototypes by
farthest-first traversal and updates them in Gauss-Seidel order under the
finite Rényi objective with `alpha = 2`.

The synthetic stage stops after three consecutive complete sweeps whose
objective decrease, normalized by the initial synthetic objective, is below
`1e-3`. The safety cap is 30 sweeps. A cap hit without satisfying the stopping
rule is reported as unresolved.

The final observed subset is obtained by a minimum-cost one-to-one Wasserstein
assignment from prototypes to training recordings. Its outer masses are equal.
No post-projection weight or membership update is applied.

Wasserstein k-medoids and Facility Location use the same training-side
distance or kernel construction. Random is evaluated only at 10% retention.
Selected recordings are represented by the mean, standard deviation, and
three quartiles of each descriptor, producing 200 inputs for an unweighted
GaussianNB classifier.
