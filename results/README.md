# Results scope

`primary_10pct/` contains the final primary comparison at 10% retention
(`m=12` of 120 outer-training recordings). It includes fold-wise outputs for
MRDS, Wasserstein k-medoids, Facility Location, the full-training reference,
and the predefined-seed Random baseline.

The two `random_*_10pct.csv` files are deliberately filtered to the 20 frozen
outer splits at 10% retention. No Random output for any higher retention budget
is included.

`structured_retention_10_50/` contains only MRDS, Wasserstein k-medoids, and
Facility Location outputs for the 10--50% grid reported in the final
manuscript. It contains no Random rows. The accompanying source files support
Table 5 and Figure 6; the corresponding manuscript assets reside in
`manuscript/source/`.
