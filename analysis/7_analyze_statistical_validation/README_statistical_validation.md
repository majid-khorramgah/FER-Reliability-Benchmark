# Statistical Validation

This directory contains the statistical validation of the
A / C / B reliability analysis.

## Definitions

### A — Representation Drift

A critical viewpoint is detected when:

A_angular_distance_deg >= 13.43702602

for 3 consecutive viewpoints.

### C — Expression Separability

A critical viewpoint is detected when:

C_margin <= 0.00237080

for 3 consecutive viewpoints.

### B — Expression Consistency Failure

B is defined as prediction failure:

predicted_folder != true folder

for 3 consecutive viewpoints.

## Statistical tests

The analysis contains:

1. Bootstrap confidence intervals.
2. Paired permutation tests.
3. Paired rank-biserial effect sizes.
4. Left-vs-right comparisons.
5. Early-warning analysis.
6. Exact sign tests.
7. Viewpoint-shuffle null model.
8. Benjamini-Hochberg FDR correction.

## Important

The analysis does NOT assume:

A < C < B

The observed ordering is reported directly.

The results therefore test whether the proposed
representation-to-separability-to-prediction progression
is actually supported by the data.

## Interpretation

A statistically significant A < B relationship does NOT prove
that representation drift causes prediction failure.

It only establishes that the detected A boundary tends to occur
before the detected B boundary under the current operational
definitions.

Causal interpretation requires additional experimental design.