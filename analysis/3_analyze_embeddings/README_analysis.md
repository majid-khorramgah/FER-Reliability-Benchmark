# Embedding Analysis

This directory contains the first representation-level analysis of the
FER Reliability Benchmark.

## Three measurements

### A — Representation Drift

For each expression sequence, the exact frontal embedding at V107 is the
reference.

For viewpoint v:

- cosine similarity to V107 is computed
- angular embedding distance is computed
- the full viewpoint trajectory is integrated with the trapezoidal rule

The A boundary is the first sustained (3-view) crossing of the robust
near-frontal drift threshold.

A threshold:
`13.437026` angular degrees

### B — Expression Consistency

There is NO six-class emotion classifier here.

Each complete dataset folder is treated as one controlled expression
sequence.

A near-frontal prototype is built from viewpoints within +/-5 degrees,
excluding V107.

Every test viewpoint is matched against all complete expression prototypes.

B asks:

> Does the viewpoint image still retrieve its own expression sequence?

A failure boundary is the first sustained 3-viewpoint run in which the
correct sequence is no longer top-1.

### C — Expression Separability

For each viewpoint:

`C_margin = similarity_to_own_prototype - similarity_to_best_rival`

Positive margin:
the correct expression prototype is closer.

Zero:
the correct and rival prototypes are tied.

Negative:
another expression prototype is closer.

The C boundary uses a robust lower threshold estimated from the near-frontal
region.

C also reports a hard boundary where the margin becomes negative.

## Important scientific point

The mathematical ingredients themselves are not claimed as new:

- cosine similarity
- nearest-prototype retrieval
- margin
- numerical integration
- bootstrap confidence intervals

The research contribution is the controlled combination:

1. fixed expression
2. viewpoint-only perturbation
3. continuous representation trajectory
4. prediction consistency trajectory
5. separability trajectory
6. critical viewpoint boundary
7. integrated trajectory measures
8. lead/lag analysis of representation change versus prediction failure

The most important question is whether A or C changes systematically BEFORE B
fails.

That result, rather than the choice of cosine itself, is the potential
scientific contribution.
