# Multi-Metric Trajectory Analysis

This run evaluates the same controlled viewpoint experiment with several
representation geometries.

## A metrics

- Cosine distance
- Angular distance
- Euclidean distance on normalized embeddings
- Cumulative path length from V107
- Local representation-change rate
- Discrete curvature
- Second-difference trajectory instability

The default A threshold for each metric is its 95th percentile in the
near-frontal |angle| <= 5 degree baseline.

A boundary requires 3 consecutive viewpoints above threshold.

## B

B is prototype retrieval consistency, not a six-class emotion classifier.

## C

C is:

    own-prototype similarity - strongest rival similarity

## Scientific purpose

The individual metrics are not claimed to be novel. The purpose is to test
whether the same representation-instability-before-prediction-failure
phenomenon is robust to the choice of representation geometry.

A finding that appears only for cosine would be metric-dependent.
Agreement across substantially different metrics would be stronger evidence.
