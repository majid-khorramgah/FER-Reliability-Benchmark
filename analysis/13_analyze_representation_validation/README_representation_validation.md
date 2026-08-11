# Stage 9 — Representation-Level Validation

## Purpose

This stage tests whether viewpoint-induced representation instability
appears before the already-validated prediction-failure boundary.

## Fixed configuration

- A threshold: 13.43702602
- C threshold: 0.00237080
- Sustained viewpoints: 3
- Frontal viewpoint: V107
- Expected viewpoints per complete sequence: 215
- Near-frontal prototype window: ±5 degrees
- Embedding dimension: 768

## Viewpoint convention

The benchmark uses V107 as frontal.

Signed angle is reconstructed deterministically as:

    angle = viewpoint - 107

Therefore:

- V000 = -107 degrees
- V106 = -1 degree
- V107 = 0 degrees
- V108 = +1 degree
- V214 = +107 degrees

Left/right traversal is outward from V107.

## Representation measures

`R_frontal_angular_deg`
: angular distance between the current representation and the exact
  frontal V107 representation of the same sequence.

`R_expression_margin`
: similarity to the sequence's own near-frontal expression prototype
  minus similarity to the strongest rival expression prototype.

`R_retrieval_correct`
: whether the own-expression prototype is the nearest prototype.

## Boundary interpretation

A positive:

    B_boundary - representation_boundary

means representation instability appears earlier than the validated
prediction-failure boundary.

This is temporal/statistical precedence, not causal evidence.

## Important implementation detail

Embedding shards are paired by their exact numeric suffix:

    embeddings_00000.npy <-> metadata_00000.csv
    embeddings_00001.npy <-> metadata_00001.csv
    ...
    embeddings_00009.npy <-> metadata_00009.csv

The script does not assume `metadata_0.csv`.

Invalid/legacy viewpoint strings such as `V000` are converted to integer
viewpoints automatically. Angle is reconstructed from the benchmark's
V107-centered convention.

JSON NaN and infinite values are converted to JSON `null`, so the report
is always valid JSON.
