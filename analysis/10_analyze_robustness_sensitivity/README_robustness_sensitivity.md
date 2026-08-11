
# Robustness / Sensitivity Analysis

## Purpose

This analysis tests whether the main early-warning result is
sensitive to reasonable changes in the operational definitions.

The main question is:

> Does representation drift tend to precede prediction failure,
> and does this conclusion survive reasonable analytical choices?

## Baseline

A threshold:
13.43702602

C threshold:
0.0023708

Sustained viewpoints:
3

Primary A metric:
A_angular_distance_deg

## Sensitivity tests

### 1. A threshold

The A threshold is multiplied by:

[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

### 2. C threshold

The C threshold is multiplied by:

[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

### 3. Sustained viewpoints

Tested values:

[1, 2, 3, 4, 5]

### 4. A metrics

Tested metrics:

['angular', 'cosine', 'euclidean', 'path', 'rate', 'curvature', 'instability']

### 5. Early-warning horizons

Tested horizons:

[1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

### 6. Bootstrap

Bootstrap repetitions:

1000

### 7. Permutation

Permutation repetitions:

2000

## Interpretation

A result is considered more robust when:

1. A-before-B remains high across reasonable thresholds.
2. Median A-before-B lead remains positive.
3. Left and right sides show similar qualitative behavior.
4. Bootstrap confidence intervals remain away from zero.
5. Permutation tests remain significant.
6. Results do not depend on one specific A metric.

## Important

Robustness does not establish causality.

It establishes that the observed statistical pattern is
not easily explained by one arbitrary choice of threshold,
sustained duration, metric, or side.

## Main output

See:

robustness_sensitivity_report.json

robustness_sensitivity_results.csv

robustness_horizon_results.csv

robustness_bootstrap.csv

robustness_permutation.csv

robustness_score.csv
