# A / C / B Ordering Diagnostic Analysis

This analysis does NOT assume A < C < B.

A = representation drift
C = expression separability
B = expression prediction consistency

The analysis follows each complete 215-viewpoint expression sequence.

A boundary is detected when the angular representation distance exceeds the robust A threshold for 3 consecutive viewpoints.

C boundary is detected when C_margin falls below the robust C threshold for 3 consecutive viewpoints.

B boundary is detected when prediction differs from the true expression folder for 3 consecutive viewpoints.

## Thresholds

- A_angular: 13.43702602
- A_cosine: 0.0273741
- A_euclidean: 0.23398377
- A_path: 0.41527239
- A_rate: 0.10537625
- A_curvature: 43.199919
- A_instability: 0.16426415
- C_margin: 0.0023708

## Detection diagnostics

 side  total_expressions  A_detected  C_detected  B_detected  A_missing  C_missing  B_missing  all_three
 left                427         427         426         426          0          1          1        426
right                427         427         426         426          0          1          1        426

## Directional summary

 side  n_complete  median_A  median_C  median_B  median_A_C_gap  median_C_B_gap  median_A_B_gap  A_before_C  C_before_B  A_before_B    A_C_B    B_C_A
 left         426      99.5      92.0      89.0             7.0             2.0            10.5    0.136150    0.000000    0.023474 0.000000 0.530516
right         426     115.0     124.0     127.0             8.0             1.0            10.0    0.788732    0.699531    0.896714 0.502347 0.000000

## Interpretation

The ordering is empirical. The script reports all observed orderings and does not impose a preferred causal sequence.

A missing boundary means that the corresponding event was not detected under the current operational definition; it does not mean that the representation or model necessarily remained perfectly stable.