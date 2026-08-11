CROSS-IDENTITY ANALYSIS
=======================

A threshold: 13.43702602
C threshold: 0.00237080
Sustained viewpoints: 3
Frontal viewpoint: 107

A: A_angular_distance_deg >= threshold for 3 consecutive viewpoints.
C: C_margin <= threshold for 3 consecutive viewpoints.
B: B_predicted_folder != TRUE FOLDER for 3 consecutive viewpoints.

Traversal:
  left  = 107 -> 106 -> ... -> 0
  right = 107 -> 108 -> ... -> 214

Lead:
  B_distance_from_frontal - A_distance_from_frontal
  lead > 0 means A-before-B.

A horizon definition:
  0 < lead <= H

Sequences must contain exactly one row for every viewpoint 0..214.
Female/Male pairing is expression + side level, never a raw sequence Cartesian product.
Small appearance groups are descriptive only.
Positive A-before-B is precedence, not causality.
