EARLY-WARNING HORIZON ANALYSIS
==============================

This analysis evaluates whether representation instability A
appears before prediction failure B.

Coordinate system
-----------------

Frontal viewpoint:

    107°

LEFT:

    107 -> 0

RIGHT:

    107 -> 214

For both sides, the analysis converts viewpoint to distance
from the frontal reference:

    distance = abs(viewpoint - 107)

Therefore left and right are directly comparable.

Definitions
-----------

A:
Representation drift / instability detected using:

    A_angular_distance_deg >= A threshold

C:
Confidence-margin instability detected using:

    C_margin <= C threshold

B:
Prediction failure:

    predicted folder != true folder

Sustained detection
-------------------

An event is considered detected only if the condition remains
true for 3 consecutive viewpoints.

Early-warning lead
------------------

    lead = B_distance - A_distance

If:

    lead > 0

representation drift precedes prediction failure.

If:

    lead = 0

both events occur at the same detected viewpoint.

If:

    lead < 0

prediction failure occurs before representation drift.

Horizon
-------

For horizon H:

    B_distance - A_distance >= H

means the representation signal provides at least H degrees
of warning before prediction failure.

Important
---------

This analysis does NOT prove causality.

It tests predictive/temporal precedence.

A positive and statistically significant horizon means that
representation drift tends to appear earlier than prediction
failure under the operational definitions used here.