# Cross-expression analysis

## Scientific question

Does the early-warning phenomenon generalize across
expression categories, or is it concentrated in a
small subset?

## Validated configuration

Frontal viewpoint:

    107

Expected viewpoints:

    215

A threshold:

    13.43702602

C threshold:

    0.00237080

Sustained viewpoints:

    3

Primary minimum expression size:

    8

## A definition

A boundary is detected when:

    A_angular_distance_deg >= 13.43702602

for 3 consecutive viewpoints.

## C definition

A boundary is detected when:

    C_margin <= 0.00237080

for 3 consecutive viewpoints.

## B definition

A boundary is detected when:

    predicted_folder != true folder

for 3 consecutive viewpoints.

## Critical methodological point

Thresholds are FIXED.

They are not re-estimated for individual
expression categories.

This makes this analysis directly comparable
with the validated early-warning and horizon analyses.

## Primary analysis

Expression categories with at least
8 complete sequences are included
in the primary analysis.

Small groups remain visible in the complete
CSV files but are treated as exploratory.

## Interpretation

Positive A-before-B means:

    representation drift boundary
    occurs before
    prediction failure boundary.

This is evidence of temporal/statistical precedence.

It is not proof of causality.

## Output

sequence_events.csv
expression_coverage.csv
expression_summary_all.csv
expression_summary_primary.csv
expression_horizon_summary.csv
pooled_primary_summary.csv
bootstrap_expression_rates.csv
expression_heterogeneity_permutation.csv
kruskal_wallis.csv
leave_one_expression_out.csv
cross_expression_report.json

plots/expression_warning_rate_left.png
plots/expression_warning_rate_right.png
plots/expression_lead_left.png
plots/expression_lead_right.png
plots/expression_horizon_heatmap.png