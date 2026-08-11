# Stage 10 — Final Synthesis

## Purpose

Stage 10 integrates the validated results of the FER Reliability Benchmark.

It does not re-estimate A/C thresholds and does not replace previous analyses.

## Fixed benchmark configuration

- A threshold: `13.43702602`
- C threshold: `0.0023708`
- Frontal viewpoint: `107`
- Expected viewpoints: `215`
- Sustained rule: `3`
- Complete sequences: `427`
- Expressions: `99`

## Main integrated questions

### 1. Does viewpoint-induced reliability degradation precede classifier failure?

This is evaluated by comparing the temporal ordering of representation-level instability and prediction-level failure.

### 2. Does identity change the timing of this instability?

The Female/Male paired analysis tests whether the lead/lag structure differs across identities.

### 3. Is the observed effect robust?

Stages 6–9 provide robustness, bootstrap, expression-level and representation-level validation.

## Important interpretation

A-before-B is interpreted as statistical/temporal precedence.

It is NOT interpreted as causal evidence.

Small appearance groups are descriptive and should not be used as the primary generality claim.

## Stage 9 representation result

Same-vs-rival difference:

`-0.0392828233411425`

Permutation p-value:

`9.999000099990002e-05`

## Female/Male paired comparison

Female − Male A-before-B difference:

`None`

Permutation p-value:

`None`

Female − Male mean lead difference:

`None` degrees

Permutation p-value:

`None`

## Output

This directory contains:

- final_summary.csv
- final_stage_status.csv
- final_key_results.csv
- final_identity_comparison.csv
- final_representation_results.csv
- final_early_warning_results.csv
- final_expression_results.csv
- final_horizon_results.csv
- final_robustness_results.csv
- final_publication_table.csv
- final_claims.csv
- final_report.json
- plots/

## Scientific role

Stage 10 is the synthesis layer.

The statistical evidence is produced by Stages 1–9.
Stage 10 organizes that evidence into a publication-ready result.
