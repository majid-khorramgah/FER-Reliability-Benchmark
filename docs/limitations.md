
---

## `docs/limitations.md`

```markdown
# Limitations

The FER Reliability Benchmark is designed to evaluate representation behavior and early-warning relationships under controlled viewpoint variation. The following limitations should be considered when interpreting the results.

## 1. No Causal Claim

The benchmark evaluates statistical precedence, association, and predictive utility.

An observed A-before-B relationship does not establish that representation instability causes recognition failure.

Therefore, the results should not be interpreted as causal evidence.

## 2. Viewpoint-Centered Evaluation

The benchmark focuses primarily on viewpoint variation.

Real-world facial images may contain additional sources of variation, including changes in illumination, occlusion, image quality, facial appearance, pose, and other factors.

The current benchmark does not establish that the same behavior will occur under all such conditions.

## 3. Dataset Dependence

The results depend on the dataset and sequence construction used by the benchmark.

The identities, expressions, viewpoints, image characteristics, and sequence eligibility criteria may influence the observed results.

Independent validation on other datasets is required before making broader generalization claims.

## 4. Representation Dependence

Representation-level findings depend on the model and representation space used in the benchmark.

A different architecture, model checkpoint, embedding layer, representation metric, or training procedure may produce different results.

Therefore, the current findings should not automatically be generalized to all facial expression recognition systems.

## 5. Expression Coverage

Although the benchmark evaluates multiple facial expression groups, not every expression group necessarily satisfies the same eligibility criteria.

Primary cross-expression conclusions therefore depend on the set of primary eligible expression groups.

Small or incomplete groups should be interpreted cautiously.

## 6. Identity Coverage

The benchmark evaluates multiple identity conditions, but the number and composition of identities are finite.

Observed consistency across the tested identity groups does not guarantee identical behavior for unseen populations or datasets.

## 7. Viewpoint Direction

The benchmark separately evaluates left and right viewpoint directions.

Differences between the two directions may reflect characteristics of the underlying data, representation, or sequence construction.

Symmetry between left and right directions should therefore not be assumed without evidence.

## 8. Incomplete Sequences

Not every sequence necessarily contains all expected viewpoints.

The analysis therefore applies sequence-completeness and eligibility criteria where required.

Results based on complete sequences should not be interpreted as representing every available image equally.

## 9. Early Warning Is Not Perfect Prediction

An early-warning relationship does not imply that every future recognition failure can be predicted.

The benchmark evaluates statistical precedence and predictive utility rather than deterministic failure prediction.

## 10. Threshold Dependence

Some event definitions depend on fixed thresholds.

The benchmark addresses this issue through sensitivity and robustness analyses, but threshold-dependent results should still be interpreted within the tested parameter ranges.

## 11. Horizon Dependence

The strength of an early-warning signal can depend on the selected warning horizon.

A warning effect observed at one horizon does not necessarily imply the same effect at every possible horizon.

## 12. Multiple Analyses

The benchmark contains multiple analyses across:

- expressions;
- identities;
- viewpoint directions;
- warning horizons;
- representation metrics;
- robustness settings.

The complete evidence should therefore be considered rather than relying on a single statistic or analysis.

## 13. Statistical Precedence vs. Mechanism

The observation that event A tends to occur before event B establishes an ordering relationship within the tested sequences.

It does not by itself identify the mechanism responsible for that ordering.

Additional controlled experiments would be required to determine why the representation changes and whether those changes directly contribute to subsequent recognition failure.

## 14. Final Synthesis Is Not a New Statistical Analysis

The final synthesis stage consolidates existing evidence from the previous analysis stages.

It does not rerun or re-estimate the underlying analyses.

Therefore, the final synthesis should be understood as an evidence-integration stage rather than an independent statistical experiment.

## 15. Missing Evidence

If a previous analysis stage or output file is unavailable, the final synthesis reports the missing evidence rather than treating it as completed.

Consequently, conclusions from the final synthesis are limited to the evidence that was actually available at the time of synthesis.

## 16. General Interpretation

The most appropriate interpretation of the benchmark is:

> The benchmark provides evidence for systematic representation-level changes and A-before-B statistical precedence under controlled viewpoint variation, with supporting analyses across robustness, expression, identity, and representation-level conditions.

This should not be restated as:

> Representation instability causes FER failure.

The latter is a causal claim that is not established by the current benchmark.

## 17. Future Validation

Stronger validation could include:

- independent datasets;
- independent FER architectures;
- additional representation layers;
- additional viewpoint conditions;
- controlled illumination and occlusion experiments;
- preregistered evaluation criteria;
- out-of-distribution validation;
- experiments specifically designed to test causal mechanisms.

These extensions would help determine whether the observed representation-level early-warning relationship generalizes beyond the current benchmark configuration.