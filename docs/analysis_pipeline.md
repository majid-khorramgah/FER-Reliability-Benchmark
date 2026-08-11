# Analysis Pipeline

This repository implements a sequential empirical analysis pipeline for the FER Reliability Benchmark.

The project investigates whether representation-level changes under controlled viewpoint variation systematically precede recognition failure in facial expression recognition (FER).

The pipeline is designed to move from controlled representation measurements to event ordering, statistical validation, early-warning analysis, robustness testing, cross-expression and cross-identity validation, representation-level validation, and final evidence synthesis.

The analysis is intended to distinguish:

- representation-level change;
- representation separability or margin degradation;
- recognition failure;
- temporal/statistical precedence between these events;
- robustness of the observed relationship;
- generalization across expressions and identities.

The central empirical relationship is:

```text
Representation-level event A
            ↓
        lead / lag
            ↓
Recognition failure B
```

The main question is whether:

```text
A occurs before B
```

systematically across valid sequences.

Importantly, A-before-B precedence is interpreted as temporal/statistical precedence and predictive utility. It is not interpreted as causal evidence.

---

# 1. Pipeline Overview

The validated project analysis consists of the following major stages:

```text
Stage 1
Statistical Validation
        ↓
Stage 2
Left/Right Validation
        ↓
Stage 3
Permutation Validation
        ↓
Stage 4
Early-Warning Analysis
        ↓
Stage 5
Early-Warning Horizon Analysis
        ↓
Stage 6
Robustness / Sensitivity Analysis
        ↓
Stage 7
Cross-Expression Validation
        ↓
Stage 8
Cross-Identity Validation
        ↓
Stage 9
Representation-Level Validation
        ↓
Stage 10
Final Synthesis
```

Stages 1–9 constitute the empirical analysis and validation pipeline.

Stage 10 is an evidence-integration and reporting stage. It does not rerun, re-estimate, or modify the preceding analyses.

---

# 2. Experimental Structure

The benchmark evaluates facial-expression recognition behavior under systematically controlled viewpoint changes.

A frontal viewpoint is used as the reference condition.

For a given expression and identity, the viewpoint is progressively changed while the underlying expression configuration remains controlled.

The analysis therefore follows sequences of the form:

```text
Frontal reference
      ↓
Viewpoint change
      ↓
Representation change
      ↓
Representation event A
      ↓
Recognition instability / failure B
```

The central objective is not simply to measure accuracy at different viewpoints.

Instead, the analysis asks whether measurable representation-level changes appear before recognition failure.

---

# 3. Core Events

The project defines several measurable events.

## Event A — Representation-Level Change

Event A represents the predefined representation-related degradation event relative to the fixed frontal reference and validated benchmark thresholds.

Depending on the analysis, representation change can be characterized using multiple representation-level quantities, including:

- cosine-based representation change;
- angular representation change;
- Euclidean representation change;
- trajectory/path behavior;
- representation-rate measures;
- instability measures;
- representation boundaries.

The exact operational definition of A is determined by the validated analysis configuration.

A should therefore not be interpreted as a generic statement that "the embedding changed."

It represents the specific event definition used by the validated benchmark analysis.

---

## Event C — Representation Separability / Margin Change

Event C represents the predefined representation-margin or separability degradation event.

It provides an additional representation-level signal beyond the primary A event.

The analysis therefore allows the relationship between:

```text
A = representation drift / degradation event
C = representation separability / margin degradation
B = recognition failure
```

to be evaluated.

---

## Event B — Recognition Failure

Event B represents the predefined recognition-failure event used by the benchmark.

The precise operational definition is determined by the validated recognition analysis.

The central temporal comparison is:

```text
A → B
```

and, where appropriate:

```text
A + C → B
```

---

# 4. Fixed Benchmark Configuration

The validated benchmark configuration includes the following fixed parameters:

```text
Fixed A threshold: 13.43702602
Fixed C threshold: 0.0023708
Frontal viewpoint: 107
Expected viewpoints: 215
```

These parameters are part of the validated analysis configuration.

They should not be changed when reproducing the reported results unless a new sensitivity analysis is explicitly performed.

---

# 5. Stage 1 — Statistical Validation

Stage 1 performs statistical validation of the benchmark-level observations and event relationships.

The purpose is to determine whether the observed patterns are statistically distinguishable from appropriate null expectations.

The analysis establishes the statistical foundation for interpreting the subsequent early-warning results.

The statistical validation is based on the controlled sequence structure and the predefined event definitions.

The output of this stage provides evidence used by later stages rather than constituting a standalone causal explanation.

---

# 6. Stage 2 — Left/Right Validation

Stage 2 evaluates whether the observed event ordering is preserved across the two directions of viewpoint change.

The benchmark separates:

```text
Left viewpoint progression
```

and:

```text
Right viewpoint progression
```

This is important because a phenomenon observed only on one side could potentially reflect a directional artifact.

The left/right analysis therefore evaluates whether the A-before-B relationship remains present under both directions.

The later results show that the broad precedence pattern is preserved across both sides, although the magnitude of the effect can differ.

This stage therefore provides directional validation rather than assuming left/right symmetry.

---

# 7. Stage 3 — Permutation Validation

Stage 3 evaluates the observed event ordering against permutation-based null expectations.

The purpose is to determine whether the observed A-before-B ordering is unlikely to arise simply from random ordering of the relevant events.

Permutation analysis provides a non-parametric validation of the ordering relationship.

The permutation results are interpreted as evidence against the corresponding null ordering expectation.

They do not establish causality.

A small permutation p-value therefore supports the statement:

```text
The observed ordering is unlikely under the tested permutation null.
```

It should not be interpreted as:

```text
A causes B.
```

---

# 8. Stage 4 — Early-Warning Analysis

Stage 4 directly evaluates whether representation-related degradation can precede recognition failure.

The central relationship is:

```text
Representation event A
        ↓
        ↓ lead
        ↓
Recognition failure B
```

The analysis quantifies:

- A-before-B rate;
- event locations;
- lead distance;
- median lead;
- mean lead;
- warning behavior;
- predictive performance;
- permutation-based ordering evidence.

The main quantity is:

```text
A-before-B precedence
```

defined as the proportion of valid sequences in which A occurs before B.

The current analyses provide substantial evidence for this relationship.

For example, the validated early-warning analysis reported high A-before-B rates across viewpoint directions, with baseline rates substantially above chance-level ordering expectations.

The analysis also evaluated predictive performance.

A viewpoint-only baseline produced approximately:

```text
AUC ≈ 0.890
```

while incorporating representation-related signals together with viewpoint produced approximately:

```text
AUC ≈ 0.9999
```

with an approximate gain of:

```text
ΔAUC ≈ +0.110
```

These values are treated as results of the validated analysis configuration rather than as universal performance claims.

---

# 9. Interpretation of Early Warning

An early-warning relationship means that A is observed before B sufficiently often to provide potentially useful predictive information.

It does not mean:

- every A event is followed by recognition failure;
- every recognition failure is predictable;
- A deterministically determines B;
- A is causally responsible for B;
- the warning signal will transfer automatically to another model or dataset.

The appropriate interpretation is:

```text
A provides an earlier statistical signal associated
with subsequent recognition failure B.
```

The distinction between temporal precedence, predictive utility, and causal explanation is maintained throughout the project.

---

# 10. Stage 5 — Early-Warning Horizon Analysis

Stage 5 investigates how far before recognition failure the warning relationship remains detectable.

Rather than relying on a single lead distance, the analysis evaluates multiple viewpoint horizons.

The horizon analysis includes:

- viewpoint horizon;
- number of eligible sequences;
- number of warnings;
- warning rate;
- lead distance;
- permutation-based evaluation.

The observed warning relationship gradually weakens as the requested prediction horizon becomes larger.

Representative horizon results from the validated analysis include:

| Horizon | Left | Right |
|---:|---:|---:|
| 1° | 95.3% | 89.7% |
| 2° | 91.1% | 84.7% |
| 3° | 85.0% | 77.7% |
| 5° | 76.3% | 70.0% |
| 7° | 68.5% | 62.2% |
| 10° | 54.5% | 51.6% |
| 15° | 32.9% | 35.4% |
| 20° | 22.5% | 25.6% |

These values indicate that the warning signal is strongest at shorter horizons and progressively weaker at longer horizons.

The pattern is therefore better described as an early-warning signal with a measurable lead range rather than as a perfect long-range predictor.

---

# 11. Stage 6 — Robustness and Sensitivity

Stage 6 evaluates whether the main A-before-B relationship depends strongly on a single analytical configuration.

The robustness analysis includes:

- threshold sensitivity;
- sustained-viewpoint definitions;
- metric sensitivity;
- horizon sensitivity;
- bootstrap analysis;
- permutation analysis;
- robustness scoring.

The purpose is to determine whether the main phenomenon remains present when reasonable analytical choices are changed.

The validated analyses indicate that the broad A-before-B relationship persists across multiple sensitivity configurations.

The effect may weaken under more demanding thresholds or longer horizons, but the overall ordering pattern does not disappear.

This supports the interpretation that the result is not solely an artifact of one threshold or one horizon.

---

# 12. Statistical Interpretation of the Robustness Results

The project distinguishes robustness from universality.

A robust result means:

```text
The observed relationship remains present
under multiple tested analytical conditions.
```

It does not mean:

```text
The relationship must occur in every model,
dataset, identity, expression, or real-world setting.
```

The current evidence therefore supports robustness within the validated experimental configuration.

Broader generalization remains an open question.

---

# 13. Stage 7 — Cross-Expression Validation

Stage 7 evaluates whether the observed A-before-B relationship is restricted to a small number of facial expressions.

The analysis includes expression-level summaries and pooled comparisons.

Primary generality claims are based on expression groups satisfying the predefined eligibility criteria.

Small or otherwise ineligible groups are treated descriptively rather than as equivalent evidence for the primary generalization claim.

The validated cross-expression analysis included:

```text
31 primary expression groups
```

with sufficient data for the primary expression-level analysis.

For these groups, the observed A-before-B precedence was broadly preserved.

Representative aggregate results were:

```text
Left:
Mean A-before-B ≈ 94.82%
Median A-before-B = 100%

Right:
Mean A-before-B ≈ 85.66%
Median A-before-B = 90%
```

The observed ranges were approximately:

```text
Left: 77.78% – 100%
Right: 70% – 100%
```

Representative median lead values were:

```text
Left ≈ 9°
Right ≈ 6°
```

The expression-level analyses also produced strong evidence of heterogeneity in the magnitude or location of the effect.

For example:

```text
Kruskal-Wallis:
Left  p ≈ 1.38 × 10⁻⁸
Right p ≈ 2.13 × 10⁻⁶
```

These results should not be interpreted as evidence that every expression behaves identically.

The appropriate interpretation is:

```text
The early-warning precedence generalizes broadly across
eligible expression groups, while the magnitude or boundary
of the effect may remain expression-dependent.
```

This distinction is important for the scientific interpretation of the project.

---

# 14. Leave-One-Expression-Out Analysis

The cross-expression analysis also evaluates whether the overall result depends excessively on a single expression group.

Leave-one-expression-out analysis tests the stability of the aggregate relationship when individual expression groups are removed.

The purpose is to determine whether the observed phenomenon is driven by one particularly strong expression.

The resulting evidence supports the interpretation that the overall precedence pattern is not simply attributable to one expression group.

However, expression-level heterogeneity remains scientifically meaningful and motivates further investigation.

---

# 15. Stage 8 — Cross-Identity Validation

Stage 8 evaluates whether the observed relationship remains present across the available identity conditions.

The benchmark contains matched identity conditions that allow comparison under controlled expression and viewpoint settings.

The analysis includes:

- identity-level A-before-B rates;
- median lead;
- mean lead;
- confidence intervals;
- identity-specific horizon analysis;
- bootstrap analysis;
- paired identity comparisons;
- identity robustness scores;
- appearance-related identity summaries.

The main cross-identity results are:

| Identity | Side | A-before-B | Median lead |
|---|---|---:|---:|
| Female | Left | 93.659% | 12° |
| Female | Right | 85.366% | 12° |
| Male | Left | 96.832% | 9° |
| Male | Right | 93.665% | 8° |

The important observation is that all four evaluated identity/side conditions show substantial A-before-B precedence.

This supports the statement that the observed phenomenon is not restricted to one evaluated identity condition.

---

# 16. Interpretation of Cross-Identity Results

The cross-identity analysis should not be interpreted as a population-level demographic fairness study.

The benchmark contains a limited number of controlled identities.

Therefore the appropriate conclusion is:

```text
The observed early-warning relationship is preserved
across the evaluated matched identity conditions.
```

It is not:

```text
The phenomenon is proven to hold across all demographic groups.
```

The limited identity diversity remains an explicit limitation.

---

# 17. Identity Differences in Lead

The cross-identity results indicate that the proportion of A-before-B cases is high across all evaluated conditions.

Additional analysis found that the difference in mean lead between female and male conditions can be statistically detectable even when the overall A-before-B proportion does not differ substantially.

Therefore the project distinguishes between:

```text
Whether A occurs before B
```

and:

```text
How far before B A occurs
```

This distinction is important because two identity conditions can have similar precedence rates while exhibiting different lead distributions.

---

# 18. Stage 9 — Representation-Level Validation

Stage 9 moves beyond the final recognition outcome and directly evaluates representation behavior.

The representation analysis includes:

- viewpoint-level representation metrics;
- representation boundaries;
- sequence-level summaries;
- expression-level representation summaries;
- viewpoint profiles;
- pairwise representation similarity;
- pairwise statistical testing;
- paired identity representation analysis;
- identity-level statistical testing;
- representation lead/lag analysis;
- bootstrap analysis;
- false-discovery-rate analysis.

The purpose is to determine whether the observed early-warning behavior is accompanied by measurable changes in the learned representation itself.

---

# 19. Representation-Level Measurements

The representation analysis examines several complementary quantities.

These include:

```text
Cosine representation change
Angular representation change
Euclidean representation change
Trajectory/path length
Representation-rate measures
Instability measures
Representation boundaries
Representation margins
Same-expression similarity
Rival-expression similarity
```

The use of multiple representation measures is important because a single embedding metric may provide an incomplete description of representation dynamics.

The project therefore treats representation-level evidence as a collection of related measurements rather than as a single scalar quantity.

---

# 20. Representation Lead/Lag Analysis

The representation-level analysis evaluates whether representation changes precede recognition failure.

The validated cross-identity representation analysis reported the following A-before-B rates:

| Identity | Side | A-before-B rate | Mean lead | Median lead |
|---|---|---:|---:|---:|
| Female | Left | 93.659% | 15.937° | 12° |
| Female | Right | 85.366% | 16.288° | 12° |
| Male | Left | 96.833% | 14.394° | 9° |
| Male | Right | 93.665% | 14.516° | 8° |

The corresponding C-before-B rates were also evaluated:

| Identity | Side | C-before-B rate |
|---|---|---:|
| Female | Left | 73.171% |
| Female | Right | 70.244% |
| Male | Left | 71.041% |
| Male | Right | 69.683% |

These results provide additional evidence that representation-level changes can precede recognition failure.

---

# 21. Same-Expression vs Rival-Expression Representation

A central representation-level comparison evaluates whether representations remain more aligned with the correct expression than with rival expressions as viewpoint changes.

The analysis compares:

```text
Same-expression representation behavior
```

against:

```text
Rival-expression representation behavior
```

The validated representation analysis reported a same-vs-rival difference of approximately:

```text
-0.039283
```

with:

```text
Permutation p ≈ 0.000100
```

The sign and exact interpretation depend on the metric convention used by the analysis.

The scientifically relevant point is that the observed same-vs-rival representation difference is statistically distinguishable under the tested permutation procedure.

This supports representation-level evidence beyond final classification accuracy.

---

# 22. Representation Interpretation

Representation drift should not automatically be interpreted as representation failure.

A change in representation can have at least two possible interpretations:

```text
Viewpoint change
      ↓
Representation changes
      ↓
Possible loss of expression separability
```

or:

```text
Viewpoint change
      ↓
Representation adapts to viewpoint
      ↓
Semantic expression identity remains preserved
```

Therefore an important unresolved scientific question is whether representation change reflects:

1. harmful viewpoint-expression entanglement;
2. benign viewpoint adaptation;
3. downstream readout limitations;
4. or a combination of these mechanisms.

The current benchmark provides evidence for temporal association but does not by itself distinguish these mechanisms completely.

This distinction motivates the proposed next research stage.

---

# 23. A-B-C Relationship

The project can be summarized using the following conceptual sequence:

```text
A
Representation-related degradation
        ↓
        ↓
C
Representation separability / margin degradation
        ↓
        ↓
B
Recognition failure
```

The ordering between these events is empirically measured rather than assumed.

The current evidence is particularly strong for the A-before-B relationship.

C provides an additional representation-level signal and is therefore useful for testing whether multiple representation measurements jointly improve prediction of B.

---

# 24. Early-Warning Predictive Interpretation

The early-warning analyses indicate that representation-related signals can contain predictive information about subsequent recognition failure.

The current results therefore support the more precise statement:

```text
Representation-level measurements provide information
about recognition failure before the failure event occurs
within the tested viewpoint sequences.
```

This is stronger than simply observing that:

```text
accuracy decreases with viewpoint.
```

However, it is weaker than claiming:

```text
representation drift causes recognition failure.
```

The distinction is maintained throughout the project.

---

# 25. Stage 10 — Final Synthesis

Stage 10 is the final evidence-integration stage.

It does not rerun earlier analyses.

It reads the existing evidence generated by Stages 1–9 and produces a consolidated representation of the validated results.

The final synthesis performs the following tasks:

1. Verify the availability of Stage 1–9 evidence.
2. Load existing CSV and JSON outputs.
3. Inspect available schemas.
4. Detect compatible metric columns.
5. Extract relevant measurements.
6. Normalize compatible values where necessary.
7. Build synthesis tables.
8. Build publication-oriented summary tables.
9. Build final claims.
10. Generate final plots.
11. Generate the final synthesis report.
12. Record the status of the analyzed stages.

The final synthesis is therefore an evidence-integration stage rather than a new statistical estimation stage.

---

# 26. Final Synthesis Outputs

The final synthesis produces consolidated outputs such as:

```text
analysis/final_synthesis/
├── final_early_warning_results.csv
├── final_horizon_results.csv
├── final_robustness_results.csv
├── final_expression_results.csv
├── final_identity_comparison.csv
├── final_representation_results.csv
├── final_publication_table.csv
├── final_key_results.csv
├── final_claims.csv
├── final_stage_status.csv
├── final_report.json
└── plots/
    ├── final_identity_comparison.png
    ├── final_representation_lead_lag.png
    ├── final_robustness.png
    └── final_stage_overview.png
```

The exact output set may depend on the current implementation and available evidence files.

---

# 27. Dynamic Schema Handling

The final synthesis inspects the schema of evidence files before extracting values.

This is necessary because equivalent quantities may be stored under different column names across analysis stages.

For example, A-before-B information may appear as:

```text
A_before_B
A_before_B_percent
A_before_B_rate
A_before_B_rate_normalized
```

The synthesis therefore:

1. inspects available columns;
2. identifies compatible fields;
3. normalizes values when required;
4. records the source field used;
5. avoids assuming a single fixed schema.

This reduces the risk of silently extracting an incorrect metric.

---

# 28. Evidence Flow

The complete evidence flow is:

```text
Controlled Images
       ↓
Validated Sequences
       ↓
Representation Measurements
       ↓
Representation Trajectories
       ↓
Event Detection
       ↓
Event Ordering
       ↓
Statistical Validation
       ↓
Early-Warning Analysis
       ↓
Horizon Analysis
       ↓
Robustness / Sensitivity
       ↓
Cross-Expression Validation
       ↓
Cross-Identity Validation
       ↓
Representation-Level Validation
       ↓
Final Evidence Synthesis
```

The final interpretation is based on the combined evidence rather than any single statistic.

---

# 29. Main Scientific Question

The pipeline investigates:

> Does representation-level degradation systematically precede recognition failure as viewpoint moves away from a controlled frontal reference?

The broader research question emerging from the completed analyses is:

> Can viewpoint-induced degradation in visual recognition be understood as a structured transition in representation space, and can these representational changes predict recognition failure before the failure occurs?

This broader question remains open.

The current benchmark provides evidence relevant to the question but does not claim to have completely solved it.

---

# 30. What the Current Evidence Supports

The completed analysis supports the following empirical statements:

1. Representation-related events can occur before recognition failure.
2. A-before-B precedence is substantial across the evaluated viewpoint directions.
3. The precedence relationship remains present across multiple tested horizons, although it weakens at longer horizons.
4. The relationship remains present under multiple robustness and sensitivity configurations.
5. The relationship generalizes broadly across the primary eligible expression groups.
6. The relationship is preserved across the evaluated matched identity conditions.
7. Representation-level measurements provide evidence beyond final recognition accuracy.
8. Same-expression and rival-expression representation behavior are statistically distinguishable under the tested representation analysis.
9. The representation-level and behavioral analyses together motivate further investigation of the mechanism underlying the transition.

---

# 31. What the Current Evidence Does Not Establish

The pipeline does not establish that:

- representation drift universally causes recognition failure;
- the phenomenon is universal across all vision models;
- the phenomenon automatically transfers beyond FER;
- the current synthetic benchmark represents all real-world facial imagery;
- two identities establish population-level demographic conclusions;
- A is a causal mechanism;
- every recognition failure can be predicted;
- representation drift necessarily means semantic representation has failed;
- the current representation space is the only or optimal representation for early-warning prediction.

These remain open questions for future research.

---

# 32. Scientific Interpretation

The central empirical observation can be stated conservatively as:

> Under the controlled viewpoint sequences evaluated in this benchmark, representation-related changes frequently precede recognition failure, and this temporal precedence remains detectable across multiple expressions, identities, horizons, and robustness configurations.

The stronger unresolved question is:

> Why does the representation change before recognition failure, and what does that change mean for the underlying visual representation?

This distinction separates the completed empirical benchmark from the next research problem.

---

# 33. Reproducibility

Reproducing the validated analysis requires preserving:

1. input images;
2. metadata;
3. sequence definitions;
4. identity and expression mappings;
5. viewpoint definitions;
6. representation model and configuration;
7. fixed thresholds;
8. analysis scripts;
9. Python environment and dependencies;
10. execution order;
11. generated evidence files;
12. final synthesis configuration.

The validated fixed configuration includes:

```text
A threshold = 13.43702602
C threshold = 0.0023708
Frontal viewpoint = 107
Expected viewpoints = 215
```

Changing these parameters creates a different analytical configuration and should be reported as such.

---

# 34. Reproducibility Principle

The project separates:

```text
Analysis
```

from:

```text
Synthesis
```

The analysis stages generate the empirical evidence.

The final synthesis collects and organizes that evidence.

Therefore:

```text
Stages 1–9
    ↓
Generate / validate evidence
    ↓
Stage 10
    ↓
Integrate evidence
```

Stage 10 should not be treated as an independent re-analysis of the original data.

---

# 35. Scientific Scope

The benchmark is a controlled study of FER reliability under systematic viewpoint variation.

The results should therefore be interpreted within the tested experimental configuration.

The project is designed to establish a measurable phenomenon first and investigate broader implications second.

The intended progression is:

```text
Controlled observation
        ↓
Empirical validation
        ↓
Failure characterization
        ↓
Representation-level interpretation
        ↓
Mechanistic hypothesis
        ↓
Generalization / transfer
        ↓
Potential new method
```

The current work has reached the stage where the empirical phenomenon has been characterized sufficiently to motivate the next research question.

---

# 36. Next Research Direction

The next stage is not simply to run more of the same benchmark.

The main open problem is to determine whether the observed representation dynamics reflect a broader property of visual representations.

Possible directions include:

- comparing different representation models;
- comparing different recognition architectures;
- analyzing representation trajectories rather than only endpoint metrics;
- testing whether early-warning signals transfer across models;
- testing new identities and expressions;
- testing more realistic image conditions;
- evaluating transfer beyond facial-expression recognition;
- separating representation degradation from downstream classifier/readout failure;
- investigating whether representation changes can be used for interpretable failure prediction.

A particularly important experimental question is:

> Does the representation itself lose semantic separability under viewpoint change, or does the representation remain informative while the downstream recognition head becomes unable to decode it?

This question provides a natural bridge from the completed benchmark to representation learning and visual reasoning.

---

# 37. Final Summary

The complete analysis pipeline can be summarized as:

```text
Controlled Viewpoint Sequences
          ↓
Statistical Validation
          ↓
Left/Right Validation
          ↓
Permutation Validation
          ↓
Early-Warning Analysis
          ↓
Warning Horizons
          ↓
Robustness / Sensitivity
          ↓
Cross-Expression Validation
          ↓
Cross-Identity Validation
          ↓
Representation-Level Validation
          ↓
Final Evidence Synthesis
```

The central empirical finding is not simply that recognition accuracy decreases with viewpoint.

The more specific observation is:

```text
Viewpoint change
      ↓
Representation-level change
      ↓
measurable lead
      ↓
Recognition failure
```

The current evidence indicates that this ordering is substantial and reproducible within the tested benchmark configuration.

The remaining scientific question is whether this represents a general phenomenon of visual representation dynamics, and whether the observed representation changes can be used to predict or explain recognition failure before it occurs.

That question is the intended bridge from the current benchmark to the next stage of research.
