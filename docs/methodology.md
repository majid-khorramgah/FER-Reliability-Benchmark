# Methodology

## Overview

The FER Reliability Benchmark is a controlled empirical framework for studying how facial-expression recognition behavior changes under systematic viewpoint variation.

The central methodological question is whether measurable representation-level changes occur before recognition failure as viewpoint moves away from a fixed frontal reference.

The analysis is organized as a sequential pipeline:

'''text
Stage 01 — Build Metadata
        ↓
Stage 02 — Extract Embeddings
        ↓
Stage 03 — Analyze Embeddings
        ↓
Stage 04 — Analyze Embedding Trajectory
        ↓
Stage 05 — Analyze Ordering
        ↓
Stage 06 — Ordering Diagnostics
        ↓
Stage 07 — Statistical Validation
        ↓
Stage 08 — Early-Warning Analysis
        ↓
Stage 09 — Early-Warning Horizon Analysis
        ↓
Stage 10 — Robustness and Sensitivity
        ↓
Stage 11 — Cross-Expression Validation
        ↓
Stage 12 — Cross-Identity Validation
        ↓
Stage 13 — Representation Validation
        ↓
Stage 14 — Final Synthesis
'''
Each later stage uses evidence produced by earlier stages. Stage 14 integrates the existing evidence and does not rerun or re-estimate the previous analyses.

---

# 1. Experimental Principle

The benchmark uses controlled facial-expression sequences in which viewpoint is systematically varied while the underlying expression configuration is kept fixed within a sequence.

A frontal viewpoint is used as the reference condition.

The basic experimental trajectory is:

Fixed expression
Fixed identity
        ↓
Systematic viewpoint change
        ↓
Representation change
        ↓
Recognition behavior

This structure allows viewpoint-dependent changes in representation and recognition to be studied along an ordered trajectory rather than as unrelated images.

The benchmark is intended for controlled diagnosis of reliability behavior. It does not assume that viewpoint-induced representation change is necessarily harmful or causal.

---

# 2. Benchmark Configuration

The validated analysis uses the following fixed configuration:

Frontal viewpoint: 107
Expected viewpoints: 215

Fixed A threshold: 13.43702602
Fixed C threshold: 0.0023708

These values define the validated configuration used by the downstream analysis.

The thresholds are treated as fixed analysis parameters rather than being re-estimated during the final synthesis.

---

# 3. Stage 01 — Build Metadata

The first stage constructs the metadata required to organize the benchmark into valid experimental sequences.

The metadata identifies the relevant experimental factors, including:

- identity;
- expression;
- viewpoint;
- image path;
- gender or identity condition;
- sequence information;
- viewpoint direction.

The purpose of this stage is to establish the structural correspondence between images and their experimental conditions.

The resulting metadata is used by downstream stages to identify valid viewpoint trajectories and matched comparisons.

---

# 4. Stage 02 — Extract Embeddings

The second stage extracts image representations from the benchmark images using the selected representation model.

The extracted embeddings provide the representation-level data used by subsequent analyses.

Depending on the analysis stage, the embeddings are used to measure quantities such as:

- representation similarity;
- representation distance or drift;
- representation margins;
- viewpoint-dependent representation changes;
- same-expression versus rival-expression relationships;
- identity-related representation behavior.

The embedding extraction stage does not by itself establish that representation changes predict recognition failure. It provides the representation data required to test that hypothesis.

---

# 5. Stage 03 — Analyze Embeddings

The third stage performs the initial characterization of the extracted embeddings.

The purpose is to determine how the learned representation behaves across:

- viewpoint;
- expression;
- identity;
- sequence;
- viewpoint direction.

This stage provides the measurements required for subsequent trajectory and event-ordering analyses.

The output of this stage is treated as descriptive representation-level evidence rather than as final evidence of an early-warning effect.

---

# 6. Stage 04 — Analyze Embedding Trajectory

The fourth stage analyzes representation changes along the ordered viewpoint trajectory.

For each valid sequence, representation behavior is evaluated relative to the fixed frontal reference.

The analysis considers representation-derived quantities including, where applicable:

- cosine-based similarity or distance;
- angular representation change;
- Euclidean representation change;
- trajectory/path behavior;
- representation margins;
- viewpoint-dependent instability.

The objective is to determine where measurable representation-level changes occur as viewpoint departs from the frontal condition.

This stage provides the basis for defining and evaluating representation-related events.

---

# 7. Stage 05 — Analyze Event Ordering

The fifth stage evaluates the temporal or sequential ordering of predefined events along each viewpoint trajectory.

The central relationship is:

A → B

where:

A = predefined representation-related event
B = predefined recognition-failure event

The primary quantity is the proportion of valid sequences in which:

A occurs before B

This quantity is referred to as:

A-before-B precedence

The analysis also records the corresponding lead between the two events.

A positive lead indicates that A occurs earlier in the viewpoint trajectory than B.

---

# 8. Stage 06 — Ordering Diagnostics

The sixth stage performs diagnostic analyses of the observed event ordering.

The purpose is to examine whether the A-before-B relationship is stable and interpretable before drawing stronger statistical conclusions.

The diagnostics examine the distribution and behavior of:

- event locations;
- A-before-B ordering;
- lead distance;
- viewpoint direction;
- sequence-level behavior.

These diagnostics are used to identify potential irregularities and to characterize the structure of the observed ordering.

---

# 9. Stage 07 — Statistical Validation

The seventh stage evaluates the statistical evidence associated with the observed event ordering.

The analysis uses statistical tests and permutation-based null comparisons appropriate to the benchmark design.

The purpose is to determine whether the observed ordering is distinguishable from an appropriate null expectation.

Statistical validation is interpreted as evidence against the relevant null hypothesis.

It does not establish causality.

---

# 10. Stage 08 — Early-Warning Analysis

The eighth stage directly evaluates whether the representation-related event A can serve as an earlier statistical signal of recognition failure B.

The basic structure is:

Representation-related event A
             ↓
        lead interval
             ↓
Recognition failure B

The analysis quantifies:

- A-before-B precedence;
- A event location;
- B event location;
- lead distance;
- median lead;
- mean lead where applicable;
- warning behavior;
- predictive-model performance;
- permutation-based ordering evidence.

The early-warning analysis distinguishes between:

- temporal/statistical precedence;
- predictive utility;
- causal interpretation.

In particular:

A-before-B ≠ causal evidence

A-before-B means that A tends to occur before B within the evaluated viewpoint sequences. It does not demonstrate that A causes B.

---

# 11. Stage 09 — Early-Warning Horizon Analysis

The ninth stage evaluates the early-warning relationship across multiple viewpoint horizons.

The purpose is to determine how far before recognition failure the warning signal remains detectable.

Rather than relying on a single lead distance, the analysis evaluates multiple tested horizons.

The horizon analysis records quantities including:

- viewpoint horizon;
- number of eligible sequences;
- number of warnings;
- warning rate;
- lead behavior;
- permutation-based evaluation.

The resulting horizon profile indicates how warning performance changes as the required prediction horizon increases.

A decrease in warning performance at larger horizons is therefore not interpreted as evidence that the phenomenon disappears; it indicates that earlier prediction becomes more difficult.

---

# 12. Stage 10 — Robustness and Sensitivity

The tenth stage evaluates whether the observed early-warning relationship remains present under alternative analytical conditions.

The robustness analysis includes:

- threshold sensitivity;
- sustained-viewpoint definitions;
- metric sensitivity;
- horizon sensitivity;
- bootstrap analysis;
- permutation analysis;
- robustness scoring.

The purpose is to determine whether the principal ordering pattern depends strongly on a single analytical configuration.

A result that remains present across multiple reasonable analytical choices provides stronger evidence that the observed relationship is not solely an artifact of one parameter setting.

Robustness analysis does not eliminate the limitations of the benchmark or establish universal generalization.

---

# 13. Stage 11 — Cross-Expression Validation

The eleventh stage evaluates whether the A-before-B relationship generalizes across facial-expression groups.

The analysis produces expression-level summaries and pooled comparisons.

The analysis distinguishes between:

All available expression groups

and:

Primary eligible expression groups

Primary generality claims are based on the predefined eligibility criteria.

Small or insufficiently represented expression groups are treated descriptively rather than being used as equivalent evidence for the primary generalization claim.

The purpose of this stage is to determine whether the observed early-warning relationship is concentrated in a small subset of expressions or is broadly observable across the eligible expression groups.

The analysis also allows the magnitude or location of the boundary to vary across expressions.

Therefore, broad precedence across expressions should not automatically be interpreted as identical expression-wise behavior.

---

# 14. Stage 12 — Cross-Identity Validation

The twelfth stage evaluates whether the observed relationship remains present across the evaluated identity conditions.

The analysis compares the available identity groups under matched expression and viewpoint conditions.

The evaluated quantities include:

- identity-level A-before-B rates;
- median lead;
- mean lead;
- confidence intervals where available;
- identity-specific horizon behavior;
- bootstrap results;
- robustness scores;
- paired identity comparisons.

The purpose is to determine whether the observed early-warning relationship is restricted to a particular identity condition.

The current benchmark provides controlled identity comparisons, but the available identities do not constitute a population-level demographic sample.

Therefore, cross-identity results should be interpreted as evidence within the tested benchmark conditions rather than as general demographic conclusions.

---

# 15. Stage 13 — Representation Validation

The thirteenth stage provides additional validation directly at the representation level.

The analysis includes:

- viewpoint-level representation metrics;
- representation boundaries;
- sequence-level representation summaries;
- expression-level representation summaries;
- viewpoint profiles;
- pairwise representation similarity;
- pairwise statistical testing;
- paired identity representation analysis;
- identity-level statistical testing;
- representation lead/lag analysis;
- bootstrap analysis;
- false-discovery-rate analysis.

A key analysis compares representation behavior for:

Same-expression pairs

against:

Rival-expression pairs

This comparison is intended to determine whether representation behavior contains structure related to expression identity beyond the final recognition output.

The representation-level analysis therefore complements the earlier event-ordering analyses.

It does not by itself prove that representation drift is the causal mechanism producing recognition failure.

---

# 16. Stage 14 — Final Synthesis

Stage 14 consolidates the evidence generated by the preceding analyses.

It does not rerun, re-estimate, or modify the underlying analyses.

Instead, it:

1. checks which earlier-stage evidence is available;
2. loads existing CSV and JSON outputs;
3. checks the schemas of available evidence files;
4. identifies compatible columns;
5. extracts relevant metrics;
6. builds synthesis tables;
7. builds final claims;
8. generates publication-oriented summary tables;
9. generates final plots;
10. produces the final synthesis report.

The purpose of Stage 14 is therefore evidence integration rather than new statistical estimation.

---

# 17. Event Definitions

The central event-ordering framework uses the following notation:

A = predefined representation-related event
C = predefined representation-margin / separability event
B = predefined recognition-failure event

The primary ordering relationship is:

A before B

Additional analyses may evaluate relationships involving C and B.

The exact operational definitions and thresholds are those established by the validated analysis pipeline.

The benchmark therefore distinguishes between:

- representation change;
- representation-margin change;
- recognition failure.

---

# 18. Early-Warning Interpretation

An early-warning relationship means that a measurable event is observed before a subsequent recognition failure often enough to provide potentially useful statistical information.

It does not imply that:

- every recognition failure is predictable;
- every A event is followed by recognition failure;
- the relationship is deterministic;
- the relationship is causal;
- the representation event necessarily explains the failure mechanism.

The scientifically appropriate interpretation is that the representation-related event may contain information that appears earlier along the viewpoint trajectory than the final recognition failure.

---

# 19. Representation Interpretation

The representation analysis is motivated by the following trajectory:

Fixed expression
      ↓
Viewpoint changes
      ↓
Representation changes
      ↓
Representation-level instability / margin change
      ↓
Recognition failure

An observed change in representation should not automatically be interpreted as representation failure.

A representation can change under viewpoint variation while still preserving semantic information.

Therefore, an important unresolved question is whether the observed representation changes reflect:

- loss of useful semantic information;

or:

- viewpoint adaptation while semantic information remains recoverable.

The current benchmark provides empirical evidence for studying this distinction but does not by itself resolve the underlying mechanism.

---

# 20. Predictive Interpretation

The early-warning analyses evaluate whether representation-derived signals contain information about subsequent recognition failure.

Predictive performance is therefore interpreted as evidence of predictive utility within the evaluated experimental setting.

It should not be interpreted as proof that the representation itself is the causal source of failure.

A useful conceptual distinction is:

Representation signal
        ↓
Predictive information
        ≠
Causal mechanism

Further experiments would be required to distinguish between competing mechanisms such as representation degradation and downstream readout failure.

---

# 21. Cross-Expression Interpretation

Cross-expression analysis addresses whether the observed precedence is concentrated in only a small subset of expressions.

Evidence that A-before-B occurs across many eligible expressions supports broader within-benchmark generalization.

However, expression-specific differences in:

- boundary location;
- lead;
- warning rate;
- robustness;

may still exist.

Therefore, the appropriate interpretation is:

Broad precedence across expressions
+
Potential expression-dependent boundary behavior

rather than the assumption that every expression follows an identical trajectory.

---

# 22. Cross-Identity Interpretation

Cross-identity validation tests whether the observed phenomenon remains present under the available matched identity conditions.

If A-before-B is observed consistently across the evaluated identities, this provides evidence that the effect is not restricted to one tested identity.

However:

Controlled identity comparison
≠
Population-level demographic validation

The benchmark should therefore not be used to make broad demographic fairness claims without additional identities and independent datasets.

---

# 23. Statistical Interpretation

The statistical analyses are designed to evaluate whether the observed event ordering and representation relationships are unlikely under the relevant null comparisons.

Permutation results provide evidence relative to the specified permutation procedure.

Bootstrap analyses characterize uncertainty in the estimated quantities.

Multiple-comparison procedures, including false-discovery-rate control where applicable, are used in the relevant representation-level analyses.

Statistical significance should therefore be interpreted together with:

- effect magnitude;
- consistency across sequences;
- robustness;
- horizon behavior;
- expression generalization;
- identity generalization.

A statistically significant result is not by itself evidence of practical importance or causality.

---

# 24. Reproducibility of the Method

Reproducing the validated analysis requires preserving:

1. the benchmark images;
2. metadata;
3. representation model and configuration;
4. fixed analysis thresholds;
5. analysis scripts;
6. Python environment and dependencies;
7. execution order;
8. intermediate evidence files;
9. final synthesis outputs.

The fixed validated configuration includes:

Frontal viewpoint = 107
Expected viewpoints = 215
A threshold = 13.43702602
C threshold = 0.0023708

Changing these parameters or the representation model may change the resulting measurements and should therefore be treated as a new experimental configuration.

---

# 25. Scope of the Method

The methodology is designed to study reliability behavior in facial-expression recognition under controlled viewpoint variation.

The results are evidence within the tested experimental configuration.

The methodology does not by itself establish:

- universal behavior across all FER systems;
- universal behavior across all visual recognition tasks;
- a causal mechanism;
- guaranteed prediction of individual failures;
- generalization to all identities;
- generalization to real-world imagery;
- that representation drift is inherently undesirable.

Broader claims require independent validation.

---

# 26. Scientific Logic of the Pipeline

The methodological logic can be summarized as:

Control viewpoint
        ↓
Construct valid sequences
        ↓
Extract representations
        ↓
Measure representation behavior
        ↓
Identify predefined events
        ↓
Test event ordering
        ↓
Statistically validate ordering
        ↓
Evaluate early-warning behavior
        ↓
Test prediction horizons
        ↓
Test robustness
        ↓
Test expression generalization
        ↓
Test identity generalization
        ↓
Validate representation-level behavior
        ↓
Synthesize the evidence

The central empirical relationship is:

Representation-level signal A
              ↓
        earlier viewpoint
              ↓
Recognition failure B

The key quantity is therefore:

P(A before B)

within valid benchmark sequences.

The scientific objective is not to assume that A causes B, but to determine whether the ordering is reproducible, statistically supported, robust, generalizable within the benchmark, and informative enough to motivate further research into representation dynamics and recognition reliability.

---

# 27. Methodological Summary

The methodology follows a discovery-first empirical approach:

Controlled intervention
        ↓
Representation measurement
        ↓
Event detection
        ↓
Temporal ordering
        ↓
Statistical validation
        ↓
Early-warning evaluation
        ↓
Robustness testing
        ↓
Cross-expression validation
        ↓
Cross-identity validation
        ↓
Representation-level validation
        ↓
Scientific interpretation

The final research claim is therefore based on converging evidence across multiple analyses rather than on a single metric.

The current methodology supports investigation of the following central question:

> Can viewpoint-induced changes in visual representations be characterized as a structured transition that contains measurable information about subsequent recognition failure?

This question remains open to further validation beyond the current controlled facial-expression benchmark.
