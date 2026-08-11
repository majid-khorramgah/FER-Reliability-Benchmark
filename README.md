# FER-Reliability-Benchmark

### A Controlled Benchmark for Studying Facial Expression Recognition Reliability Under Systematic Viewpoint and Appearance Changes

---

## Overview

Facial Expression Recognition (FER) systems are commonly evaluated using aggregate recognition performance. However, aggregate accuracy does not fully describe **when a model becomes unreliable**, **under which controlled visual changes this occurs**, or whether changes in model behavior can provide measurable warning signals before recognition failure.

This project introduces a controlled synthetic benchmark designed to study these questions.

The benchmark uses matched digital-human identities with systematically varied facial expressions and viewpoints. Because the images are rendered under controlled conditions, viewpoint and selected appearance factors can be studied while keeping important underlying factors fixed.

The goal is not simply to train another facial-expression recognition model.

Instead, the project investigates whether controlled visual interventions can reveal and characterize:

- recognition failure boundaries;
- viewpoint-dependent reliability changes;
- expression-specific degradation;
- identity-dependent behavior;
- robustness of observed effects;
- and representation-level signals that may precede recognition failure.

The project follows a **discovery-first research strategy**: empirical evidence is established before proposing a new reliability method.

---

# Main Research Question

> **Can controlled 3D viewpoint interventions reveal expression-specific failure boundaries in facial-expression recognition models?**

A related empirical question is:

> **Do FER models remain appropriately reliable as they move beyond their reliable operating range, or can they become confidently wrong under controlled viewpoint changes?**

The subsequent representation-level analysis motivates a broader research question:

> **Can viewpoint-induced degradation in visual recognition be understood as a structured transition in representation space, and can these representational changes predict recognition failure before the failure occurs?**

The broader question is treated as a research direction motivated by the empirical findings, not as an established conclusion.

---

# Researcher Question

The project has reached a stage where the empirical benchmark and validation pipeline can support a broader research discussion.

The specific question for an external researcher is:

> **What representation-level signals emerge before recognition failure under controlled changes in viewpoint, and are these signals sufficiently structured to support early prediction and explanation of failure?**

The motivation is to determine whether the observed phenomenon can be formulated as a broader **representation-learning or visual-reasoning problem**, rather than remaining specific to facial-expression recognition.

The full question prepared for discussion is available in:

`docs/question_for_researcher.md`

---

# Scientific Motivation

A model may achieve high overall accuracy while still failing systematically under particular visual conditions.

For example, a model may correctly recognize an expression from a frontal viewpoint but become unreliable as the face rotates away from the camera.

Therefore, the important scientific question is not only:

> **How accurate is the model?**

but also:

> **At what point does the model become unreliable, how does this transition occur, and can the transition be detected before recognition failure?**

The controlled structure of this benchmark makes it possible to study these transitions systematically.

---

# Dataset

The benchmark is based on synthetically rendered facial images generated using **DAZ Studio**.

The dataset contains approximately:

- **95,030 rendered images**
- **2 primary digital-human identities**
- matched male and female identity configurations
- systematically controlled facial-expression configurations
- systematically varied viewpoints
- 1024 × 1024 image resolution
- additional facial-hair configurations for the male identity

The benchmark is designed around matched experimental conditions so that viewpoint, expression, identity, and selected appearance variables can be compared systematically.

The full image dataset is not currently distributed through the repository.

---

# Controlled Experimental Structure

The central property of the benchmark is its controlled structure.

For a given expression, a structured sequence of viewpoints is evaluated while keeping the expression condition fixed:

Expression E
|
+-- Viewpoint 1
+-- Viewpoint 2
+-- Viewpoint 3
|   ...
+-- Viewpoint N

The same structured viewpoint sequence is evaluated across matched identity conditions.

This allows viewpoint to be treated as a controlled intervention rather than an uncontrolled nuisance variable.

---

# Experimental Variables

## Viewpoint

Viewpoint is the primary controlled variable.

The benchmark evaluates model behavior across systematically ordered viewpoints relative to a fixed frontal reference.

The analysis investigates whether recognition performance and representation behavior:

- change gradually;
- degrade abruptly;
- exhibit expression-specific boundaries;
- or show measurable changes before recognition failure.

---

## Expression

The current validated synthesis contains **99 analyzed expression groups**.

Expression-level analysis examines whether different expressions exhibit different sensitivity to viewpoint changes.

The analysis does not assume that all expressions have the same reliability boundary.

Where eligibility criteria are applied, the primary generality claim is based on the predefined eligible expression groups rather than on small or incomplete groups.

---

## Identity

Matched male and female identities are evaluated under controlled expression and viewpoint conditions.

For a matched condition:

Expression: E
Viewpoint: V

Male
|
+------ Same controlled condition ------+
                                       |
Female                                 |

The identity analysis is interpreted as a **controlled identity comparison within this synthetic benchmark**.

It is not treated as evidence for population-level demographic fairness.

---

## Appearance

Additional facial-hair configurations are available for the male identity.

These configurations provide a controlled secondary experiment for examining whether selected appearance changes affect:

- prediction;
- confidence;
- recognition reliability;
- or representation behavior.

Appearance analysis remains descriptive and secondary to the main viewpoint research question.

---

# What Is Measured?

Depending on the analysis stage and model outputs, the benchmark records and analyzes quantities including:

- prediction;
- confidence;
- correctness;
- expression;
- viewpoint;
- identity;
- appearance condition;
- representation-level measurements;
- lead/lag relationships;
- robustness measures;
- permutation-based significance;
- bootstrap estimates.

The central conceptual sequence is:

Controlled Viewpoint Change
        |
        v
Representation Change
        |
        v
Prediction / Confidence Change
        |
        v
Recognition Failure

The objective is to determine whether measurable changes appear **before** recognition failure.

---

# Analysis Pipeline

The project is organized as a sequential 14-stage validation pipeline.

The stages are:

Stage 01 — Build Metadata
        |
        v
Stage 02 — Extract Embeddings
        |
        v
Stage 03 — Analyze Embeddings
        |
        v
Stage 04 — Analyze Embedding Trajectory
        |
        v
Stage 05 — Analyze Ordering
        |
        v
Stage 06 — Ordering Diagnostics
        |
        v
Stage 07 — Statistical Validation
        |
        v
Stage 08 — Early-Warning Analysis
        |
        v
Stage 09 — Early-Warning Horizon Analysis
        |
        v
Stage 10 — Robustness and Sensitivity
        |
        v
Stage 11 — Cross-Expression Validation
        |
        v
Stage 12 — Cross-Identity Validation
        |
        v
Stage 13 — Representation Validation
        |
        v
Stage 14 — Final Synthesis

The stages are intended to be interpreted cumulatively. Later stages build on evidence produced by earlier stages and test whether the observed phenomenon remains consistent under increasingly demanding analyses.

---

# Stage 01 — Build Metadata

The first stage prepares the metadata required by the downstream analyses.

The metadata organizes the benchmark data according to the relevant experimental factors, including:

- identity;
- expression;
- viewpoint;
- image path;
- gender or identity condition;
- sequence information;
- viewpoint direction.

The output of this stage provides the structural information required to identify valid facial sequences.

---

# Stage 02 — Extract Embeddings

The second stage extracts facial representations from the images using the selected representation model.

The resulting embeddings provide the basis for the representation-level analyses performed in later stages.

The extracted representations are used to evaluate:

- representation similarity;
- representation drift;
- representation margins;
- viewpoint-dependent representation changes;
- identity-related representation behavior.

---

# Stage 03 — Analyze Embeddings

The third stage performs the initial analysis of the extracted embeddings.

The purpose of this stage is to characterize the behavior of the learned representation across the available viewpoints and experimental conditions.

The resulting measurements provide the basis for subsequent trajectory and event-ordering analyses.

---

# Stage 04 — Analyze Embedding Trajectory

The fourth stage analyzes how facial representations change as viewpoint moves away from the frontal reference.

The analysis follows facial sequences across viewpoints and evaluates representation behavior relative to the frontal viewpoint.

The main quantities of interest include:

- representation drift;
- changes in representation margins;
- angular distance;
- cosine similarity;
- Euclidean distance;
- path length;
- rate of representation change;
- instability-related measurements where supported by the analysis.

The purpose is to characterize how representation behavior evolves along the controlled viewpoint trajectory.

---

# Stage 05 — Analyze Ordering

The fifth stage investigates the ordering of representation-related events and recognition failure.

A central relationship investigated by the benchmark is:

A → B

where:

A = predefined representation-related event

B = predefined recognition-failure event

The analysis determines whether event A tends to occur before event B as viewpoint changes.

The central quantity is the proportion of valid sequences in which:

A occurs before B

This quantity is referred to as **A-before-B precedence**.

---

# Stage 06 — Ordering Diagnostics

The sixth stage performs additional diagnostics for the observed event ordering.

The purpose is to examine whether the ordering relationship is stable and meaningful before proceeding to statistical validation and early-warning analyses.

These diagnostics provide additional evidence about the structure of the A-before-B relationship.

---

# Stage 07 — Statistical Validation

The seventh stage evaluates the statistical evidence associated with the observed ordering relationship.

The statistical validation is intended to determine whether the observed pattern is distinguishable from an appropriate null expectation.

The analysis uses statistical tests and null comparisons appropriate to the benchmark design.

Permutation-based analyses are used as an additional non-parametric validation of the observed ordering structure.

The results from this stage provide statistical support for subsequent interpretation of the ordering relationship.

---

# Stage 08 — Early-Warning Analysis

The eighth stage evaluates whether the representation-related event can act as an early warning signal for recognition failure.

The main relationship is:

Representation event A
        |
        | lead
        v
Recognition failure B

The analysis quantifies:

- A-before-B precedence;
- A and B event locations;
- lead distance;
- median lead;
- warning behavior;
- warning-model performance;
- permutation-based ordering evidence.

The early-warning analysis is concerned with statistical precedence and predictive utility.

It does not establish a causal relationship between representation change and recognition failure.

---

# Stage 09 — Early-Warning Horizon Analysis

The ninth stage extends the early-warning analysis across multiple viewpoint horizons.

The purpose is to determine how far before recognition failure the warning signal remains detectable.

The analysis evaluates warning rates across multiple tested horizons.

The horizon analysis includes:

- viewpoint horizon;
- number of eligible sequences;
- number of warnings;
- warning rate;
- median lead;
- permutation-based evaluation.

The analysis can therefore determine whether the warning relationship persists at progressively larger viewpoint distances.

The current results indicate that the early-warning relationship weakens as the required warning horizon becomes larger, rather than remaining equally strong at all horizons.

---

# Stage 10 — Robustness and Sensitivity

The tenth stage evaluates whether the observed early-warning relationship remains stable under alternative analysis conditions.

The robustness analysis includes multiple forms of sensitivity analysis, including:

- threshold sensitivity;
- sustained-viewpoint conditions;
- metric sensitivity;
- warning-horizon analysis;
- bootstrap analysis;
- permutation analysis;
- robustness scoring.

The purpose of this stage is to determine whether the main finding depends strongly on a single analytical choice.

A stable result across these analyses provides stronger evidence for robustness.

Robustness does not eliminate all limitations and does not establish causality.

---

# Stage 11 — Cross-Expression Validation

The eleventh stage evaluates whether the observed A-before-B relationship generalizes across facial expression groups.

The analysis includes expression-level summaries and pooled comparisons.

The analysis distinguishes between:

All analyzed expression groups

and:

Primary eligible expression groups

Primary generality claims are based on the predefined eligibility criteria.

Small or otherwise ineligible groups are not treated as equivalent evidence for the primary generalization claim.

The current validated synthesis contains:

- 99 analyzed expression groups;
- a separately defined subset of primary eligible groups used for the primary generality analysis.

The exact eligibility rule and corresponding count should be taken from the final cross-expression analysis output rather than inferred from the total number of expression groups.

The purpose of this stage is to determine whether the A-before-B phenomenon is restricted to a small subset of expressions or is broadly observed across the eligible expression groups.

---

# Stage 12 — Cross-Identity Validation

The twelfth stage evaluates whether the observed relationship generalizes across identity conditions.

The analysis evaluates identity-specific behavior and compares the relevant identity groups across viewpoint directions.

The analysis includes:

- identity-level A-before-B rates;
- median lead;
- mean lead;
- confidence intervals;
- identity-specific horizon analysis;
- bootstrap analysis;
- identity robustness scores;
- paired identity comparisons.

The current synthesis reports the following observed A-before-B rates:

Identity | Side  | A-before-B
-------- | ----- | ----------
Female   | Left  | 93.659%
Female   | Right | 85.366%
Male     | Left  | 96.833%
Male     | Right | 93.665%

These results indicate substantial A-before-B precedence across all four evaluated identity/side conditions in the current benchmark.

The interpretation remains limited to the controlled synthetic identity comparisons.

These results should not be interpreted as evidence for population-level demographic fairness or universal identity generalization.

---

# Stage 13 — Representation Validation

The thirteenth stage performs additional validation directly at the representation level.

The analysis includes:

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

A key comparison evaluates same-expression representation similarity against rival-expression representation similarity.

The current synthesis reports a same-vs-rival representation difference of:

-0.039283

with a permutation result of:

p = 0.000100

This result is interpreted as evidence that same-expression and rival-expression representation behavior differs under the tested experimental setup.

The result does not by itself establish a general theory of visual representation, nor does it establish that representation change causes recognition failure.

---

# Stage 14 — Final Synthesis

The fourteenth stage consolidates the evidence generated by the previous stages.

The final synthesis stage does not rerun the earlier analyses.

Instead, it reads the existing evidence files and produces a consolidated synthesis of the completed analyses.

The final synthesis performs the following tasks:

1. Detect available evidence from the preceding stages.
2. Load existing CSV and JSON outputs.
3. Inspect available file schemas.
4. Detect compatible columns dynamically.
5. Extract relevant metrics.
6. Build synthesis tables.
7. Build final claims.
8. Build publication-oriented summary tables.
9. Generate final plots.
10. Generate the final synthesis report.

The final synthesis is therefore an evidence-integration stage.

It does not re-estimate or modify the underlying analyses.

---

# Dynamic Schema Handling

The final synthesis is designed to inspect the schema of each evidence file before extracting information.

This is important because different analysis stages may use different column names for equivalent quantities.

For example, an A-before-B quantity may appear as:

A_before_B

or:

A_before_B_percent

or:

A_before_B_rate

The final synthesis should therefore avoid assuming that one specific column name will always exist.

Instead, it should:

1. inspect the available columns;
2. identify compatible fields;
3. normalize values when necessary;
4. extract the relevant information;
5. record the source column used.

This makes the final synthesis more robust to schema differences between analysis outputs.

---

# Fixed Benchmark Configuration

The validated benchmark configuration includes the following fixed values:

Fixed A threshold: 13.43702602

Fixed C threshold: 0.0023708

Frontal viewpoint: 107

Expected viewpoints: 215

These parameters define important parts of the benchmark configuration and should remain consistent when reproducing the validated analysis.

They should not be re-estimated by the final synthesis stage.

---

# Evidence Flow

The overall evidence flow can be summarized as:

Input Images
     |
     v
Metadata
     |
     v
Facial Embeddings
     |
     v
Representation Trajectories
     |
     v
Event Detection
     |
     v
Event Ordering
     |
     v
Statistical Validation
     |
     v
Early-Warning Analysis
     |
     v
Warning Horizons
     |
     v
Robustness / Sensitivity
     |
     v
Cross-Expression Validation
     |
     v
Cross-Identity Validation
     |
     v
Representation Validation
     |
     v
Final Synthesis

---

# Main Scientific Question

The pipeline investigates whether a representation-level change can systematically precede recognition failure as viewpoint moves away from the frontal condition.

The primary relationship is:

A → B

where:

A = predefined representation-related event

B = predefined recognition-failure event

The central quantity is the proportion of valid sequences in which:

A occurs before B

This is referred to as:

**A-before-B precedence**

---

# Interpretation of A-before-B

An A-before-B relationship indicates statistical or temporal precedence within the evaluated sequences.

It may support the interpretation that event A contains information that appears before event B.

However:

A-before-B ≠ causal evidence

The benchmark therefore distinguishes between:

- statistical precedence;
- predictive utility;
- representation-level association;
- causal interpretation.

The current pipeline does not establish that representation instability causally produces recognition failure.

---

# Interpretation of Early Warning

An early-warning relationship means that event A is observed before event B often enough to provide potentially useful predictive information.

It does not mean that:

- every failure is predictable;
- every A event is followed by failure;
- the relationship is deterministic;
- the relationship is causal.

The appropriate interpretation is that A may provide an earlier statistical signal associated with subsequent B.

---

# Interpretation of Robustness

A finding is considered more robust when it remains present under multiple analytical conditions.

The robustness stage therefore examines the relationship under different:

- thresholds;
- horizons;
- sustained-viewpoint definitions;
- metrics;
- bootstrap procedures;
- permutation procedures.

Robustness does not eliminate all limitations, but it provides evidence against the result being entirely dependent on a single analytical configuration.

---

# Cross-Expression Interpretation

Cross-expression analysis evaluates whether the observed relationship extends beyond a single facial expression group.

The primary generality claim should be based on expression groups satisfying the predefined eligibility criteria.

Expression groups with insufficient or incomplete data should be treated descriptively rather than being used to support a broad generalization claim.

The fact that expression-level effects may differ is not treated as a contradiction of the main phenomenon.

A more precise interpretation is:

**The precedence phenomenon may generalize broadly across eligible expressions while the magnitude or location of the representation boundary may remain expression-dependent.**

---

# Cross-Identity Interpretation

Cross-identity analysis evaluates whether the observed relationship remains present across different identity conditions.

Consistency across identity groups provides evidence that the observed pattern is not restricted to one evaluated identity condition.

However, the tested identities represent only the available benchmark population and do not automatically establish generalization to all possible populations.

The current identity analysis is therefore interpreted as evidence of within-benchmark consistency rather than demographic generalization.

---

# Representation-Level Interpretation

The representation validation stage provides an additional level of evidence by directly examining the learned representation.

Important representation-level quantities include:

- representation drift;
- expression margins;
- same-expression similarity;
- rival-expression similarity;
- representation boundaries;
- lead/lag relationships.

The purpose is to determine whether the observed early-warning behavior is accompanied by measurable changes in the underlying representation.

The representation-level analysis is therefore complementary to the final classification outcome.

It does not establish that the measured representation geometry is the unique or optimal explanation of recognition failure.

---

# Current Evidence Summary

The completed analyses provide preliminary evidence for a recurring A-before-B relationship under the tested controlled viewpoint conditions.

The evidence has been examined through:

- statistical validation;
- left/right validation;
- permutation validation;
- early-warning analysis;
- horizon analysis;
- robustness and sensitivity analysis;
- cross-expression analysis;
- cross-identity analysis;
- representation-level validation.

The current synthesis contains:

- 427 complete sequences in the representation-level analysis;
- 99 analyzed expression groups;
- multiple validated identity and viewpoint-side conditions.

The cross-identity analysis reports A-before-B rates between approximately 85% and 97% across the four evaluated identity/side conditions.

The representation-level analysis additionally finds a statistically detectable difference between same-expression and rival-expression representation behavior under the tested setup.

These observations motivate further investigation of representation dynamics before recognition failure.

They are treated as empirical evidence within the benchmark, not as a universal theory of visual recognition.

---

# Scientific Position of the Project

The project follows a **discovery-first approach**.

The analysis does not begin by assuming that a new reliability method is necessary.

Instead:

Controlled Dataset
       |
       v
Baseline Evaluation
       |
       v
Statistical Validation
       |
       v
Failure Characterization
       |
       v
Early-Warning Analysis
       |
       v
Robustness Validation
       |
       v
Cross-Expression Analysis
       |
       v
Cross-Identity Analysis
       |
       v
Representation Analysis
       |
       v
Final Synthesis
       |
       v
Research Question
       |
       v
Potential New Method

The objective is to **measure whether the phenomenon exists, characterize it, investigate its representation-level structure, and determine whether it can ultimately be predicted or mitigated.**

A new reliability model is therefore a possible future outcome, not an assumption built into the benchmark.

---

# Broader Research Direction

The current results motivate a broader question beyond facial-expression recognition:

> **Can viewpoint-induced degradation in visual recognition be understood as a structured transition in representation space?**

A more specific formulation is:

> **What representation-level signals emerge before recognition failure under controlled changes in viewpoint, and are these signals sufficiently structured to support early prediction and explanation of failure?**

This reframing connects the benchmark to broader questions in:

- representation learning;
- visual reasoning;
- model reliability;
- failure prediction;
- representation dynamics;
- controlled visual interventions.

The benchmark is therefore intended as a controlled experimental setting for asking a broader scientific question rather than as the final endpoint of the research.

---

# Potential Future Reliability Model

If the observed representation-level signals prove sufficiently robust and transferable, a future stage could investigate a reliability estimator.

Conceptually:

Input Image
     |
     v
Existing Vision Model
     |
     +--------------------+
     |                    |
     v                    v
Expression Prediction   Representation
                              |
                              v
                    Reliability Estimator
                              |
                              v
                       Trust / Do Not Trust

The purpose would be to determine whether representation-level information can provide a warning before a recognition failure occurs.

This is a **future research direction** and is not presented as a completed contribution.

---

# Why a Controlled Synthetic Benchmark?

Real-world facial datasets contain many variables that are difficult to isolate simultaneously, including:

- camera viewpoint;
- identity;
- expression;
- lighting;
- occlusion;
- background;
- image quality;
- camera characteristics;
- facial appearance.

A synthetic controlled environment cannot replace real-world data.

Instead, it provides a complementary experimental setting in which selected variables can be manipulated systematically.

The purpose of this benchmark is therefore:

Controlled Diagnosis
        |
        v
Representation-Level Understanding
        |
        v
Validation on More Realistic Data

Real-world generalization remains a separate scientific question.

---

# Main Contributions of the Current Work

The current project provides four main components.

## 1. Controlled Benchmark

A structured benchmark for evaluating FER behavior under systematic viewpoint variation.

## 2. Reliability Analysis

Quantitative analysis of the relationship between controlled viewpoint changes, prediction behavior, and recognition reliability.

## 3. Failure and Early-Warning Analysis

Investigation of whether measurable signals precede later recognition failure and whether this pattern generalizes across expressions, identities, horizons, and robustness conditions.

## 4. Representation-Level Analysis

Investigation of whether representation-level behavior provides structured evidence associated with the transition toward recognition failure.

Any future reliability-prediction method will be treated as a separate research contribution and will require independent validation.

---

# What This Project Does Not Claim

The project does **not** currently claim that:

- the benchmark is the first viewpoint-controlled FER dataset;
- the observed phenomenon is unprecedented;
- synthetic results automatically generalize to real-world faces;
- two identities are sufficient to establish demographic fairness;
- facial hair explains all appearance-related FER errors;
- existing FER models are generally unreliable;
- representation-level precedence establishes causality;
- the current results constitute a universal theory of visual recognition failure;
- the current representation geometry is the unique explanation of recognition failure;
- early-warning performance automatically implies causal understanding.

Such claims require additional evidence, literature comparison, broader datasets, and/or independent validation.

---

# Limitations

The benchmark has several important limitations.

1. The current benchmark uses a limited number of primary identities.
2. The images are synthetically generated.
3. Synthetic appearance and rendering may differ from real-world facial imagery.
4. The current benchmark does not establish population-level demographic conclusions.
5. Facial-hair variation is limited to the available configurations.
6. Real-world generalization must be evaluated separately.
7. The representation-level results are obtained under the current controlled experimental design.
8. The broader representation-learning interpretation requires validation beyond facial-expression recognition.
9. The current results establish empirical precedence and predictive utility where supported, not causal relationships.
10. Future claims about generality require external datasets and independent replication.

A detailed limitations document is available at:

docs/limitations.md

---

# Reproducibility

The project is organized so that the completed analysis can be inspected and reproduced from documented stages and outputs.

The reproducibility record documents:

- dataset organization;
- experimental conditions;
- model versions;
- model configuration;
- evaluation protocol;
- analysis stages;
- metrics;
- statistical tests;
- permutation procedures;
- bootstrap procedures;
- robustness analyses;
- output files;
- final synthesis.

The reproducibility documentation is available at:

docs/reproducibility.md

The project distinguishes between:

1. source data and benchmark construction;
2. model evaluation;
3. statistical validation;
4. intermediate analysis outputs;
5. final synthesis.

This separation is intended to make it clear which results originate from which stage.

---

# Research Documentation

The repository contains documentation for the scientific reasoning behind the project.

Important documents include:

docs/
├── research_question.md
├── question_for_researcher.md
├── methodology.md
├── analysis_pipeline.md
├── reproducibility.md
└── limitations.md

The most important document for external scientific feedback is:

docs/question_for_researcher.md

It presents the broader research question and asks for guidance on how the representation-level problem could be formulated and experimentally strengthened.

---

# Repository Structure

FER-Reliability-Benchmark/
|
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
|
├── data/
│   ├── README.md
│   └── metadata.csv
|
├── scripts/
│   ├── 01_build_metadata.py
│   ├── 02_extract_embeddings.py
│   ├── 03_analyze_embeddings.py
│   ├── 04_analyze_embeddings_trajectory.py
│   ├── 05_analyze_ordering.py
│   ├── 06_analyze_ordering_diagnostic.py
│   ├── 07_analyze_statistical_validation.py
│   ├── 08_analyze_early_warning.py
│   ├── 09_analyze_early_warning_horizons.py
│   ├── 10_analyze_robustness_sensitivity.py
│   ├── 11_analyze_cross_expression.py
│   ├── 12_analyze_cross_identity.py
│   ├── 13_analyze_representation_validation.py
│   └── 14_analyze_final_synthesis.py
|
├── analysis/
│   ├── 03_analyze_embeddings/
│   ├── 04_analyze_embeddings_trajectory/
│   ├── 05_analyze_ordering/
│   ├── 06_analyze_ordering_diagnostic/
│   ├── 07_analyze_statistical_validation/
│   ├── 08_analyze_early_warning/
│   ├── 09_analyze_early_warning_horizons/
│   ├── 10_analyze_robustness_sensitivity/
│   ├── 11_analyze_cross_expression/
│   ├── 12_analyze_cross_identity/
│   ├── 13_analyze_representation_validation/
│   └── 14_analyze_final_synthesis/
|
├── docs/
│   ├── research_question.md
│   ├── question_for_researcher.md
│   ├── methodology.md
│   ├── analysis_pipeline.md
│   ├── reproducibility.md
│   └── limitations.md
|
├── results/
│   ├── README.md
│   └── key_results.md
|
└── Render_Images_Sequence/
    └── README.md

Large generated datasets and restricted assets are intentionally not required to be distributed through the public repository.

---

# Final Synthesis Outputs

The final synthesis directory contains the consolidated evidence generated by Stage 14.

analysis/final_synthesis/
|
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
├── final_synthesis_report.json
|
└── plots/
    ├── final_identity_comparison.png
    ├── final_representation_lead_lag.png
    ├── final_robustness.png
    └── final_stage_overview.png

The exact set of generated files may depend on the available evidence and the current implementation of the pipeline.

---

# Research Workflow

The complete research workflow can be summarized as:

Controlled Synthetic Benchmark
             |
             v
Systematic Viewpoint Intervention
             |
             v
Baseline FER Evaluation
             |
             v
Metadata and Embedding Construction
             |
             v
Representation and Trajectory Analysis
             |
             v
Event Ordering
             |
             v
Statistical / Permutation Validation
             |
             v
Early-Warning Detection
             |
             v
Horizon Analysis
             |
             v
Robustness / Sensitivity
             |
             v
Cross-Expression Generality
             |
             v
Cross-Identity Comparison
             |
             v
Representation-Level Validation
             |
             v
Final Synthesis
             |
             v
Broader Research Question
             |
             v
External Scientific Feedback
             |
             v
Potential Generalization / New Method

---

# Current Project Status

The completed analysis pipeline contains fourteen documented stages.

The validated evidence includes:

- statistical validation;
- left/right validation;
- permutation validation;
- early-warning analysis;
- horizon analysis;
- robustness and sensitivity analysis;
- cross-expression analysis;
- cross-identity analysis;
- representation-level validation;
- final evidence synthesis.

The current representation-level synthesis contains:

- **427 complete sequences**;
- **99 analyzed expression groups**.

The cross-identity analysis reports substantial A-before-B precedence across all four evaluated identity/side conditions.

The representation-level analysis provides additional evidence that same-expression and rival-expression representation behavior differ under the tested experimental setup.

The project has therefore moved from initial benchmark construction toward the next scientific question:

> **Can the observed early-warning and representation-level behavior be formulated as a broader representation-learning or visual-reasoning problem, and can the phenomenon transfer beyond the current facial-expression setting?**

---

# Research Philosophy

The project follows a discovery-first scientific philosophy:

Measure
   |
   v
Validate
   |
   v
Characterize
   |
   v
Generalize
   |
   v
Formulate the Scientific Question
   |
   v
Propose a Method Only If Justified
   |
   v
Validate Independently

The purpose is not to force the data to support a predetermined method.

The purpose is to determine:

1. whether a reproducible failure pattern exists;
2. how the pattern changes under controlled interventions;
3. whether early-warning signals are measurable;
4. whether representation-level changes characterize the transition;
5. whether the phenomenon generalizes beyond the current benchmark;
6. and only then, whether a new reliability method is scientifically justified.

---

# Collaboration / Research Discussion

The repository is intended to make the current research state understandable without requiring another researcher to reconstruct the entire analysis pipeline from the beginning.

A researcher can begin with:

README.md
     |
     v
docs/research_question.md
     |
     v
docs/analysis_pipeline.md
     |
     v
docs/reproducibility.md
     |
     v
analysis/final_synthesis/
     |
     v
docs/question_for_researcher.md

The goal is to provide enough context to evaluate the scientific question, inspect the evidence, understand the limitations, and suggest a stronger next experimental step.

The project does not assume that collaboration must begin with a new model.

The immediate objective is to obtain scientific feedback on whether the observed representation-level phenomenon constitutes a meaningful research direction and how it should be tested more rigorously.

---

# Dataset Access

The repository contains documentation, analysis code, metadata, and research outputs.

The full rendered image dataset is not currently distributed through this repository.

Access and redistribution depend on the applicable licensing and usage conditions of the software and assets used to generate the renders.

---

# Citation

A formal citation will be added when the benchmark, analysis protocol, and research contribution are finalized.

---

# Project Status

**Status: Completed controlled benchmark analysis and final synthesis; broader research formulation and external scientific validation are the next research stage.**

This is an ongoing research project.

The empirical results, research question, and future experimental direction may be refined as additional evidence and external scientific feedback become available.
