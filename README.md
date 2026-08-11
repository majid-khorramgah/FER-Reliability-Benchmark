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

A related question is:

> **Do FER models remain appropriately reliable as they move beyond their reliable operating range, or can they become confidently wrong under controlled viewpoint changes?**

The project subsequently investigates a broader representation-level question:

> **Can viewpoint-induced degradation in visual recognition be understood as a structured transition in representation space, and can these representational changes predict recognition failure before the failure occurs?**

This second question is intentionally treated as a research direction rather than as an established conclusion.

---

# Researcher Question

The project has reached a stage where the empirical benchmark and validation pipeline can support a more general research discussion.

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

For a given expression:

```text
Expression E
      |
      +-- Viewpoint 1
      +-- Viewpoint 2
      +-- Viewpoint 3
      |      ...
      +-- Viewpoint N
```

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

The benchmark contains **99 analyzed expression groups** in the current validated synthesis.

Expression-level analysis examines whether different expressions exhibit different sensitivity to viewpoint changes.

The goal is not to assume that all expressions have the same reliability boundary.

---

## Identity

Matched male and female identities are evaluated under controlled expression and viewpoint conditions.

For a matched condition:

```text
Expression: E
Viewpoint: V

Male
  |
  +------ Same controlled condition ------+
                                         |
Female                                   |
```

The identity analysis is intentionally interpreted as a **controlled identity comparison within this synthetic benchmark**.

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

```text
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
```

The objective is to determine whether measurable changes appear **before** recognition failure.

---

# Analysis Pipeline

The project is organized as a staged validation pipeline.

```text
Dataset / Experimental Setup
            |
            v
Stage 1 — Statistical Validation
            |
            v
Stage 2 — Left/Right Validation
            |
            v
Stage 3 — Permutation Validation
            |
            v
Stage 4 — Early Warning
            |
            v
Stage 5 — Early-Warning Horizons
            |
            v
Stage 6 — Robustness / Sensitivity
            |
            v
Stage 7 — Cross Expression
            |
            v
Stage 8 — Cross Identity
            |
            v
Stage 9 — Representation Validation
            |
            v
Stage 10 — Final Synthesis
```

The stages are cumulative. Later analyses are intended to test whether the observed phenomenon remains consistent across increasingly demanding forms of validation.

---

# Stage 1 — Statistical Validation

The statistical-validation stage establishes the statistical basis of the benchmark analysis.

The purpose is to determine whether the observed patterns are sufficiently structured to justify subsequent analysis.

This stage is part of the completed analysis pipeline.

---

# Stage 2 — Left/Right Validation

The left/right validation stage examines whether the observed viewpoint-related pattern is dependent on a particular direction of viewpoint change.

The purpose is to test whether the effect is stable across the relevant sides rather than being an artifact of one direction.

This stage is part of the completed analysis pipeline.

---

# Stage 3 — Permutation Validation

Permutation-based validation provides a non-parametric check of the observed ordering and precedence patterns.

The purpose is to determine whether the observed structure is stronger than would be expected under an appropriate randomized ordering.

This stage is part of the completed analysis pipeline.

---

# Stage 4 — Early Warning

The early-warning analysis investigates whether measurable changes precede recognition failure.

The analysis focuses on whether an ordered sequence contains a transition in which an earlier signal occurs before a later recognition failure.

The analysis is explicitly interpreted as **temporal/statistical precedence**, not causal evidence.

---

# Stage 5 — Early-Warning Horizons

The horizon analysis extends the early-warning question.

Instead of asking only whether a signal occurs before failure, it evaluates how far in advance the signal can precede the later failure event.

This allows the analysis to examine the concept of an **early-warning horizon**.

---

# Stage 6 — Robustness and Sensitivity

The robustness stage evaluates whether the observed pattern remains present under multiple analytical choices.

The completed analysis includes sensitivity and robustness evaluations involving:

- threshold choices;
- sustained-viewpoint criteria;
- metrics;
- horizons;
- bootstrap analysis;
- permutation analysis.

The purpose is to determine whether the main finding depends on a narrow analytical configuration.

---

# Stage 7 — Cross-Expression Analysis

The cross-expression analysis evaluates whether the observed early-warning behavior generalizes across expression groups.

The current synthesis contains:

- **99 analyzed expression groups**;
- **62 primary eligible expression groups** used for the primary generality claim.

Small groups are excluded from the primary generality claim where appropriate.

The objective is to distinguish a broad expression-level pattern from an effect driven by only a small subset of expressions.

---

# Stage 8 — Cross-Identity Analysis

The cross-identity analysis examines whether the observed precedence pattern persists across matched identity and side conditions.

The final synthesis reports the following observed A-before-B rates:

| Identity | Side | A-before-B |
|---|---|---:|
| Female | Left | 93.659% |
| Female | Right | 85.366% |
| Male | Left | 96.833% |
| Male | Right | 93.665% |

These results indicate substantial A-before-B precedence across the validated identity/side conditions in the current benchmark.

The interpretation remains limited to the controlled synthetic identity comparisons.

---

# Stage 9 — Representation Validation

The representation stage investigates whether representation-level behavior provides additional information beyond the final recognition output.

The completed representation analysis includes:

- view-level metrics;
- representation boundaries;
- sequence summaries;
- expression-level representation summaries;
- viewpoint profiles;
- pairwise representation comparisons;
- paired identity comparisons;
- identity tests;
- lead/lag analysis;
- bootstrap analysis;
- false-discovery-rate analysis.

The current synthesis reports a same-vs-rival representation difference of:

```text
-0.039283
```

with a permutation result of:

```text
p = 0.000100
```

This result is interpreted as evidence that representation-level behavior differs between same-expression and rival-expression conditions under the tested experimental setup.

It does **not** by itself establish a general theory of visual representation.

---

# Stage 10 — Final Synthesis

Stage 10 integrates the validated analyses without rerunning or modifying the preceding stages.

The final synthesis currently reports:

```text
Stage 1–9 detected as complete: 9/9
Complete sequences: 427
Expressions: 99
```

The final synthesis generates consolidated tables, claims, plots, stage-status information, and a final report.

The final synthesis output is stored under:

```text
analysis/final_synthesis/
```

The synthesis includes:

```text
final_early_warning_results.csv
final_horizon_results.csv
final_robustness_results.csv
final_expression_results.csv
final_identity_comparison.csv
final_representation_results.csv
final_publication_table.csv
final_key_results.csv
final_claims.csv
final_stage_status.csv
final_report.json
```

The final plots include:

```text
plots/final_identity_comparison.png
plots/final_representation_lead_lag.png
plots/final_robustness.png
plots/final_stage_overview.png
```

---

# Current Scientific Findings

The completed pipeline supports the following evidence-based observations.

## 1. Cross-Identity Precedence

A-before-B precedence is observed across all four validated identity/side conditions.

The observed rates range from:

```text
85.366% to 96.833%
```

---

## 2. Cross-Expression Generality

The early-warning pattern is supported across the primary eligible expression groups.

The current synthesis includes:

```text
99 expression groups
62 primary eligible groups
```

The primary generality claim excludes small groups where appropriate.

---

## 3. Robustness

The early-warning pattern remains present across the completed sensitivity and robustness analyses.

The validated Stage 6 pipeline includes threshold, sustained-viewpoint, metric, horizon, bootstrap, and permutation analyses.

---

## 4. Representation-Level Evidence

Representation-level analysis provides evidence that same-expression representation behavior differs from rival-expression representation behavior.

The current synthesis reports:

```text
Same-vs-rival difference = -0.039283
Permutation p = 0.000100
```

This supports further investigation of representation-level changes preceding recognition failure.

---

## 5. Early-Warning Interpretation

The A-before-B relationship is interpreted as:

- temporal/statistical precedence;
- predictive utility where supported;
- an empirical early-warning relationship.

It is **not interpreted as causal evidence**.

---

# Scientific Position of the Project

The project follows a **discovery-first approach**.

The analysis does not begin by assuming that a new reliability method is necessary.

Instead:

```text
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
```

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

```text
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
```

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

```text
Controlled Diagnosis
        |
        v
Mechanistic / Representation-Level Understanding
        |
        v
Validation on More Realistic Data
```

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
- the current results constitute a universal theory of visual recognition failure.

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

```text
docs/limitations.md
```

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

```text
docs/reproducibility.md
```

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

```text
docs/
├── research_question.md
├── question_for_researcher.md
├── methodology.md
├── analysis_pipeline.md
├── reproducibility.md
└── limitations.md
```

The most important document for external scientific feedback is:

```text
docs/question_for_researcher.md
```

It presents the broader research question and asks for guidance on how the representation-level problem could be formulated and experimentally strengthened.

---

# Repository Structure

```text
D:\1405\FER-Reliability-Benchmark
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── data
│   ├── README.md
│   └── metadata.csv
│
├── scripts
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
│
├── analysis
│   ├── 03_analyze_embeddings
│   ├── 04_analyze_embeddings_trajectory
│   ├── 05_analyze_ordering
│   ├── 06_analyze_ordering_diagnostic
│   ├── 07_analyze_statistical_validation
│   ├── 08_analyze_early_warning
│   ├── 09_analyze_early_warning_horizons
│   ├── 10_analyze_robustness_sensitivity
│   ├── 11_analyze_cross_expression
│   ├── 12_analyze_cross_identity
│   ├── 13_analyze_representation_validation
│   └── 14_analyze_final_synthesis
│
├── docs
│   ├── research_question.md
│   ├── question_for_researcher.md
│   ├── methodology.md
│   ├── analysis_pipeline.md
│   └── limitations.md
│
├── results
│   ├── README.md
│   └── key_results.md
│
└── Render_Images_Sequence
    └── README.md
```

Large generated datasets and restricted assets are intentionally not required to be distributed through the public repository.

---

# Final Synthesis Outputs

The final synthesis directory contains the consolidated evidence generated by Stage 10.

```text
analysis/final_synthesis/
│
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
│
└── plots/
    ├── final_identity_comparison.png
    ├── final_representation_lead_lag.png
    ├── final_robustness.png
    └── final_stage_overview.png
```

These files provide a compact record of the final validated synthesis.

---

# Research Workflow

The complete research workflow can be summarized as:

```text
Controlled Synthetic Benchmark
             |
             v
Systematic Viewpoint Intervention
             |
             v
Baseline FER Evaluation
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
```

---

# Current Project Status

The complete Stage 1–10 analysis pipeline has been executed.

Current synthesis status:

```text
Stage 1–9: Complete
Stage 10: Complete
Complete sequences: 427
Analyzed expression groups: 99
Primary expression groups: 62
```

The final synthesis integrates the completed statistical, early-warning, robustness, expression, identity, and representation analyses.

The project has therefore moved from initial benchmark construction toward the next scientific question:

> **Can the observed early-warning and representation-level behavior be formulated as a broader representation-learning or visual-reasoning problem, and can the phenomenon transfer beyond the current facial-expression setting?**

---

# Research Philosophy

The project follows a discovery-first scientific philosophy:

```text
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
```

The purpose is not to force the data to support a predetermined method.

The purpose is to determine:

1. whether a reproducible failure pattern exists;
2. how the pattern changes under controlled interventions;
3. whether early-warning signals are measurable;
4. whether representation-level changes explain or characterize the transition;
5. whether the phenomenon generalizes beyond the current benchmark;
6. and only then, whether a new reliability method is scientifically justified.

---

# Collaboration / Research Discussion

The repository is intended to make the current research state understandable without requiring another researcher to reconstruct the entire analysis pipeline from the beginning.

A researcher can begin with:

```text
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
```

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
