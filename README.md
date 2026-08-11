# FER-Reliability-Benchmark

### A Controlled Benchmark for Studying Facial Expression Recognition Reliability Under Systematic Viewpoint and Appearance Changes

---

## Overview

Facial Expression Recognition (FER) systems are commonly evaluated using aggregate recognition performance. However, aggregate accuracy does not fully describe **when** a model becomes unreliable, **under which controlled visual changes** this occurs, or whether the model's confidence remains consistent with its actual performance.

This project introduces a controlled synthetic benchmark designed to study these questions.

The benchmark uses paired male and female digital human identities with matched facial expressions and systematically varied viewpoints. Because the images are rendered under controlled conditions, viewpoint and selected appearance factors can be studied while keeping the underlying facial expression fixed.

The goal is not simply to train another facial-expression recognition model.

Instead, the project investigates whether controlled visual interventions can reveal and characterize **failure boundaries and reliability changes in existing vision models**.

---

# Research Question

## Main Research Question

> **Can controlled 3D viewpoint interventions reveal expression-specific failure boundaries in facial-expression recognition models?**

A related question is:

> **Do FER models remain appropriately calibrated as they move beyond their reliable operating range, or can they become confidently wrong under controlled viewpoint changes?**

These questions will be evaluated empirically using existing FER models before introducing any new method.

---

# Motivation

A model may achieve high overall accuracy while still failing systematically under particular visual conditions.

For example, a model may correctly recognize an expression from a frontal viewpoint but become unreliable as the face rotates away from the camera.

An important question is therefore not only:

> **How accurate is the model?**

but also:

> **At what point does the model become unreliable, and can that transition be measured or predicted?**

The controlled structure of this benchmark makes it possible to study this transition systematically.

---

# Dataset

The current dataset consists of synthetically rendered facial images generated using **DAZ Studio**.

The dataset contains:

* Approximately **95,030 rendered images**
* One male Genesis 8 identity
* One female Genesis 8 identity
* **215 matched facial expressions**
* **215 rendered viewpoints per expression**
* Matched male/female expression configurations
* 1024 × 1024 image resolution
* Additional facial-hair configurations for the male identity

The matched structure allows experiments in which expression and viewpoint can be held constant while identity or selected appearance characteristics are changed.

---

# Controlled Experimental Structure

The central property of the benchmark is its controlled design.

For a given expression:

```text
Expression E
      │
      ├── Viewpoint 1
      ├── Viewpoint 2
      ├── Viewpoint 3
      │      ...
      └── Viewpoint 215
```

The same principle is applied across the matched male and female identities.

This creates a structured experimental setting in which viewpoint can be treated as an intervention rather than an uncontrolled nuisance variable.

---

# Experimental Variables

## Primary Variable

### Viewpoint

The primary experiment systematically evaluates model behavior across the available rendered viewpoints.

The objective is to determine whether model performance changes smoothly, abruptly, or in an expression-dependent manner as viewpoint changes.

---

## Expression

The benchmark contains 215 matched expression configurations.

This enables comparison of viewpoint sensitivity across different expressions.

A central hypothesis is that different expressions may exhibit different patterns of degradation under viewpoint changes.

---

## Identity

Male and female samples are paired by expression and viewpoint.

This enables controlled cross-identity comparisons.

For a matched pair:

```text
Expression: E
Viewpoint: V

Male ─────────────┐
                  ├── Same expression + same viewpoint
Female ───────────┘
```

This does not by itself establish demographic fairness or population-level gender effects. It is intended as a controlled identity comparison within the current synthetic benchmark.

---

## Appearance

The male identity includes additional facial-hair configurations.

These configurations provide a controlled opportunity to investigate whether selected appearance changes can alter model predictions or reliability while the underlying expression remains unchanged.

Appearance analysis is treated as a secondary experiment rather than the primary research question.

---

# What Will Be Measured?

For each evaluated image, the experiments will record, where supported by the model:

* Predicted expression
* Prediction confidence
* Correctness
* Expression identity
* Viewpoint
* Identity
* Appearance configuration

The main evaluation will examine the relationship between:

```text
Viewpoint
     ↓
Model Prediction
     ↓
Confidence
     ↓
Actual Correctness
```

---

# Core Experiments

## Experiment 1 — Dataset Validation

Before evaluating any model, the dataset structure will be validated.

This includes:

* Image counts
* Expression counts
* Viewpoint counts
* Male/female pairing
* Metadata consistency
* Duplicate or corrupted image detection
* Appearance configuration verification

No research conclusions will be drawn before this validation is completed.

---

## Experiment 2 — Baseline FER Evaluation

Existing facial-expression recognition models will be evaluated without modification.

The purpose is to establish baseline behavior before proposing any new method.

For each model we will measure:

* Overall recognition performance
* Per-expression performance
* Per-viewpoint performance
* Confidence
* Correctness

---

## Experiment 3 — Viewpoint Sweep

Each expression will be evaluated across the available viewpoints.

The main analysis will examine:

```text
Accuracy × Viewpoint
```

and

```text
Confidence × Viewpoint
```

The objective is to determine whether viewpoint produces identifiable degradation patterns.

---

## Experiment 4 — Expression-Specific Failure Analysis

Viewpoint-response curves will be compared across expressions.

We will investigate whether some expressions remain robust across a wider range of viewpoints while others exhibit earlier or sharper degradation.

This experiment will determine whether the hypothesized **expression-specific failure boundary** is supported by the data.

---

## Experiment 5 — Reliability and Calibration

Model confidence will be compared against actual correctness.

The key question is:

> Does model confidence remain a useful indicator of correctness as viewpoint changes?

Potential analyses include:

* Confidence versus accuracy
* Calibration error
* Reliability diagrams
* Selective prediction
* Error rate at different confidence thresholds

The exact metrics will be selected after the baseline models and their available outputs are established.

---

## Experiment 6 — Cross-Identity Analysis

Matched male and female samples will be compared under the same expression and viewpoint conditions.

The goal is to determine whether model reliability changes when identity changes while the primary expression and viewpoint are controlled.

This is a controlled identity experiment, not a claim about real-world demographic fairness.

---

## Experiment 7 — Appearance Perturbation

Facial-hair configurations will be evaluated as a controlled appearance variation.

The objective is to investigate whether selected appearance changes can affect:

* Prediction
* Confidence
* Error rate
* Reliability

while the expression and viewpoint remain controlled.

---

# Research Hypotheses

The following hypotheses are provisional and will be evaluated empirically.

### H1 — Viewpoint Sensitivity

FER performance changes systematically as viewpoint changes.

### H2 — Expression-Specific Robustness

Different facial expressions exhibit different sensitivity to viewpoint changes.

### H3 — Reliability-Confidence Mismatch

Model confidence may not decrease at the same rate as actual recognition performance under viewpoint changes.

### H4 — Identity-Dependent Reliability

Matched identities may exhibit different prediction reliability under otherwise controlled conditions.

### H5 — Appearance Sensitivity

Selected appearance changes may alter FER predictions or reliability even when expression and viewpoint remain fixed.

These hypotheses are **not treated as established findings** until supported by experiments.

---

# Potential Research Direction

If the experiments reveal reproducible failure patterns, a subsequent research stage will investigate whether these failures can be **predicted, explained, or mitigated**.

One possible direction is a reliability model that estimates whether a prediction should be trusted:

```text
Input Image
     │
     ▼
Existing Vision Model
     │
     ├── Expression Prediction
     └── Visual Representation
              │
              ▼
      Reliability Estimator
              │
              ▼
       Trust / Do Not Trust
```

This is a future research direction, not a claimed contribution of the current benchmark.

The first objective is to establish whether the underlying phenomenon exists.

---

# Why a Controlled Synthetic Benchmark?

Real-world facial datasets contain many variables that are difficult to isolate simultaneously, including:

* Camera viewpoint
* Identity
* Expression
* Lighting
* Occlusion
* Background
* Image quality
* Camera characteristics
* Facial appearance

A synthetic controlled environment cannot replace real-world data.

However, it can provide a complementary experimental setting in which selected variables can be manipulated systematically.

The purpose of this benchmark is therefore **controlled diagnosis**, followed by validation on more realistic data where appropriate.

---

# Expected Contributions

The project is currently organized around four potential contributions.

### 1. Controlled Benchmark

A structured benchmark for evaluating FER behavior under systematic viewpoint variation.

### 2. Reliability Analysis

Quantitative analysis of the relationship between viewpoint, prediction accuracy, and model confidence.

### 3. Expression-Specific Failure Analysis

Investigation of whether different expressions have different reliability boundaries under viewpoint changes.

### 4. Future Reliability Modeling

If the observed phenomenon is sufficiently strong, development of a method for predicting or mitigating unreliable predictions.

The final contributions will depend on the experimental findings.

---

# What This Project Does Not Claim

This project does **not** currently claim that:

* the benchmark is the first viewpoint-controlled FER dataset;
* the proposed research question is unprecedented;
* synthetic results automatically generalize to real-world faces;
* two identities are sufficient to establish demographic fairness;
* facial hair explains all appearance-related FER errors;
* existing FER models are generally unreliable.

Such claims will only be made if supported by systematic literature review and experimental evidence.

---

# Current Status

**Stage: Dataset preparation and experimental setup**

Current priorities:

* [ ] Verify dataset structure
* [ ] Construct metadata
* [ ] Validate expression/viewpoint pairing
* [ ] Validate image integrity
* [ ] Select baseline FER models
* [ ] Run baseline inference
* [ ] Analyze viewpoint-dependent performance
* [ ] Analyze confidence and calibration
* [ ] Investigate expression-specific failure patterns
* [ ] Perform cross-identity analysis
* [ ] Perform appearance analysis
* [ ] Refine the research question based on empirical findings
* [ ] Determine whether a new reliability method is justified

---

# Repository Structure

```text
FER-Reliability-Benchmark/
│
├── README.md
│
├── data/
│   ├── README.md
│   └── metadata.csv
│
├── experiments/
│   ├── README.md
│   └── baseline/
│
├── results/
│   └── README.md
│
├── docs/
│   ├── research_question.md
│   └── limitations.md
│
└── requirements.txt
```

---

# Dataset Access

The repository contains documentation and metadata for the benchmark.

The full image dataset is not currently distributed through this repository.

Access and redistribution will depend on the applicable licensing and usage conditions of the software and assets used to generate the renders.

---

# Reproducibility

The project aims to make the experimental protocol reproducible.

Experiments will document:

* Model version
* Model configuration
* Dataset subset
* Evaluation protocol
* Random seeds where applicable
* Metrics
* Output format
* Analysis scripts

Experimental results will be added to the repository as the project progresses.

---

# Limitations

The current benchmark has several important limitations:

1. The benchmark currently uses only two primary identities.
2. The images are synthetically generated.
3. Synthetic appearance and rendering may differ from real-world facial imagery.
4. The current benchmark does not establish population-level demographic conclusions.
5. Facial-hair variation is limited to the available configurations.
6. Real-world generalization must be evaluated separately.
7. The final research conclusions depend on the results of the baseline and controlled experiments.

---

# Research Philosophy

The project follows a discovery-first approach:

```text
Controlled Dataset
       ↓
Baseline Models
       ↓
Empirical Observation
       ↓
Failure Characterization
       ↓
Research Question
       ↓
New Method
       ↓
Validation
```

The objective is not to assume that a particular failure mode exists.

The objective is to **measure whether it exists, characterize it, understand its causes, and determine whether it can be predicted or mitigated.**

---

# Citation

A formal citation will be added once the benchmark and research protocol are finalized.

---

# Project Status

This is an ongoing research project.

Results and research questions may change as empirical evidence is collected.
