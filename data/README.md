# Dataset Documentation

## 1. Overview

This directory contains the metadata and documentation for the controlled synthetic facial-expression benchmark used in this project.

The dataset was generated using **DAZ Studio** with digital human characters. Its purpose is to provide a controlled environment for studying how facial-expression recognition models behave when specific visual factors are systematically varied.

The benchmark is designed primarily for **experimental analysis and reliability evaluation**, rather than simply for training a new facial-expression recognition model.

---

## 2. Dataset Generation

All images in the current benchmark were rendered synthetically using DAZ Studio.

The rendering process allows the facial expression, identity, viewpoint, and selected appearance characteristics to be controlled during image generation.

This controlled generation is important because the project aims to study model behavior under specific interventions rather than uncontrolled variation found in ordinary real-world datasets.

---

## 3. Identities

The benchmark currently contains two primary digital human identities:

* Male — Genesis 8
* Female — Genesis 8

The male and female identities are paired at the expression and viewpoint levels for the main controlled subset.

The identities are not intended to represent the diversity of real human populations.

They are used as controlled digital identities for experimental comparison.

---

## 4. Facial Expressions

The benchmark contains **215 matched facial-expression configurations**.

The same expression configuration is available for the male and female identities in the paired subset.

For a matched expression:

```text
Expression E001

Male   → E001
Female → E001
```

The expression identifier should therefore be treated as a controlled experimental variable.

The current project does not assume that these expression configurations correspond perfectly to standardized human emotion categories.

Unless a verified semantic label is available, an expression should be referred to by its dataset identifier rather than assigning an emotion label such as "happy", "sad", or "angry".

---

## 5. Viewpoint Variation

Each expression is rendered across **215 viewpoint configurations**.

The viewpoint dimension is one of the primary experimental variables in this project.

Conceptually, the dataset contains:

```text
Expression
    │
    ├── Viewpoint 001
    ├── Viewpoint 002
    ├── Viewpoint 003
    │       ...
    └── Viewpoint 215
```

This structure allows the same underlying expression to be evaluated under systematically different viewpoints.

The exact geometric interpretation of each viewpoint will be documented from the original DAZ Studio rendering configuration and metadata.

**No viewpoint angle should be inferred from image ordering alone.**

If the original scene files or rendering settings provide exact camera rotations, those values will be recorded in the metadata.

---

## 6. Male/Female Pairing

The main paired subset is designed so that male and female images can be matched by:

* Expression
* Viewpoint

For example:

```text
Male:
Expression E037 + Viewpoint V090

Female:
Expression E037 + Viewpoint V090
```

This controlled pairing allows experiments in which identity is changed while expression and viewpoint remain fixed.

The pairing is one of the important structural properties of the benchmark.

---

## 7. Appearance Variation

The male identity contains additional facial-hair configurations, including beard and/or moustache variations.

These configurations provide a controlled appearance factor that can be investigated separately from the primary viewpoint experiment.

The appearance variation should not initially be interpreted as a general study of facial-hair bias.

Instead, it is treated as a **controlled appearance perturbation** within this particular synthetic benchmark.

The exact number and configuration of facial-hair variants will be documented in the metadata after dataset validation.

---

## 8. Image Resolution

All rendered images in the current benchmark are:

**1024 × 1024 pixels**

The original rendered images should be preserved without unnecessary modification whenever possible.

Any resized or preprocessed versions used for model evaluation should be generated separately and documented as derived data.

---

## 9. Approximate Dataset Size

The current dataset contains approximately:

**95,030 rendered images**

The exact count will be verified programmatically during the dataset-validation stage.

The repository should use the validated count rather than relying on an approximate count in future results.

---

## 10. Dataset Structure

The conceptual structure of the benchmark can be represented as:

```text
                    Dataset
                       │
          ┌────────────┴────────────┐
          │                         │
        Male                      Female
          │                         │
    215 Expressions           215 Expressions
          │                         │
    215 Viewpoints            215 Viewpoints
          │                         │
    Appearance variants       Controlled subset
```

The exact number of images per identity and appearance configuration will be established through metadata validation.

---

## 11. Metadata

Each image will be associated with structured metadata.

The initial metadata schema is:

| Field           | Description                                       |
| --------------- | ------------------------------------------------- |
| `image_id`      | Unique identifier for the image                   |
| `identity`      | Male or female identity                           |
| `expression_id` | Identifier of the facial-expression configuration |
| `viewpoint_id`  | Identifier of the viewpoint configuration         |
| `facial_hair`   | Facial-hair configuration, where applicable       |
| `file_path`     | Relative path to the image                        |

Additional fields may be added after the original rendering configuration is inspected.

Potential future fields include:

* Exact camera rotation
* Camera position
* Rendering configuration
* Appearance configuration
* Scene identifier
* Generation parameters

Only verified information will be added to the metadata.

---

## 12. Controlled Variables

The benchmark is designed around controlled variation.

The main variables are:

### Expression

The underlying facial-expression configuration.

### Viewpoint

The camera/viewpoint configuration.

### Identity

Male or female digital identity.

### Appearance

Selected facial-hair configurations.

The primary experimental design aims to vary one factor while keeping the relevant comparison factors fixed.

---

## 13. Primary Experimental Use

The dataset will initially be used to investigate:

> **How does facial-expression recognition reliability change under systematic viewpoint variation?**

The initial analysis will evaluate existing FER models rather than immediately introducing a new model.

For each image, where model outputs permit, we will record:

* Predicted expression
* Confidence
* Correctness
* Expression ID
* Viewpoint
* Identity
* Appearance configuration

This will allow the project to examine the relationship between viewpoint, model prediction, confidence, and actual correctness.

---

## 14. Secondary Experimental Uses

The controlled structure also permits several secondary analyses.

### 14.1 Expression-Specific Viewpoint Sensitivity

Determine whether different expressions exhibit different performance patterns across viewpoints.

### 14.2 Confidence and Reliability

Determine whether model confidence remains aligned with actual correctness as viewpoint changes.

### 14.3 Cross-Identity Comparison

Compare matched male and female samples under the same expression and viewpoint.

### 14.4 Controlled Appearance Analysis

Investigate whether facial-hair variations influence predictions or confidence while expression and viewpoint are held constant.

These analyses are exploratory until supported by experimental evidence.

---

## 15. What the Dataset Does Not Provide

The dataset currently does **not** provide:

* Human-generated emotion annotations
* Human perception ratings
* Natural-language prompts
* Real-world camera images
* Real human identity diversity
* Population-level demographic labels
* Human behavioral or physiological measurements

Therefore, the dataset should not be interpreted as a direct replacement for real-world FER datasets.

---

## 16. Expression Labels

Because the images were generated in DAZ Studio and do not currently have independently verified human emotion annotations, the project will initially use **expression IDs** rather than assigning semantic emotion labels.

For example:

```text
E001
E002
E003
...
E215
```

If reliable semantic information about the original expression configurations becomes available, it may be added as a separate metadata field.

---

## 17. Dataset Validation

Before any research conclusions are drawn, the dataset will undergo structural validation.

The validation process will check:

1. Total image count
2. Number of identities
3. Number of expressions
4. Number of viewpoints
5. Male/female pairing
6. Missing images
7. Duplicate images
8. Corrupted files
9. Metadata consistency
10. Facial-hair configurations
11. File naming consistency
12. Resolution consistency

The validated dataset statistics will replace the preliminary values currently documented here where appropriate.

---

## 18. Reproducibility

The project will preserve the original dataset organization as much as possible.

Derived files should be separated from the original renders.

For example:

```text
Original Render
      ↓
Metadata
      ↓
Model Preprocessing
      ↓
Model Input
      ↓
Prediction
      ↓
Analysis
```

Preprocessed images should not overwrite the original renders.

---

## 19. Data Access and Distribution

The full image dataset is not currently distributed through this GitHub repository.

The repository will initially contain documentation, metadata, code, experimental protocols, and derived results where permitted.

Before distributing the rendered images publicly, the applicable licensing and usage conditions of DAZ Studio and the underlying digital assets will be reviewed.

No assumption is made here that the rendered images can automatically be redistributed under an open-source license.

---

## 20. Limitations

The dataset has several important limitations.

### Limited Identity Diversity

The primary benchmark uses one male and one female digital identity.

Therefore, findings cannot automatically be generalized to human populations.

### Synthetic Images

The images are rendered rather than captured from real humans.

Differences between synthetic and real facial imagery may affect model behavior.

### Controlled Appearance

The appearance variations represent only the configurations available in the current dataset.

They do not cover the full range of real-world facial appearance.

### Limited Expression Semantics

The current dataset does not contain independently verified human emotion labels.

Expression IDs are therefore used as the primary labels.

### Rendering Bias

Results may depend on the specific digital humans, rendering pipeline, and scene configuration used to generate the benchmark.

These limitations will be considered when interpreting all experimental results.

---

## 21. Intended Role in the Research

The benchmark is not intended to establish that existing FER models are universally unreliable.

Instead, it provides a controlled environment for asking:

> **Under what controlled visual conditions does a model's reliability change?**

The research process therefore follows:

```text
Controlled Dataset
        ↓
Baseline Evaluation
        ↓
Observed Behavior
        ↓
Failure Characterization
        ↓
Research Question
        ↓
Potential Reliability Method
        ↓
Validation
```

The benchmark is therefore a **research instrument**, not only a training dataset.

---

## 22. Dataset Status

**Current status: Dataset documentation and validation**

Next steps:

* [ ] Inspect the actual dataset structure
* [ ] Verify image counts
* [ ] Verify expression count
* [ ] Verify viewpoint count
* [ ] Verify male/female pairing
* [ ] Identify facial-hair configurations
* [ ] Build `metadata.csv`
* [ ] Validate all image files
* [ ] Begin baseline experiments
