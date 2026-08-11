# Methodology

## Overview

The FER Reliability Benchmark evaluates the reliability of facial expression recognition representations under controlled viewpoint variation.

The analysis proceeds from metadata construction and embedding extraction to ordering analysis, statistical validation, early-warning analysis, robustness testing, cross-expression and cross-identity validation, representation-level validation, and final synthesis.

## Experimental Principle

The central experimental principle is to vary viewpoint while controlling facial identity and expression as much as possible.

This allows representation changes along the viewpoint trajectory to be studied independently of intentional expression changes.

## Fixed Configuration

The benchmark uses the fixed configuration established by the analysis pipeline:

- Frontal viewpoint: 107
- Expected viewpoints: 215
- Fixed A threshold: 13.43702602
- Fixed C threshold: 0.0023708

These parameters are treated as configuration values for the validated analysis pipeline and are not re-estimated by Stage 14.

## Representation Analysis

Representation behavior is evaluated using embedding-derived metrics including:

- angular distance from the frontal representation;
- cosine similarity;
- expression margin;
- prototype similarity;
- representation accuracy;
- classifier failure indicators.

## Ordering Analysis

The benchmark evaluates the ordering of representation-related events relative to classifier failure.

The A-before-B event is used as the primary precedence relationship.

A positive lead indicates that event A occurs before event B along the viewpoint trajectory.

## Early-Warning Analysis

The early-warning analysis evaluates whether A occurs before B and quantifies the associated lead.

The analysis is extended across different warning horizons to determine how far before classifier failure the warning signal can be detected.

## Statistical Validation

The pipeline includes statistical validation and permutation-based analyses to determine whether the observed ordering is distinguishable from an appropriate null distribution.

## Robustness

Robustness is assessed through sensitivity analyses, bootstrap procedures, permutation procedures, alternative thresholds, and horizon-based analyses.

## Cross-Expression Validation

The observed pattern is evaluated across expression groups to determine whether the phenomenon is specific to a small subset of expressions or generalizes across primary eligible expression groups.

## Cross-Identity Validation

The analysis compares identity conditions, including female and male groups, and evaluates whether the observed precedence remains consistent across identity and viewpoint direction.

## Representation-Level Validation

Representation-level analyses compare same-expression and rival-expression representation behavior and evaluate lead/lag relationships directly in the learned representation space.

## Final Synthesis

Stage 14 reads existing evidence from the previous stages and produces a consolidated synthesis.

Stage 14 does not rerun or re-estimate the previous analyses.