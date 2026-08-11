# Rendered Image Sequences

This directory contains the rendered facial image sequences used to construct controlled viewpoint trajectories for the benchmark.

## Purpose

The image sequences provide controlled viewpoint variation for studying representation stability and FER reliability.

The benchmark analyzes viewpoint trajectories while attempting to keep facial identity and expression fixed within the corresponding sequences.

## Data Policy

The full rendered image dataset is not necessarily distributed with this repository.

Large image collections should not be committed directly to Git unless explicitly intended for redistribution.

## Reproduction

Users who have access to the required source data and rendering procedure can reproduce the corresponding image sequences using the project configuration and metadata.

## Relationship to the Analysis Pipeline

The rendered sequences provide the visual input underlying:

- metadata construction;
- embedding extraction;
- viewpoint trajectory analysis;
- representation validation;
- early-warning analysis.

The downstream analysis code is located in:

`scripts/`