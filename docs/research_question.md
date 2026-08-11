# Research Question

## Main Research Question

When facial identity and expression are held fixed and only viewpoint changes, how does the learned facial representation change, and can early representation instability be detected before facial expression recognition fails?

## Motivation

Facial expression recognition (FER) systems may appear reliable when evaluated on standard image-level classification metrics, while their internal representations can change substantially under viewpoint variation.

This benchmark studies whether viewpoint-induced representation changes can be detected before the final expression prediction becomes unreliable.

## Core Hypothesis

A facial representation may exhibit measurable instability before the final classifier produces an incorrect or unreliable expression prediction.

Therefore, representation-level changes may provide an early-warning signal for FER reliability degradation.

## Controlled Factors

The benchmark is designed around controlled sequences in which:

- facial identity is held fixed;
- facial expression is held fixed within a sequence;
- viewpoint changes systematically;
- representation behavior is measured across viewpoints;
- classifier reliability is evaluated along the same viewpoint trajectory.

## Main Questions

1. Does representation instability emerge before classifier failure?
2. Does the A-before-B ordering remain consistent across viewpoint directions?
3. Does the observed pattern generalize across facial expressions?
4. Does the pattern generalize across identities?
5. Does the pattern remain under sensitivity and robustness analyses?
6. Is the phenomenon visible at the representation level independently of final classification accuracy?

## Scientific Interpretation

The benchmark evaluates statistical precedence, association, and predictive utility.

The analysis does **not** establish that representation instability causally produces classifier failure.

A-before-B precedence should therefore be interpreted as an early-warning or predictive relationship rather than causal evidence.