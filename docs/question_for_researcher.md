# Question for Researcher Discussion

## Main Question

When only viewpoint changes while facial identity and expression are held fixed, how should changes in the learned facial representation be interpreted, and can representation instability be considered an early-warning signal for impending FER failure?

## Specific Questions

### 1. Representation Stability

If expression and identity remain fixed, to what extent should a learned facial representation remain invariant to viewpoint?

### 2. Representation Drift

When the representation changes systematically as viewpoint changes, how can we distinguish meaningful representation drift from normal viewpoint-dependent variation?

### 3. Early Warning

If representation instability consistently appears before the final FER prediction fails, can this be interpreted as an early-warning signal of reliability degradation?

### 4. Ordering

Does the observed A-before-B precedence provide meaningful evidence that representation-level changes precede classification failure?

### 5. Generalization

Should the same phenomenon be expected to generalize across:

- identities;
- facial expressions;
- left/right viewpoint directions;
- different representation metrics;
- different analysis thresholds?

### 6. Evaluation

What would be the strongest additional experiment needed to demonstrate that representation instability is not merely correlated with viewpoint change, but is specifically informative about impending FER failure?

## Important Scientific Constraint

The current benchmark is designed to establish statistical precedence and predictive utility.

It does not claim causal evidence that representation instability causes classification failure.