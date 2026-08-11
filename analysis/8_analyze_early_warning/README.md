# Early-Warning Prediction

Main question:

> Does representation drift (A) provide useful early-warning information
> about later prediction failure (B)?

Data:
- 91,805 image rows
- 427 complete expression sequences
- 215 viewpoints per complete sequence
- frontal viewpoint = 107°

Thresholds:
- A = 13.43702602 degrees
- C = 0.00237080
- sustained = 3 viewpoints

The key comparison is:
1. viewpoint-only baseline
2. A-only
3. A + viewpoint
4. A + C + viewpoint
5. trajectory model

Cross-validation is grouped by expression folder to reduce identity/expression
leakage between train and test.

Permutation testing evaluates whether A-before-B ordering is unusually high
under the tested null.

Do not interpret significance as proof of causality.
