# Imported GOLLUM TAO sources

Repository: https://github.com/HephyAnalysisSW/GOLLUM

- `devel_rs/TAO`: exact snapshot of devel-rs, commit 9a4e2dea5510116798daa6609c183fd55a37e800 (2025-07-16).
- `master/TAO` and `master/ML/TAO`: exact snapshot of master, commit f95bd6ba90fbe81397ce43bc697ba6a88b271e2b.
- `syncer_original.py`: exact working-tree copy of common/syncer.py from the source checkout on 2026-10-22.

Both source branches were read using git archive; the original checkout remained on sbi-pdf-paper and was not modified.
Runtime hard-tree copies in src/tao remove unused plotting imports, use relative imports, and filter zero-weight routing ties before fitting binary split classifiers. Original algorithm snapshots above remain unchanged.
Soft trees and archived drivers are preserved here; they have not been ported or run in the benchmark environment.

An optional newton_l2 split fit uses L2-regularized logistic regression with sklearn newton-cholesky instead of the original L1/liblinear surrogate solver. It keeps hard oblique routing and alternating forest updates. Runs explicitly record this solver variant.

## Current benchmark implementation

`src/tao/hard.py` implements the reviewed hard-tree training path: linear or constant vector-logit leaves, categorical cross-entropy with fixed other-tree offsets, exact L1 leaf parameterization, binary loss-difference routing, full-training joint refit, and actual-objective update acceptance. Original snapshots above are not changed. It uses one identity transform on fixed standardized data; the raw-statistics transform explicitly takes sqrt(variance). The original scalar residual prototypes remain only for historical comparison.

The current default split fit is still L1/liblinear. The optional L-BFGS-B split fit minimizes the same normalized weighted logistic + L1 objective; it is distinct from the retired scalar prototype's L2 experiment. Explicit regularization normalization and K-1 reference-class logits are recorded in the README and run configs. Measurement snapshots under results/tao_steps retain the implementations corresponding to the step reports.
