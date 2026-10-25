# Frozen-support refit validation — 2026-10-25

Implementation: `src/tao/support.py`, `scripts/train_tao_support_refit.py`, and optional masked-coordinate hooks in the existing leaf solver, resumable epoch and KKT check.

## Checks performed

The CLIP `tree-benchmark` environment passed **56 tests**, including all 39 pre-existing tests and 17 new support/refit tests. Run:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The recorded run completed in 7.759 seconds. Its full log is `results/support_refit_full_suite.log` on CLIP. New tests use synthetic data; the worker fixture contains no test arrays.

| Property | Verification |
|---|---|
| Exact support selection | Exact zeros, including biases and per-class coefficients; a coefficient of 1e-25 remains allowed |
| Freeze preserves the model | Parent unchanged; initial logits unchanged; support and fixed-routing contracts checked by SHA-256 |
| Support remains fixed | Local leaf fits, joint leaf fits, split proposals, threshold searches and checkpoint restores cannot revive forbidden coordinates |
| Sign freedom | A surviving positive coefficient can become negative |
| Unpenalized leaf optimization | Masked multiclass fit compared with an independently expressed reduced-coordinate BFGS objective, including offsets and nonuniform sample weights |
| Correct stationarity check | KKT residual ignores equality-constrained coordinates and checks every allowed leaf coordinate; weighted normalization checked independently |
| Leaf-only stage | All split coefficients and event routes remain exactly unchanged |
| Full stage | Split weights and bias can move only within the original support; a frozen-zero bias cannot be revived by threshold search |
| Constant routing | Original zero-weight splits keep their bias exactly, including a tiny positive bias that routes right |
| Stage transition | Full refit inherits the original mask; an allowed coefficient temporarily equal to zero is not mistakenly frozen again |
| Resume | Pickle/resume after every node matches uninterrupted epochs, for both leaf modes, binary/multiclass outputs and both stages |
| Finite-optimum check | LP tests cover complete separation, quasi separation, overlapping labels, null forest directions, multiclass logits and zero-weight examples |
| Router separation | A split already realizing every event's lower-loss child is recognized as optimal for the reduced hard objective; auxiliary logistic divergence is not confused with classifier divergence |
| No false success | Separation, inconclusive LP, numerical iteration limits and scheduler interruption never produce a completed terminal model |
| Provenance | Wrong source hashes, unconverged parents, metadata/data mismatches and changed resume policy/implementation are rejected |
| Existing behavior | All pre-existing training, convergence, scheduling and portable-inference tests still pass |

## Real saved-parent compatibility

Without performing any training, both freeze stages were checked against **all 132,800 validation vectors** for each of these converged terminal parents. Logits and every leaf route agreed exactly. No held-out test arrays were loaded.

| Parent | Source SHA-256 | Allowed coefficients |
|---|---|---:|
| Three D4 linear trees, both L1=1e-4, terminal | `2dab62f6de3c533829dcd825c695cf84e3991212366ef9afeca0a28c997e9db5` | 520 |
| One D4 linear tree, both L1=0, terminal | `7494325a77093728afb45a781de5f9886c2f2580b5bb85c44643549d01aa25a4` | 452 |

Machine-readable evidence is `results/support_refit_real_parent_checks.json` on CLIP.

The three-tree terminal has **520**, rather than the **577** nonzeros of its earlier best-efficiency snapshot. The refit must compare against its own exact parent. Combining the earlier snapshot's efficiency with the terminal model's support count would be incorrect.

## Scope and remaining checks before a scan

No production refit scan or full-data refit has been launched. The immutable FPGA handoff has not changed. End-to-end refits were exercised on synthetic fixtures; the real-data checks above validate freezing, not eventual refit performance or large-data LP cost.

The LP checks the existence of a recession direction for the fixed-route unpenalized multinomial logistic problem. It is a numerical diagnostic, not an exact-arithmetic proof. A time limit, solver failure, violated feasibility tolerance or ambiguous positive score gives an inconclusive result and blocks successful completion. Its sparse constraint matrix may still be costly on the full dataset; a representative resource pilot should precede a wide scan.

The leaf certificate applies only to the frozen support and current routes. Full-tree convergence still means exhaustion of configured proposals and stable routing, not global optimality. Training cross-entropy is required to decrease; validation efficiency is measured and may decrease. No coefficient bounds, L2 fallback, epsilon pruning, threshold retuning on test data or physical branch compaction are introduced.
