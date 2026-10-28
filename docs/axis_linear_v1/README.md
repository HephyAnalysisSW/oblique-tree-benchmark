# Axis-aligned cuts with sparse linear leaves

Converged sources: 0/4; converged frozen-leaf refits: 0/4.

Only checkpoints from converged source stages enter the selected comparison. Latest values from incomplete sources remain explicitly provisional. Same fixed train/validation arrays; no test evaluation.

| Layout | Phase | Selected stage | Continuous slots + feature indices | Learned nonzero scalars | Signal efficiency at 1% bkg [%] | Realizable [%] |
|---|---|---|---|---:|---:|---:|
| 1 × D3 | initial_refit | ineligible | — | — | pending | — |
| 1 × D4 | initial_refit | ineligible | — | — | pending | — |
| 3 × D3 | pending | ineligible | — | — | pending | — |
| 3 × D4 | pending | ineligible | — | — | pending | — |

Axis cuts use a learned feature index, a threshold and a direction bit. Fixed unit coefficients are excluded from learned-scalar counts; their sparse portable representation is reported separately in JSON. Dense continuous slots count one threshold per internal node plus all 17 leaf coefficients. Feature indices, orientation bits and constant-routing flags are separate storage costs. All routing multiplications can be eliminated; affine leaves still require products.

Training searches every feature, both directions and every representable threshold interval using current child-subtree cross-entropies. Updates are accepted only if the true hard loss decreases. Tied child-loss events have no influence on the split. Leaf L1 is 1e-5; split L1 is zero because unit directions are fixed/discrete. CART initialization uses only training rows and seed 42. Each source must pass the existing objective/KKT/routing stability rules. Frozen-support, unpenalized leaf-only refits follow, with unchanged cuts and explicit separation checks.

`configs/axis_linear_v1.json` lists four cases and their exact policies. `scripts/train_axis_tao.py` isolates the subclass from ongoing oblique runs. `scripts/monitor_axis_study.py` submits at most two jobs, continues only clean exit 75 and automatically starts refits from converged terminal parents. Numerical failures require inspection; held refits are not promoted.

CPU timings in the ROC legend are NumPy full-batch API timings, not the separate matched C++ benchmark. [ROC](roc.pdf) · [machine-readable status](status.json).

Reproduce in the pinned training environment with `PYTHONPATH=src:. python scripts/train_axis_tao.py --config configs/axis_linear_v1/case_0.json --index 0`. Use your own checkout/output area. `python scripts/summarize_axis_study.py` refreshes this comparison. No existing frozen handover is replaced.
