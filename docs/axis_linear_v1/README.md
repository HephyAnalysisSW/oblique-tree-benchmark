# Axis-aligned cuts with sparse linear leaves

Converged sources: 4/4; converged frozen-leaf refits: 3/4.

Only checkpoints from converged source stages enter the selected comparison. Latest values from incomplete sources remain explicitly provisional. Same fixed train/validation arrays; no test evaluation.

| Layout | Phase | Selected stage | Continuous slots + feature indices | Learned nonzero scalars | Signal efficiency at 1% bkg [%] | Realizable [%] |
|---|---|---|---|---:|---:|---:|
| 1 × D3 | refit complete | leaf_refit | 143 + 7 | 132 | 50.100 | 50.100 |
| 1 × D4 | refit complete | leaf_refit | 287 + 15 | 228 | 54.740 | 54.740 |
| 3 × D3 | refit complete | regularized | 429 + 21 | 274 | 53.834 | 53.834 |
| 3 × D4 | separated | regularized | 861 + 45 | 479 | 56.322 | 56.322 |

The scan is terminal: all four regularized sources converged. Held refits are explicit outcomes, not converged unpenalized models.

**3 × D4 refit: separated.** The selected result uses the converged regularized source. A separation certificate means the unpenalized logistic objective has a recession direction; the held refit is excluded from selection.

Best axis working point: **3 × D4**, 56.322% efficiency, 479 learned nonzero scalars plus 45 feature indices/direction bits.

| Selected axis layout | All-reachable leaf products | Selected-path leaf products (bound) | Path comparisons (bound) |
|---|---:|---:|---:|
| 1 × D3 | 117 | 16 | 3 |
| 1 × D4 | 197 | 16 | 4 |
| 3 × D3 | 229 | 42 | 9 |
| 3 × D4 | 386 | 35 | 12 |

For context, the frozen oblique linear Compact refit reaches 56.20% with 418 nonzero coefficients, 327 all-reachable products and an 80-product selected-path bound. The axis 3×D4 candidate reaches 56.32% with 479 learned nonzero scalars plus discrete cut choices: it has fewer path products (35), but more all-reachable products (386). This distinction matters for sequential versus fully parallel hardware. No synthesized resource or latency winner is established, and the small working-point difference is not an established accuracy gain.

Axis cuts use a learned feature index, a threshold and a direction bit. Fixed unit coefficients are excluded from learned-scalar counts; their sparse portable representation is reported separately in JSON. Dense continuous slots count one threshold per internal node plus all 17 leaf coefficients. Feature indices, orientation bits and constant-routing flags are separate storage costs. All routing multiplications can be eliminated; affine leaves still require products.

Training searches every feature, both directions and every representable threshold interval using current child-subtree cross-entropies. Updates are accepted only if the true hard loss decreases. Tied child-loss events have no influence on the split. Leaf L1 is 1e-5; split L1 is zero because unit directions are fixed/discrete. CART initialization uses only training rows and seed 42. Each source must pass the existing objective/KKT/routing stability rules. Frozen-support, unpenalized leaf-only refits follow, with unchanged cuts and explicit separation checks.

`configs/axis_linear_v1.json` lists four cases and their exact policies. `scripts/train_axis_tao.py` isolates the subclass from ongoing oblique runs. `scripts/monitor_axis_study.py` submits at most two jobs, continues only clean exit 75 and automatically starts refits from converged terminal parents. Numerical failures require inspection; held refits are not promoted.

CPU timings in the ROC legend are NumPy full-batch API timings, not the separate matched C++ benchmark. [ROC](roc.pdf) · [machine-readable status](status.json).

Reproduce in the pinned training environment with `PYTHONPATH=src:. python scripts/train_axis_tao.py --config configs/axis_linear_v1/case_0.json --index 0`. Use your own checkout/output area. `python scripts/summarize_axis_study.py` refreshes this comparison. No existing frozen handover is replaced.
