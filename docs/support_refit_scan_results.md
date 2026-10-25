### Support-refit scan results

Completed **6/10** full refits; **4** cases have explicitly held outcomes. All results use validation; test data are untouched.

- Highest efficiency: **3 x D4 linear**, 837 nonzero coefficients, **57.12%** signal efficiency at 1% background, AUC 0.92967. Selector `leaf3`; saved `leaf_only` checkpoint from a converged source stage. Support selected with split/leaf L1 **1e-05/3e-05**. Fixed sparse splits; surviving linear-leaf coefficients refitted without L1.
- Fewest coefficients matching XGBoost: **3 x D4 linear**, 418 nonzero coefficients, **56.20%** signal efficiency at 1% background, AUC 0.92733. Selector `split300`; saved `full_best` checkpoint from a converged source stage. Support selected with split/leaf L1 **0.003/1e-05**. Surviving split and linear-leaf coefficients refitted without L1 inside immutable masks.
- Fewest sparse path multiplications matching XGBoost: **1 x D4 linear**, 452 nonzero coefficients, **55.63%** signal efficiency at 1% background, AUC 0.92704. Selector `previous_1t_no_l1`; saved `previous_best` checkpoint from a converged source stage. Original trained reference checkpoint.

XGBoost reference: **55.48%** at the same working point.

[Final table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_final_report.md) · [ROC](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_roc.png) · [Screenshot table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_recommendations.png).

See [full results](support_refit_scan_results.md) and the coefficient-pinned [research candidates](../research_handoff/support_refit_scan_v1/selection.json). The FPGA v2 implementation baseline remains unchanged; these candidates are floating-point references, not quantized or synthesized implementations.

Winner minus XGBoost: +1.63 percentage points; paired validation bootstrap 95% interval [+0.65, +2.55]. This conditions on selected trained models and does not correct for validation selection or training-seed variation.

# Frozen-support refit scan: coefficient-matched final comparison

2026-10-25T22:05:22.807796+00:00

| Selector | Architecture | Split/leaf L1 | Support | Parent eff. | Leaf refit | Full terminal | Best saved full | AUC | CPU us/event | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| both10 | 3 x D4 linear | 0.0001/0.0001 | 520 | 55.58% | 55.82% | 56.49% | 56.49% | 0.92855 | 1.045 | complete |
| split30 | 3 x D4 linear | 0.0003/1e-05 | 675 | 56.98% | 56.74% | — | 56.71% * | 0.92909 | 1.066 | inconclusive |
| leaf3 | 3 x D4 linear | 1e-05/3e-05 | 837 | 56.91% | 57.12% | — | 57.24% * | 0.92990 | 1.115 | inconclusive |
| both30 | 3 x D4 linear | 0.0003/0.0003 | 394 | 54.48% | 54.85% | — | 54.78% * | 0.92572 | 0.998 | separated |
| compact1t | 1 x D4 linear | 1e-05/1e-05 | 403 | 55.20% | 55.06% | 55.02% | 55.08% | 0.92703 | 0.460 | complete |
| split100 | 3 x D4 linear | 0.001/1e-05 | 561 | 56.06% | 56.32% | 56.28% | 56.46% | 0.92836 | 1.083 | complete |
| both3 | 3 x D4 linear | 3e-05/3e-05 | 777 | 56.62% | 56.75% | 56.66% | 56.66% | 0.92914 | 1.151 | complete |
| control | 3 x D4 linear | 1e-05/1e-05 | 889 | 56.40% | 56.45% | 56.72% | 56.74% | 0.92941 | 1.281 | complete |
| split300 | 3 x D4 linear | 0.003/1e-05 | 418 | 54.59% | 54.81% | 55.98% | 56.20% | 0.92733 | 0.974 | complete |
| both100 | 3 x D4 linear | 0.001/0.001 | 317 | 51.27% | 52.52% | — | 52.94% * | 0.92092 | 1.021 | separated |

Selected tradeoffs (parent and established reference checkpoints remain eligible):

| Role | Architecture | Dense slots | Nonzeros | Reachable | MAC bound | Eff. at 1% | AUC | CPU us/event |
|---|---|---|---|---|---|---|---|---|
| XGBoost reference | 100 x D4 constant | 3099 | — | — | — | 55.48% | 0.92056 | 1.083 |
| Highest efficiency | 3 x D4 linear | 1581 | 837 | 837 | 163 | 57.12% | 0.92967 | 1.131 |
| Fewest coefficients | 3 x D4 linear | 1581 | 418 | 379 | 80 | 56.20% | 0.92733 | 0.974 |
| Fewest path MACs | 1 x D4 linear | 527 | 452 | 452 | 65 | 55.63% | 0.92704 | 0.412 |

Paired class-stratified validation bootstrap, 1,000 replicates (95% percentile intervals):

| Candidate | Efficiency [95% interval] | Difference vs XGBoost, percentage points [95% interval] |
|---|---:|---:|
| XGBoost | 55.48% [54.28%, 56.80%] | +0.00 [+0.00, +0.00] |
| winner | 57.12% [55.86%, 58.40%] | +1.63 [+0.65, +2.55] |
| compact_match_xgb | 56.20% [54.67%, 57.51%] | +0.72 [-0.43, +1.52] |
| compact_path_match_xgb | 55.63% [54.01%, 56.92%] | +0.15 [-1.08, +0.97] |
| winner_parent | 56.91% [55.73%, 58.03%] | +1.42 [+0.45, +2.29] |

winner_vs_parent: +0.21 percentage points [-0.23, +0.64].

Conditional validation sampling uncertainty only; no correction for checkpoint/hyperparameter selection or training-seed variation. Test not evaluated.

Recorded auxiliary solver budget continuations (all stopping tolerances unchanged):

- `control`: split iterations 800 -> 4000, after 17 evaluated history rows. Auxiliary split 37 repeatedly exhausted 800 iterations, including rejected unchanged proposals in full passes 7-12. The latest pass had zero routing changes and leaf KKT 1.55e-7. Increase solver work without changing objective, masks or stopping tolerances. Coefficients were bit-identical at the transition; original checkpoint SHA-256 `cac7cbae6176aff1e9c76e3e30748e3f14d1bde5d01f80763ee3d132bf97dc5d` is retained in the policy archive.

Previously established reference checkpoints (from completed runs):

| Model | Architecture | Nonzeros | Efficiency at 1% background | AUC |
|---|---|---:|---:|---:|
| previous_1t_no_l1 | 1 x D4 linear | 452 | 55.63% | 0.92704 |
| previous_3t_peak | 3 x D4 linear | 827 | 57.10% | 0.92896 |

Parent and refit values use the same support. Recommendations consider converged parents, completed refit stages, their archived best snapshots and the established references above. A selected snapshot can precede its source stage terminal. An asterisk marks a diagnostic snapshot from an unsuccessful full stage; it is not eligible for recommendation. Held full refits are not treated as converged; a completed leaf-only stage remains a valid candidate. Parameter and operation counts refer to standardized-input models; folding preprocessing into raw-input coefficients can change bias values and their zero counts. These counts are not synthesized FPGA resource or latency measurements. Test data remain untouched.

![Selected table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_recommendations.png)

![Full table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_final_table.png)

![ROC](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_roc.png)
