# Frozen-support refit scan v1

Authorized to run autonomously on 2026-10-25. The purpose is to recover validation signal efficiency at 1% background after L1 has selected a sparse support, and to compare mild versus aggressive support selection fairly.

`configs/tao_support_refit_scan_v1.json` pins ten converged terminal parents by path and SHA-256. Nine have three depth-4 hard oblique trees with linear leaves (1,581 dense slots); the compact case has one such tree (527 slots). All have 16 standardized input features. Exact support includes split/leaf weights and biases.

| Selector | Split / leaf L1 | Retained coefficients | Purpose |
|---|---:|---:|---|
| control | 1e-5 / 1e-5 | 889 | Mild baseline |
| leaf3 | 1e-5 / 3e-5 | 837 | Performance-focused parent |
| split30 | 3e-4 / 1e-5 | 675 | Strong performance with fewer split coefficients |
| split100 | 1e-3 / 1e-5 | 561 | Stronger split sparsity |
| split300 | 3e-3 / 1e-5 | 418 | Aggressive split sparsity |
| both3 | 3e-5 / 3e-5 | 777 | Moderate joint selection |
| both10 | 1e-4 / 1e-4 | 520 | Existing pilot, adopted without duplicate training |
| both30 | 3e-4 / 3e-4 | 394 | Aggressive joint selection |
| both100 | 1e-3 / 1e-3 | 317 | Test whether strong shrinkage hid a useful small support |
| compact1t | 1e-5 / 1e-5 | 403 | Smallest architecture in this scan |

Each independent case performs unpenalized leaf-only fitting with fixed splits, then full fitting within the same immutable support. Zero coordinates cannot regrow. True constant split decisions stay fixed. No L2 fallback, held-out test access, new data splits or altered stopping rules are introduced.

## Execution

`monitor_support_refit_scan.py` maintains at most three pending/running refits, including the adopted pilot. Each allocation requests one CPU, 12 GiB RAM and one hour. The controller resumes only a worker's explicit scheduler-budget exit 75. Submission receipts and a singleton controller lock prevent duplicate jobs. Missing receipts and unexpected exits remain visible for inspection. Separated, inconclusive and stalled outcomes remain labelled as unsuccessful; they do not count as converged.

New cases allow up to 600 seconds for a separation LP, to avoid rejecting a larger support merely because the pilot's smaller LP took about 50 seconds. The pilot retains its original 120-second limit and checkpoint policy. Increasing this diagnostic time limit does not change the fitted model objective or constraints.

The existing architecture scan continues independently with its four-job bound. Plot-staging failures caused by the full shared `/groups` filesystem were inspected and authorized for checkpoint recovery. New publication processes use the home-filesystem cache if the shared cache is full; CERN URLs remain unchanged. Cache uploads deduplicate identical public destinations by file modification time, so an old pending cache cannot overwrite a newer figure.

## Measurements and exact checkpoints

The first pilot leaf-only refit converged in about 10 minutes: efficiency improved from 55.58% to 55.82% with the same 520 coefficients. Full refitting is still active; current progress is available in the live report.

Every new evaluated epoch can be archived as a best-efficiency checkpoint before another node modifies the model. Snapshot score and coefficients are saved together; old history on resume is not mistakenly archived with partially updated weights. Archiving for the already-running pilot begins at its next wrapper continuation; earlier unarchived history is diagnostic only.

The live report labels unfinished results provisional. Final publication re-evaluates every saved candidate on validation data and checks its metrics against the archived record, measures one-thread CPU inference, retains selected files by SHA-256 and produces a ROC comparison and screenshot table. Best snapshots are eligible for final model selection only after the corresponding source stage converges. The frozen FPGA v2 baseline remains immutable during the study.

- [Live report](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_report.md)
- [Live tradeoffs](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_tradeoffs.png)
- Final outputs: `support_refit_scan_v1_final_report.md`, `support_refit_scan_v1_final_table.png`, `support_refit_scan_v1_roc.png` and the machine-readable final summary, under `plots/comparisons/`.

The controller calls `publish_support_refit_results.py` when every case is terminal. Unsupported/separated cases remain in the final table. Model files and diagnostics stay outside Git, with the exact configurations, code and documentation tracked. A final research/handoff recommendation must distinguish a selected saved checkpoint from its source run's numerical convergence.

## Validation

**65 tests pass**: all existing tests plus controller capacity/continuation checks, coefficient-matched checkpoint archiving, final-publication metric verification and publication-cache failover/deduplication. The worker's existing mask, convergence, separation and resume tests remain unchanged.

Final delivery also records coefficient JSON files and validation golden vectors under `research_handoff/support_refit_scan_v1/`, updates a bounded README section, and pushes only those explicit result paths. Delivery is checked end-to-end on a synthetic fixture with Git commands mocked; retained model margins and raw-folded routes must agree before the research candidates are recorded. The v2 FPGA baseline is not replaced automatically.
