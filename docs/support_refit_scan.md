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

`monitor_support_refit_scan.py` maintains at most five pending/running refits, including the adopted pilot (initially three; expanded after all cases started). Each allocation requests one CPU, 12 GiB RAM and one hour. The controller resumes only a worker's explicit scheduler-budget exit 75. Submission receipts and a singleton controller lock prevent duplicate jobs. Allocations rotate across cases so a long continuation cannot indefinitely postpone an unstarted support budget. Missing receipts and unexpected exits remain visible for inspection. Separated, inconclusive and stalled outcomes remain labelled as unsuccessful; they do not count as converged.

New cases allow up to 600 seconds for a separation LP, to avoid rejecting a larger support merely because the pilot's smaller LP took about 50 seconds. The pilot retains its original 120-second limit and checkpoint policy. Increasing this diagnostic time limit does not change the fitted model objective or constraints.

The existing architecture scan continues independently with its four-job bound. Plot-staging failures caused by the full shared `/groups` filesystem were inspected and authorized for checkpoint recovery. New publication processes use the home-filesystem cache if the shared cache is full; CERN URLs remain unchanged. Cache uploads deduplicate identical public destinations by file modification time, so an old pending cache cannot overwrite a newer figure.

## Measurements and exact checkpoints

The first pilot leaf-only refit converged in about 10 minutes: efficiency improved from 55.58% to 55.82% with the same 520 coefficients. The pilot full stage subsequently converged at **56.49%** after 15 passes, with the same 520 coefficients. Other cases continue; current progress is available in the live report.

Every new evaluated epoch can be archived as a best-efficiency checkpoint before another node modifies the model. Snapshot score and coefficients are saved together; old history on resume is not mistakenly archived with partially updated weights. Archiving for the already-running pilot begins at its next wrapper continuation; earlier unarchived history is diagnostic only.

The live report labels unfinished results provisional. Final publication re-evaluates every saved candidate on validation data and checks its metrics against the archived record, measures one-thread CPU inference, retains selected files by SHA-256 and produces a ROC comparison and screenshot table. Best snapshots are eligible for final model selection only after the corresponding source stage converges. Original converged parents, completed leaf-only stages and the previously established one-tree/high-efficiency references remain eligible; a harmful refit cannot displace a better established checkpoint merely because it is newer. The frozen FPGA v2 baseline remains immutable during the study.

- [Live report](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_report.md)
- [Live tradeoffs](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_tradeoffs.png)
- Final outputs: `support_refit_scan_v1_final_report.md`, `support_refit_scan_v1_final_table.png`, `support_refit_scan_v1_roc.png` and the machine-readable final summary, under `plots/comparisons/`.

The controller calls `publish_support_refit_results.py` when every case is terminal. Unsupported/separated cases remain in the final table. Model files and diagnostics stay outside Git, with the exact configurations, code and documentation tracked. A final research/handoff recommendation must distinguish a selected saved checkpoint from its source run's numerical convergence.

## Validation

**74 tests pass**: all existing tests plus controller capacity/continuation checks, coefficient-matched checkpoint archiving, final-publication metric verification and publication-cache failover/deduplication. The worker's existing mask, convergence, separation and resume tests remain unchanged.

Final delivery also records coefficient JSON files and validation golden vectors under `research_handoff/support_refit_scan_v1/`, updates a bounded README section, and pushes only those explicit result paths. Delivery is checked end-to-end on a synthetic fixture with Git commands mocked; retained model margins and raw-folded routes must agree before the research candidates are recorded. The v2 FPGA baseline is not replaced automatically.

## Statistical and interchange checks

Final recommendations retain original parents and established references if a refit harms performance. A snapshot from an unsuccessful full stage remains diagnostic, with an asterisk, and is excluded from selection. Recommendations distinguish highest efficiency, fewest nonzero coefficients matching XGBoost, and fewest sparse path multiplications matching XGBoost; parameter count alone does not determine inference cost.

The final report includes 1,000-replicate paired, class-stratified validation bootstrap intervals at the same interpolated ROC working point, seed 20261025. Tied scores and zero-weight resampled rows are checked against scikit-learn's weighted ROC. Intervals condition on the fitted/selected models and do not correct for training seeds or validation selection. The primary selection metric remains the predeclared point efficiency.

The portable research candidates include standardized and raw-folded coefficient JSONs and validation vectors selected randomly and near the working-point/split boundaries. Independent standardized inference is checked against native margins on all validation rows; raw-folded routes must agree. Each input-space JSON has a separately calibrated strict margin cut accepting at most floor(1% of validation background), distinct from the interpolated ROC metric. This avoids treating a last-bit raw-folding difference as a different operational background budget. The current closed-model probe passed all 132,800 rows for the pilot and both established references: native margin error zero, raw-folding error below 4.3e-14, with exact leaf-route agreement.

Parameter and operation counts in the comparison refer to standardized-input models. Folding preprocessing into raw-input coefficients can change bias values and their zero counts; feature-weight sparsity is preserved by the nonzero feature scales. FPGA resource and latency measurements require the chosen input space and an actual implementation.

## Explicit auxiliary solver budget continuation

A control-run auxiliary router repeatedly reached its 800-iteration limit, including rejected proposals. A larger auxiliary budget can be applied with `scripts/amend_support_solver_budget.py` after a clean scheduler exit. Pause scan submissions (`concurrency=0`), wait for the worker lock to be released, then invoke the tool with the case config, a strictly larger `--split-max-iter`, and a measured `--reason`. Restore scan concurrency afterwards. This is an explicit checkpoint-policy migration, not an implicit resume override.

The tool retains the original checkpoint by SHA-256 and records old/new policies, implementation and data hashes, coefficient and support hashes, the completed-history count and the reason. It changes only the auxiliary split iteration budget; coefficients, masks, history, objective and stopping tolerances are preserved. Active workers, held/completed runs, policy mismatches and smaller budgets are rejected. An interrupted migration fails closed at the worker's exact policy check; the archive and receipt permit recovery. The final summary and report disclose any applied amendments. Tests check bit-identical predictions and coefficients, active-worker exclusion, archive retention, policy guards and an actual resumed training run.

A partially completed epoch retains its original per-node fit records across a budget amendment. The larger budget applies to subsequent auxiliary solves; an already evaluated 800-iteration proposal is not relabelled. A direct wrapped-solver test verifies that the next solve receives the new budget.
