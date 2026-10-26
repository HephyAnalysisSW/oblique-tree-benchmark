# Handover status — 26 October 2026

Checked at `2026-10-26T14:22:07.945138+00:00` on CLIP. This is a dated research update; it does not change any frozen FPGA coefficients.

**Dorian should still implement `fpga_handoff/v3/compact_refit_3t_d4/`.** The default remains three depth-4 hard oblique trees with linear leaves, 418 nonzero coefficients (379 reachable) in 1,581 dense slots, validation efficiency **56.20% at 1% background**. The authoritative pointer is [`CURRENT_MODEL.json`](../fpga_handoff/CURRENT_MODEL.json).

## Study progress

| Study | Completed/converged | Current state |
|---|---:|---|
| Architecture / penalty scan | 37/58 converged | 3 running in the controller snapshot; no attention flags |
| L1 continuation paths | 16/16 converged | Complete; no active jobs or attention flags |
| Frozen-support refit scan | 6 full refits converged / 10 cases | All terminal; 2 separated and 2 diagnostic timeouts held |

Architecture controller snapshot: `2026-10-26T14:20:29.658582+00:00`. The L1 path final report is dated `2026-10-25T21:33:09.155959+00:00`. These studies use validation only; test data remain untouched.

## Rechecked research checkpoints

These are saved best-efficiency checkpoints, not necessarily terminal iterates. Metrics below were re-evaluated on the fixed validation arrays and matched the values saved with the coefficients. An unfinished source run is labelled provisional and is not promoted into the FPGA handover.

| Case | Layout | Dense / nonzero | Sparse path MAC bound | Signal efficiency at 1% background | AUC | Source run | Saved / terminal pass |
|---|---|---:|---:|---:|---:|---|---|
| 50 | 5 x D4 linear | 2635 / 1258 | 221 | 57.13% | 0.92907 | Provisional — training | 5 / 5 |
| 57 | 3 x D4 linear | 1581 / 1072 | 156 | 56.55% | 0.92936 | Converged | 9 / 22 |
| 32 | 3 x D4 linear | 1581 / 895 | 145 | 56.39% | 0.92866 | Converged | 8 / 16 |
| 12 | 1 x D4 linear | 527 / 452 | 65 | 55.63% | 0.92704 | Converged | 6 / 18 |
| 13 | 1 x D4 linear | 527 / 452 | 65 | 55.58% | 0.92733 | Converged | 7 / 21 |

The five-tree case 50 reaches 57.13% provisionally, versus 57.12% for the already frozen 837-coefficient higher-efficiency alternative: a point-estimate difference of only 0.014 percentage points, with 1,258 rather than 837 coefficients. It has not converged. This is not a new established improvement or a reason to change Dorian's default.

Compact refit and the existing v3 alternatives remain the implementation comparison. Dense/nonzero counts refer to standardized inputs and include biases. Sparse MAC bounds are analytical scalar counts, not synthesized FPGA latency or resources. No new CPU timing or uncertainty analysis was performed for these research checkpoints.

## Provenance and unchanged handover

All v3 manifest file hashes were checked. The dataset NPZ hash was independently checked before accepting the re-evaluated results.

- Default checkpoint SHA-256: `e89b8923fba66f5df02046e04a1828a8ea4c56f60c1d7b201a2a2f92a0ab00fa`.
- v3 manifest SHA-256: `1fd47d052053e351547e28f150546f351f2032cb4defa79080d39e16985a4b21`.
- Dataset SHA-256: `f5ec6f7ed442010656a904700f8f6421cc34b674a1ec810c5cf6b2de003ed926`.

The [machine-readable snapshot](handover_status_2026-10-26.json) records research checkpoint hashes, immutable CLIP archive paths, configs, epochs, source convergence and the controller job snapshot. Original best-checkpoint paths can change as training continues; use the recorded SHA-pinned copies under `models/research_updates/` for this dated comparison. Those pickles remain on CLIP and are not committed to Git.

[Dorian handover and reproduction instructions](DORIAN_HANDOVER.md) · [v3 model comparison](../fpga_handoff/v3/comparison.md) · [live architecture report](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/hyper_scan_v1_report.md) · [final L1 path report](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/l1_path_v1_report.md).
