# FPGA v3: Compact refit is the implementation default

| Role | Candidate | Trees x depth | Dense / nonzero | Efficiency at 1% background | AUC |
|---|---|---|---:|---:|---:|
| DEFAULT | compact_refit_3t_d4 | 3 x D4 linear | 1581 / 418 | 56.20% | 0.92733 |
| Alternative | highest_efficiency_3t_d4 | 3 x D4 linear | 1581 / 837 | 57.12% | 0.92967 |
| Alternative | single_tree_1t_d4 | 1 x D4 linear | 527 / 452 | 55.63% | 0.92704 |

XGBoost: 100 depth-4 constant-leaf trees; 3,099 scalar slots; efficiency 55.48%, AUC 0.92056.

Validation only; test untouched. Default is chosen for compression, not highest efficiency. Counts refer to standardized inputs. Selected snapshots can precede their converged source stage terminal. Floating point only; no quantization or synthesis claim.
