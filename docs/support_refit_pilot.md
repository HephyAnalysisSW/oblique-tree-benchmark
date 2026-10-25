# Full-data frozen-support pilot

Started 2026-10-25 on CLIP, job **3674490**. This is one parent study, not a scan.

- Configuration: `configs/tao_support_refit_pilot_v1.json`.
- Architecture: **3 × depth-4 hard oblique trees, linear leaves, 16 standardized features**.
- Parent: converged terminal from the split/leaf L1=1e-4 path, pinned by SHA-256 `2dab62f6de3c533829dcd825c695cf84e3991212366ef9afeca0a28c997e9db5`.
- Support: **520 retained coefficients** out of 1,581 dense slots. This is the terminal parent, not its earlier 577-coefficient best-efficiency snapshot.
- Sequence: frozen splits plus unpenalized leaf refit, followed by full unpenalized refit within exactly the same support.
- Data: fixed 531,200 training / 132,800 validation events. Test data are not loaded.
- Resources: one CPU, 12 GiB requested RAM, one-hour Slurm allocations; checkpoint/resubmit on worker exit 75 only. Scientific/numerical failure statuses are held for inspection.

## First measured result

The fixed-routing logistic recession-direction LP used all **531,200 training events** and **189 allowed leaf coefficients**. It returned `no_recession_direction` (HiGHS optimal status), permitting unpenalized fitting to proceed.

| Measurement | Result |
|---|---:|
| Separation-check wall time, including matrix construction | 52.58 seconds |
| Process high-water RSS through the check | 3,563,852 KiB = 3.40 GiB |
| LP score | -2.01e-14 |
| Minimum numerical margin | -2.08e-14 |

RSS includes the loaded dataset and model; it is not incremental LP allocation. This is a numerical check for the current fixed routes. Full refitting can change routes, in which case the check is repeated.

Both stages have now converged on 2026-10-25. All 520 retained coefficients remain nonzero, and the original support hash is unchanged.

| Stage | Efficiency at 1% background | AUC | Training cross-entropy | Passes |
|---|---:|---:|---:|---:|
| Exact sparse terminal parent | 55.58% | 0.92732 | 0.32925146 | — |
| Leaf-only refit | 55.82% | 0.92791 | 0.32795310 | 5 |
| Full refit | **56.49%** | **0.92855** | **0.32649483** | 15 |

Total training compute wall time was 89.0 minutes, including about 9.9 minutes through leaf-only convergence. Final full-stage leaf KKT residual was 7.55e-8, routing change zero, and three consecutive stable passes satisfied the configured numerical convergence rule. This is a 0.91 percentage-point validation efficiency improvement over the exact parent, without increasing the support. It is not a global-optimum or held-out-test claim.

The final saved full checkpoint SHA-256 is `04ea8abde2b3c2546e91777a0b722a406e2e68d39bfb4780ff5af216c18da09d`. The pilot is adopted as the `both10` case in the wider support-refit scan.

## Monitoring and reproducibility

`results/qg_vs_wzt_support_refit_pilot_3t_d4_both1e4_v1_status.json` contains the parent metrics, current stage, constrained leaf KKT residual, support hash/counts, stage histories and stopping diagnostics. Model checkpoints have the same prefix under `models/`.

Per-check timings and peak RSS are appended to `results/support_refit_pilot_separation_resources.jsonl`. Per-allocation process resource records are `results/support_refit_pilot_<job-id>_resources.txt`; console logs are `results/tao-support-pilot-<job-id>.log`.

The measurement wrapper records resources without altering the training objective, support masks, stopping rules or plan. To submit this same explicit study on CLIP:

```bash
sbatch scripts/train_tao_support_refit_pilot.slurm configs/tao_support_refit_pilot_v1.json
```

Do not submit a second copy while one is active. Worker locking prevents concurrent checkpoint writers. Continued jobs keep the same run name and resume policy. Only after this pilot's outcome and resource costs are understood should a wider support-selection/refit scan be chosen.
