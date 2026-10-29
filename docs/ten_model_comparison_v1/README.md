# Final ten-model comparison

The final research comparison is pinned by
[`CURRENT_COMPARISON.json`](../../fpga_handoff/CURRENT_COMPARISON.json).
Read [Dorian's handover](../DORIAN_HANDOVER.md) for model files, setup and training
pointers, or the [table](table.md) and [full measured report](validation.json).
The scientific and annotated papers both end with this table.

Three representatives per family span smaller and higher-efficiency scalar
budgets; each includes the highest eligible working point in its family at
the snapshot. Axis/linear uses 3×D4, 1×D4 and 1×D3; axis 3×D3 is dominated
in the primary working-point/scalar-count comparison. Oblique/constant uses
5/3/1 trees at D4. Oblique/linear includes the new 1275-scalar five-tree winner,
the completed 837-scalar leaf refit, and the 418-scalar Compact full refit.
Held refit stages and unfinished source runs contribute no selected entry.
Selected checkpoints may precede a converged source terminal.

All rows use the unchanged 132,800-event validation partition. The recorded [host metadata](cpu_metadata.txt) identifies a KVM guest reporting
Intel Xeon Gold 6138 processors (nominal 2.00 GHz), without a fixed-frequency
guarantee. The common C++
reference is single-thread binary64 sparse path traversal, with XGBoost inputs
rounded to binary32 before timing. Nine repeats follow a warmup; parsing,
input conversion, allocation and I/O are excluded. All validation leaf IDs agree
exactly with canonical TAO/native XGBoost. TAO margins differ by less than 1e-9;
the largest XGBoost difference is 2.89e-6 because its native accumulation differs.
Native optimized XGBoost separately measured 1.478 µs/event including probability
output, versus 3.675 in the common reference. The shared reference is not the
fastest possible implementation of each algorithm.

Scalar counts include biases. Axis fixed unit normals are excluded from learned
scalars, with feature indices and direction bits separate. The table's extra
feature-index column is not a complete accounting of sparse-coordinate,
connectivity or memory-layout overhead. All multiplication counts remove exact
zeros and ±1 factors and prune only input-independent unreachable branches.
`path` is an upper bound along one route per tree; `all` evaluates every
reachable affine function. Bias additions, comparisons, score aggregation,
preprocessing, sigmoid and general constant-multiplier transformations remain
separate. These are not DSP/LUT/BRAM utilization, FPGA latency or initiation
interval measurements.

Reproduce numerical evaluation/timing in the pinned EL9/CMSSW environment:

```bash
scripts/run_ten_model_comparison.sh
```

Data and selected pickle archives are on CLIP. The script records source/model
hashes in ignored `artifacts/ten_model_comparison_v1/`; the reviewed report here
is frozen. Timings vary with host. `fpga/comparison/benchmark.cc` is the common
kernel. `paper/build_comparison_assets.py` renders the frozen scientific table;
`--plot` additionally uses archived validation scores and the hash-checked NPZ.
No new test evaluation or synthesis was performed.

[Research paper](../../paper/main.pdf) · [Implementation edition](../../paper/annotated.pdf)
· [Family ROC](../../paper/figures/family_roc.pdf)
· [Scalar/arithmetic tradeoff](../../paper/figures/family_tradeoff.pdf)

Working points are specific to the accumulation order. The default handover
threshold is calibrated for the scalar C++ reference; TAO JSONs also include
`working_points.portable_numpy`. The [full validation audit](working_point_checks.json)
checks both against the 525-background-event budget. Reproduce with
`scripts/check_comparison_working_points.py` in the pinned environment after
reproducing or restoring the archived C++ scores. Coefficients and paper point
estimates are unchanged; alternate arithmetic needs its own threshold check.
