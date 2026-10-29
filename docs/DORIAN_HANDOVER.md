# Dorian: final three-family tree comparison and FPGA handover

**Start with [`fpga_handoff/CURRENT_COMPARISON.json`](../fpga_handoff/CURRENT_COMPARISON.json).** It pins the final ten-model paper table, all three family winners and the compressed oblique/linear Compact refit alternative. The immutable [manifest](../fpga_handoff/comparison_v1/manifest.json) and [verification](../fpga_handoff/comparison_v1/verification.json) are the implementation authority for this comparison.

Repository: https://github.com/HephyAnalysisSW/oblique-tree-benchmark (`main`). Reference CLIP checkout: `/users/robert.schoefbeck/CMS/tree_benchmark`. Use your own checkout/output area for new training. The comparison is frozen as of 29 October 2026; all selected TAO entries come from converged source stages; XGBoost is the completed fixed 100-round reference. Earlier selected checkpoints need not themselves be stationary. Historical bundles and `CURRENT_MODEL.json` retain the old Compact default; there is no synthesized hardware winner.

## What to implement and compare

| Role | Candidate ID under `fpga_handoff/comparison_v1/` | Architecture | Learned nonzero scalars | Eff. interp. / strict [%] | C++ µs/event | MUL path / all |
|---|---|---|---:|---:|---:|---:|
| Axis/linear winner | `axis_linear_d4_n3` | 3 × D4, linear leaves | 479 + 45 indices/direction bits | 56.32 / 56.32 | 0.101 | 35 / 386 |
| Oblique/constant winner | `oblique_constant_d4_n5` | 5 × D4, scalar leaves | 1250 | 55.40 / 54.75 | 0.965 | 272 / 1088 |
| Oblique/linear performance winner | `oblique_linear_performance_winner` | 5 × D4, linear leaves | 1275 | 57.73 / 57.73 | 0.412 | 212 / 1096 |
| Compressed oblique/linear alternative | `compact_refit_3t_d4` | 3 × D4, linear leaves | 418 | 56.20 / 56.20 | 0.167 | 78 / 325 |
| XGBoost reference | `xgboost` | 100 × D4, scalar leaves | 3099 + 1499 feature indices | 55.48 / 55.48 | 3.675 | 0 / 0 |

The [full ten-row table](ten_model_comparison_v1/table.md) also includes axis/linear 1×D4 and 1×D3, oblique/constant 3×D4 and 1×D4, and the 837-coefficient oblique/linear leaf refit. Its three smaller/winning representatives per family are a selection of useful scalar budgets, not an exhaustive Pareto frontier. Axis 3×D3 (274 scalars, 53.83%) is omitted because axis 1×D4 is smaller and better at the primary working point, although its AUC is slightly lower.

The arithmetic convention now removes exact-zero and ±1 factors. Earlier v3 and two-model reports counted unit products, so their path/all numbers are slightly larger (e.g. Compact 80/327 versus final 78/325). Neither convention estimates DSPs directly. Sparse index, orientation, connectivity, adder, comparison and preprocessing costs are additional. Computing one path versus all nodes can reverse the relative ranking: axis winner 35/386 versus Compact 78/325. All performance values are validation point estimates; the historical XGBoost reproduction evaluated test data, while these tree studies and emulator runs did not.

## Backend status

- **XGBoost:** existing conifer → CMSSW/HLS C++ chain validated; [setup and precision study](conifer_xgboost_v1/README.md).
- **Oblique/constant:** uses unmodified conifer **1.11** weighted-cut/scalar-leaf backends through the adapter `fpga.leaf_competition.models.conifer_constant`. The older CMSSW **1.7** external is axis-aligned and cannot emulate these weights. The earlier [constant-winner fixed-point check](leaf_competition_v1/README.md) agrees exactly with modern conifer on all validation scores and with generated HLS C++ at the stated 512-vector precision check.
- **Axis/linear and oblique/linear:** floating-point C++ reference is validated; affine leaves still need a conifer HLS backend extension. Axis routing itself requires only feature selection, a directional comparison and a threshold. The earlier independent fixed-point affine-leaf reference is numerical guidance, not a finished HLS backend.

No Vivado/Vitis synthesis, resource utilization, device/clock choice, hardware latency or initiation interval is claimed. The common C++ column is a single-thread sparse traversal reference: 132,800 events, median of nine repeats after warmup, parsing/input conversion/I/O excluded. Optimized native XGBoost separately measured 1.478 µs/event including probability output. The measurement node is a KVM guest reporting Intel Xeon Gold 6138 processors (nominal 2.00 GHz), with GCC 12.3.1 and no fixed-frequency guarantee; the recorded metadata is `docs/ten_model_comparison_v1/cpu_metadata.txt`. Timings from old NumPy or other C++ prototypes must not be mixed with this table. Precision checks cover their named earlier subsets, not all ten final entries.

## Exact portable files and input contract

This new bundle is **standardized-input only**. Every TAO directory contains:

- `standardized/model.json`: exact binary64 coefficients, feature/class order, scaler metadata, routing convention and the strict increasing-signal-margin working point.
- `standardized/model_coefficients.h`: identical hexadecimal coefficients for the standalone C++17 reference.
- `standardized/vectors.tsv` and `expected.tsv`: golden ordered inputs and `signal_margin leaf0 ... leafT-1`.
- `validation_row_indices.tsv` and `labels.tsv`: row identity in the original validation arrays.
- `candidate.json`: selected checkpoint/hash, training/refit configuration, stage eligibility, arithmetic and measured metrics.

The shared `input_contract.json` contains the exact 16-feature order, class order, training mean/scale and observed training ranges. Use `(raw-mean)/scale` exactly once. Raw means high-level features, not detector inputs. Historical v3 retains raw-folded variants for its original models; do not substitute those input spaces or thresholds into this bundle. Developing new raw folding requires fresh routing/working-point checks near boundaries.

TAO inference is `right iff dot(w,x)+bias > 0`, with equality left. Heap root is 0, children `2*j+1` and `2*j+2`, leaf index `node-(2**depth-1)`. Constant leaf rows have one scalar; linear rows have 16 weights plus bias. The signal margin is the **negative sum** of selected leaf outputs, without averaging. Apply a strict `>` threshold calibrated for the exact evaluator; no sigmoid is needed. The default `working_point` is calibrated on the scalar C++ comparison reference. `working_points.portable_numpy` separately calibrates `tao.portable.predict_margin`; its accumulation order can differ by an ulp. Compact's C++ threshold would accept 526 rather than 525 background events in NumPy, so do not interchange these fields. Both correctly calibrated versions respect the 525-event budget. Recheck the operating point after a change of reduction order or precision. Unit axis weights are fixed architecture values, excluded from learned scalar counts but retained in the generic portable arrays.

The XGBoost directory instead contains the pinned `native_model.json` and reconciled `conifer_model.json`, with golden vectors/expected exported margins and native node IDs. XGBoost rounds inputs to binary32, sends equality right for its strict-left cuts, adds its base margin and sums leaf contributions with learning rate absorbed. Its candidate metrics record the strict working point. Common binary64 accumulation differs from native sequential float arithmetic by at most 2.89e-6 on validation; every leaf assignment is exact. Quantization needs a new threshold calibration.

## Verify without training data

```bash
python3 scripts/verify_comparison_handoff.py --compiler c++
# Optional separate report; the frozen report is preserved:
python3 scripts/verify_comparison_handoff.py --compiler c++ --report /tmp/comparison_check.json
```

Python standard library and a C++17 compiler suffice. The frozen report checks **2169 golden vector evaluations**: nine TAO candidates through independent standalone C++ compilation, plus an independent Python exported-XGBoost walk. The full C++ comparison separately verified **every validation leaf assignment for all ten models** against canonical TAO or native XGBoost. `verification.json` identifies each method explicitly. CLIP's native compiler is too old; use EL9:

```bash
scripts/conifer_container.sh bash -c 'source scripts/conifer_runtime.sh && workspaces/python/conifer/bin/python scripts/verify_comparison_handoff.py --compiler g++'
```

## Training and benchmark reproduction

Use the [pinned training requirements](../requirements.txt) and the existing Conda environment `tree-benchmark`; complete historical environment and Compact refit chains remain in [the legacy handover](DORIAN_HANDOVER_V3_LEGACY.md). No GPU is required. Exact pickles/data live on CLIP and are intentionally not in Git. Portable coefficients suffice for firmware implementation.

| Family / role | Training configuration and entry point |
|---|---|
| Axis/linear (four layouts) | `configs/axis_linear_v1/case_0.json` … `case_3.json`; `scripts/train_axis_tao.py --config <case> --index 0`; exact leaf refit parents/plans are `refit_0.json` … `refit_3.json` |
| Oblique/constant selected 5/3/1 trees | `configs/hyper_scan_v1/case_51.json`, `case_33.json`, `case_15.json`; `scripts/train_tao_fully.py --config <case> --index 0` |
| New oblique/linear 5×D4 winner | `configs/hyper_scan_v1/case_50.json`; same resumable worker; CART seed 42, split/leaf L1 `1e-5/1e-5`; selected epoch 21 of a converged source |
| 837-scalar oblique/linear refit | `configs/support_refit_scan_v1/leaf3.json`; completed leaf-only stage retained, later held full stage excluded |
| Compact refit | `configs/support_refit_scan_v1/split300.json`; split/leaf selection L1 `0.003/1e-5`, then unpenalized exact-support refit; selected full epoch 8, source converged at 15 |
| XGBoost | `configs/bdt_qg_vs_wzt_seed42.yaml`; `src/train_bdt.py` |

The exact selected artifacts, source terminal epochs and full configuration are in each `candidate.json`; the manifest hashes them. Refit workers are `scripts/train_tao_support_refit.py`, `scripts/run_support_refit_scan_case.py` and `scripts/support_refit_artifacts.py`. Start exact refit reproduction from the archived hash-pinned converged terminal parent, not from a newly rerun optimizer fit. Named runs must not resume under changed code or policies. Only explicit clean wall-time exit 75 is automatically resumed.

The axis 3×D4 unpenalized refit detected a logistic recession direction and was held. Its regularized source is eligible. All 27 constant-leaf sources converged. The broader architecture scan was 57/58 converged at the table snapshot; unfinished sources contributed no selected entry. This bundle is immutable even if a separate continuing scan later produces another point.

For the measured table, use the [EL9/CMSSW conifer setup](conifer_xgboost_v1/README.md), whose pinned Python environment and JSON headers also compile the common reference:

```bash
scripts/setup_conifer_workspace.sh
scripts/run_ten_model_comparison.sh
python3 paper/build_comparison_assets.py --plot
make -C paper
```

The benchmark script archives selected pickles by hash and validates margins, all leaf assignments and shared working-point metrics before recording timing. Generated data are in ignored `artifacts/ten_model_comparison_v1/`; the reviewed frozen report is [validation.json](ten_model_comparison_v1/validation.json). CPU measurements vary with host; the paper's report preserves the actual measurement snapshot. `scripts/freeze_final_comparison.py` created `comparison_v1` once and refuses to overwrite it.

The [scientific paper](../paper/main.pdf) is self-contained; the [annotated edition](../paper/annotated.pdf) adds optional blue implementation notes and actual tree printouts. Both end with the same ten-model table. [Comparison ROC](../paper/figures/family_roc.pdf) and [scalar/arithmetic tradeoff](../paper/figures/family_tradeoff.pdf) accompany the discussion.
