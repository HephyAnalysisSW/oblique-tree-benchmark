# Frozen XGBoost -> conifer -> CMSSW/HLS C++ emulation

All sources live in this repository. Generated CMSSW/Python workspaces are under
`workspaces/`; generated emulator inputs, scores and HLS projects are under
`artifacts/conifer/`. Both are ignored by Git. Training code, trained models,
the dataset split and `fpga_handoff/CURRENT_MODEL.json` are unchanged.

## Environments

Run on CLIP from `/users/robert.schoefbeck/CMS/tree_benchmark`:

```bash
scripts/setup_conifer_workspace.sh
scripts/submit_xgboost_conifer.sh --output artifacts/conifer/xgboost_v1
```

For an interactive smoke test:

```bash
scripts/run_xgboost_conifer.sh --max-events 2000 \
  --output artifacts/conifer/xgboost_smoke
```

The scripts enter `cmssw-el9`, create `workspaces/cmssw/CMSSW_15_1_2` with
`SCRAM_ARCH=el9_amd64_gcc12`, and use its compiler, JSON and arbitrary-precision
headers. `cmssw/TreeBenchmark/ConiferValidation` is copied into that generated
workspace and built by SCRAM. Its executable uses the **actual CMSSW conifer
1.7 external**, rather than an independent replacement implementation.

Python conversion uses **conifer 1.11 and CPU XGBoost 3.2.0** in a separate
`workspaces/python/conifer` virtual environment. The existing training Conda
interpreter provides Python 3.10; no packages are added to that training
environment. Direct dependency pins are in
`configs/environments/conifer-emulation-requirements.txt`; setup installs the
complete `conifer-emulation-lock.txt`. Each report also retains its installed
package list. Adjust the base interpreter
path in `configs/environments/cmssw-conifer.env` for another account.

The two conifer versions deliberately exercise a useful compatibility boundary.
For this **axis-aligned, constant-leaf XGBoost** model they must agree.
The CMSSW 1.7 executable rejects oblique weight vectors, since that older header
would otherwise silently ignore them. It is not the emulator for our TAO trees.
Those require a compatible modern interface and affine-leaf extension later.

## Conversion contract and gates

The source is the frozen native XGBoost JSON at
`fpga_handoff/v3/xgboost_reference/model.json`, whose hash is checked against the
v3 manifest. The NPZ hash is checked too. Only `X_train`, `X_val`, and `y_val`
are loaded: training inputs establish type ranges; validation evaluates them.
**No test array is evaluated.**

`fpga/conifer/compat.py` preserves the unmodified upstream conversion as a
diagnostic, then reconciles it against native XGBoost JSON:

- Convert the binary logistic base probability into its initial raw logit,
  respecting its binary32 representation.
- Restore thresholds and leaf scores as exact native binary32 values. Textual
  dumps/dataframe values may differ at threshold-equality inputs.
- Derive the maximum depth from the tree topology. Loading native JSON leaves
  a default training parameter of depth 6; the actual model has depth 4.
- Preserve XGBoost's strict-left comparison: `x < threshold` goes left;
  equality goes right. Inputs are rounded to binary32 as in XGBoost DMatrix.
- Reject categorical, non-logistic and non-axis-aligned inputs to this bridge.
  NaN/missing-value routing is outside its explicitly finite-input contract.

The validation driver checks native margins and native `pred_leaf` indices.
An independent exported-tree walker compares every leaf ID against XGBoost,
while the score check exercises CMSSW's conifer external. Boundary inputs use
binary32 predecessor/equality/successor values at every covered internal cut,
seeded from reaching validation events. All supplied XGBoost handover vectors
are included and checked against their saved probabilities.

Both floating and fixed-point HLS projects are compiled with a normal C++
compiler and executed. These are **generated HLS firmware C++ simulations**, not
FPGA synthesis or a timing simulation. Fixed-point outputs must match CMSSW
**exactly** on the selected HLS vectors; floating comparisons have an explicit
`1e-5` margin tolerance against native XGBoost's sequential binary32 summation.
CMSSW and HLS both use balanced score aggregation for the fixed-point check.

## Precision scan and metrics

`configs/conifer_xgboost_emulation.json` specifies the scan. Input/threshold
integer bits are selected from training-input and model-threshold bounds.
Score integer bits use the sum of per-tree absolute leaf bounds, the intercept,
and a rounding allowance at the coarsest scanned fraction. Validation scores
do not select these ranges. Input and score fractional widths are scanned
through 18, 12, 10, 8, 6 and 4 bits, with `AP_RND_CONV, AP_SAT`.
Axis-aligned one-hot weights use `ap_fixed<2,2>`; zero and one are exact.

The primary metric remains interpolated signal efficiency at 1% background;
AUC is secondary. The native probability result is also checked through the
existing `src/evaluate.py`. Raw-margin ROC metrics are reported separately,
because probability rounding can introduce ties. Every emulator also records a
**realizable strict threshold** with at most 525 accepted background events in
the full validation sample. Its actual background/signal efficiencies must not
be confused with an interpolated ROC point, especially at coarse precision.

Generated reports contain model/data/source hashes, exact type strings, tool
versions, conversion diagnostics, range checks, native leaf comparison counts,
CMSSW/HLS agreement, and per-precision metrics. Runtime logs include I/O and
are not FPGA latency or a controlled CPU speed benchmark.

## Reusing tested HLS C++ builds

After a completed smoke run, its compiled HLS bridges can be reused for a full
validation run, avoiding repeated template compilation:

```bash
scripts/submit_xgboost_conifer.sh --output artifacts/conifer/xgboost_v1 \
  --reuse-hls artifacts/conifer/xgboost_smoke
```

Reuse checks the numerical model, precision configuration, compiler version,
conifer version and HLS tool metadata. The new run still evaluates its HLS test
vectors and records the shared-library hash. It does not reuse old predictions.
A fresh checkout can omit this option and compile all projects itself.

## FPGA synthesis is a subsequent gate

No hardware resources or latency are inferred from these emulation results.
Generated HLS configurations contain conifer's default part/clock only as export
placeholders. A resource benchmark needs an agreed FPGA, clock, interface,
throughput requirement and an accessible Vivado/Vitis toolchain. The runtime
report records which tool executables are on PATH. The arbitrary-precision
headers supplied by CMSSW suffice for these array-interface C++ tests; they are
not a full vendor synthesis installation. Streaming interfaces are not tested.

Useful upstream references:

- [CMS conifer emulation guide](https://cms-ml.github.io/documentation/inference/conifer.html)
- [conifer documentation](https://ssummers.web.cern.ch/conifer/)
- [XGBoost converter](https://github.com/thesps/conifer/blob/v1.11/conifer/converters/xgboost.py)
- [HLS tree implementation](https://github.com/thesps/conifer/blob/v1.11/conifer/backends/xilinxhls/firmware/BDT_unrolled.h)
