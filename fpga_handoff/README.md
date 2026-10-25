# FPGA implementation handoff — v3 Compact refit

**Dorian: start with `v3/compact_refit_3t_d4/`.** This is the implementation
default selected by Robert, even though the higher-efficiency alternative has a
larger ROC value. `CURRENT_MODEL.json` is authoritative. v1/v2 are preserved
historical bundles, not the default for new work.

Read [Dorian's complete handover](../docs/DORIAN_HANDOVER.md) for exact training
files, checkpoints, environments and reproduction commands.

| Role | Candidate | Layout | Dense / nonzero | Efficiency at 1% background |
|---|---|---|---:|---:|
| **DEFAULT: Compact refit** | `v3/compact_refit_3t_d4` | 3 x D4 linear | 1581 / 418 | **56.20%** |
| Higher efficiency | `v3/highest_efficiency_3t_d4` | 3 x D4 linear | 1581 / 837 | 57.12% |
| One-tree alternative | `v3/single_tree_1t_d4` | 1 x D4 linear | 527 / 452 | 55.63% |
| XGBoost reference | `v3/xgboost_reference` | 100 x D4 constant | 3099 / not counted | 55.48% |

[Manifest](v3/manifest.json) · [comparison](v3/comparison.md) ·
[C++ verification](v3/cpp_verification.json). Validation only; test untouched.
The default is the saved full-refit pass 8 from a source stage converged at pass
15; the selected pass itself is not labelled converged. The higher-efficiency
alternative uses its converged leaf-only stage; its held full-refit snapshot is
excluded. The one-tree reference is saved pass 6 from a completed source run.

## Bundle contents

- `manifest.json`: exact checkpoint hashes, selection/stage status, metrics,
  arithmetic counts and SHA-256 hashes of implementation files.
- `input_contract.json`: ordered features, class labels, saved training scaler,
  observed training ranges. Inputs must be finite.
- Every oblique candidate has `standardized/` and `raw/` variants containing
  `model.json`, `model_coefficients.h`, `ranges.json`, `vectors.tsv`,
  `expected.tsv`, and `tree_free_logits.tsv`.
- Candidate-level `validation_row_indices.tsv` identifies golden examples;
  vectors mix random rows and rows near split/working-point boundaries. v3 does
  not require labels for floating-point inference verification.
- XGBoost has native `model.json`, standardized vectors and expected signal
  probabilities, retained from the same verified reference dataset/model.
- `reference.cpp` is the standalone floating-point evaluator. All six v3
  variants pass 1,594 vector evaluations with exact routes, maximum margin
  error below 2.5e-14. Coefficients are not quantized or synthesized.

## Exact inference contract

The interchange-v1 interface is **binary classification**: class 0 is quark/gluon
background; class 1 is W/Z/top signal. Training code also supports multiclass,
but this firmware interface must not be used for multiclass without extending
its score representation.

A tree is stored in complete binary heap order. Root is node 0. For an internal
node `j`, compute `h = dot(split_weights[j], x) + split_bias[j]`:

- `h > 0`: right child `2*j + 2`.
- `h <= 0`, including exact zero: left child `2*j + 1`.
- Internal indices are `0 .. 2**depth-2`; final leaf index is
  `node - (2**depth-1)`.

For a linear leaf, its coefficient row is `[a_0, ..., a_15, bias]`; its output
is `dot(a, x) + bias`. Constant leaves have a single coefficient. The forest
output is the **sum** of the selected leaf outputs, with no division by the
number of trees. These are class-0 logits relative to class 1, whose logit is
zero. Consequently:

```text
signal_margin = -sum(selected_leaf_outputs)
P(signal) = sigmoid(signal_margin)
```

The FPGA can compare `signal_margin` directly to a threshold, with no sigmoid
or softmax. Do not sum probabilities, average tree scores, or reverse the
signal/background sign.

## Preprocessing

The standardized variant expects `x = (raw - mean) / scale`, in the exact
feature order in `input_contract.json`. Do not standardize twice. The training
model's own input adapter is identity because it consumes exported standardized
arrays directly.

Raw means the 16 unstandardized high-level features, not detector or particle
inputs. Upstream feature construction is outside this interface.

The raw variant folds preprocessing algebraically into every split and linear
leaf. For any affine function `w*x + b`:

```text
w_raw = w / scale
b_raw = b - dot(w_raw, mean)
```

It expects raw features and requires no separate scaler arithmetic. No clipping,
pruning or quantization is applied. Golden raw inputs are reconstructed in
float64 from standardized validation inputs and the saved scaler; they test
folding itself rather than raw-file rounding. Both variants are checked against
the trained model on the entire validation partition before export.

## Operating point and arithmetic

The reported 1% working point follows the common interpolated ROC convention.
`model.json` also records an implementable strict margin threshold with at most
`floor(0.01*N_background)` accepted validation background events. Its achieved
background and signal efficiencies are recorded separately. Quantized hardware
must re-evaluate this threshold on validation data.

`ranges.json` gives conservative affine output bounds within the observed
training feature box. Those are not guarantees on future physical input ranges.
Choose feature, coefficient and accumulator formats and overflow policy before
claiming bit-exact firmware behavior. There is no hidden saturation or clipping
in the floating reference.

For the default Compact refit, 3 x D4 linear allocates 1,581 scalar slots,
but only **418 coefficients are nonzero** (379 remain after algebraically
constant routes are eliminated). Sparse selected-path multiplication bound: 80;
the dense bound is 240. The one-tree alternative has 452 nonzero coefficients
and bound 65; the higher-efficiency alternative has 837 and bound 163. These are
standardized-input scalar arithmetic counts, not DSP usage, multiplier sharing,
pipeline placement, frequency, initiation interval or synthesized latency.

## Compile and verify without the training stack

A C++17 compiler and Python standard library are sufficient for the portable
reference checks. CLIP's default GCC 4.8 is too old; use a newer compiler or run
this check on another machine.

```bash
python3 scripts/verify_fpga_handoff.py --bundle fpga_handoff/v3 --compiler c++

c++ -std=c++17 -O2 -ffp-contract=off \
  -I fpga_handoff/v3/compact_refit_3t_d4/standardized \
  fpga_handoff/reference.cpp -o /tmp/tao_reference
/tmp/tao_reference < fpga_handoff/v3/compact_refit_3t_d4/standardized/vectors.tsv
```

Each output line contains `signal_margin leaf_index_tree0 leaf_index_tree1 ...`.
The verifier checks file hashes, exact leaf assignments and floating margins
against all standardized and raw golden vectors. Float tolerances are 1e-9
absolute / 1e-10 relative. Fixed-point bit-exact tests are a subsequent step.

## Reproduce the export

See [Dorian's handover](../docs/DORIAN_HANDOVER.md) for the exact training chain,
checkpoint hashes, Python 3.10 Conda setup and refit reproduction. The selection
is pinned by `configs/fpga_handoff_v3.json`. To export again, use a new version:

```bash
python scripts/freeze_support_handoff.py --output fpga_handoff/v4
python3 scripts/verify_fpga_handoff.py --bundle fpga_handoff/v4 --compiler c++
# Promotion is a separate intentional operation after verification:
# python scripts/freeze_support_handoff.py --output fpga_handoff/v4 --promote
```

The exporter consumes verified research JSON/vector files and does not train.
The verifier needs only Python standard library plus a C++17 compiler. It
preserves an existing frozen verification report; use `--report /tmp/check.json`
for a separate report. Existing bundle directories cannot be overwritten.
