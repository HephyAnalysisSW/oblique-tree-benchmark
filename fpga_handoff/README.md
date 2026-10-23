# FPGA implementation handoff

The frozen `v1/` bundle is a floating-point specification of the current
oblique candidates and the XGBoost reference. Start with `winner_3t_d3` and
compare `compact_3t_d2` and `compact_single_1t_d4` for different layouts.
The manifest identifies the exact checkpoints and selection epochs. This
handoff can be implemented while the wider training scan continues; do not
silently replace a versioned model with a newer training checkpoint.

## Frozen comparison

See [comparison.md](v1/comparison.md) and [manifest.json](v1/manifest.json).
The primary metric is signal efficiency at 1% background acceptance on the
fixed validation partition. The held-out test set has not been evaluated.
Snapshots were selected by validation efficiency, which may peak before the
terminal training iterate. The three depth-2 tree training run was still
ongoing when its saved checkpoint was frozen.

Counts include scalar model coefficients and biases, including zeros. They
exclude common input preprocessing and topology. CPU measurements compare
NumPy TAO and compiled XGBoost APIs; they are not FPGA latency measurements.

## Bundle contents

- `manifest.json`: dataset/source checkpoint hashes, code commit, selected
  epochs, metrics, training status, parameter/operation counts and file hashes.
- `input_contract.json`: the 16 features in exact order, class labels, saved
  training scaler and observed training ranges. Inputs must be finite.
- Each candidate has `standardized/` and `raw/` variants, each containing
  `model.json`, `model_coefficients.h`, `ranges.json`, `vectors.tsv`,
  `expected.tsv`, and `tree_free_logits.tsv`.
- Candidate `validation_row_indices.tsv` and `labels.tsv` identify golden
  examples. They mix random validation examples and examples near split or
  working-point boundaries. No test examples are included.
- `xgboost_reference/model.json` is the native XGBoost model, with matching
  standardized golden inputs and expected signal probabilities.
- `reference.cpp` is a standalone floating-point inference reference.
  C++ verification results are recorded in `v1/cpp_verification.json`.

## Exact inference contract

This v1 interface is **binary classification**: class 0 is quark/gluon
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

For the 3-tree depth-3 linear winner, there are 21 internal nodes and 24 leaves:
357 split coefficients plus 408 leaf coefficients = **765 scalars**. A selected
path uses 192 dense scalar multiplications, 194 additions and 9 comparisons
across the forest. Evaluating every node and leaf uses 720 multiplications.
These counts do not specify DSP usage, multiplier sharing, pipeline placement,
clock frequency, initiation interval, routing delay or synthesized latency.

## Compile and verify without the training stack

A C++17 compiler and Python standard library are sufficient for the portable
reference checks. CLIP's default GCC 4.8 is too old; use a newer compiler or run
this check on another machine.

```bash
python3 scripts/verify_fpga_handoff.py --bundle fpga_handoff/v1 --compiler c++

c++ -std=c++17 -O2 -ffp-contract=off \
  -I fpga_handoff/v1/winner_3t_d3/standardized \
  fpga_handoff/reference.cpp -o /tmp/tao_reference
/tmp/tao_reference < fpga_handoff/v1/winner_3t_d3/standardized/vectors.tsv
```

Each output line contains `signal_margin leaf_index_tree0 leaf_index_tree1 ...`.
The verifier checks file hashes, exact leaf assignments and floating margins
against all standardized and raw golden vectors. Float tolerances are 1e-9
absolute / 1e-10 relative. Fixed-point bit-exact tests are a subsequent step.

## Reproduce the export

The export requires the CLIP training environment and saved checkpoint pickles:

```bash
sbatch scripts/export_fpga_handoff.slurm
# Or select a new immutable version; existing manifest versions cannot be overwritten.
python scripts/export_fpga_handoff.py --output fpga_handoff/v2
```

The exporter never trains or changes checkpoints. Candidate selection is defined
in `scripts/export_fpga_handoff.py`; checkpoint bytes are read once, hashed and
converted to portable coefficients. The JSON/headers and golden vectors are
sufficient for implementation; the Python pickles are not required by firmware.
