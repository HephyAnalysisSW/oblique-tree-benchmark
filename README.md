# Tree Benchmark for OpenML 42468 Jet Classification

This directory is a focused, standalone benchmark for tree-based model training
and software evaluation on the same jet-classification splits used in the main
low-precision FPGA benchmark. It intentionally excludes BitNet, QKeras, HGQ,
hls4ml, Conifer synthesis, neural-network code, and plotting infrastructure.

The methodological rule is:

**BDT and oblique-tree models may differ only in the model/training algorithm.
Dataset, event split, labels, preprocessing, and evaluation must remain
unchanged.**

## Dataset

- OpenML dataset ID: `42468`
- Name: `hls4ml_lhc_jets_hlf`
- Inputs: 16 high-level jet features
- Split seed: `42`
- Model/training seed: `42`
- Split fractions: 64% train, 16% validation, 20% test
- Preprocessing: `StandardScaler` fitted on the fixed training partition only
- Event/sample weights: none

Both standardized and non-normalized raw-feature arrays are included. Each
`.npz` file contains:

```text
X_train, y_train
X_val, y_val
X_test, y_test
scaler_mean, scaler_scale
```

The raw files were reconstructed from the standardized files with:

```text
X_raw = X_standardized * scaler_scale + scaler_mean
```

Use standardized inputs for reproducing the main reference BDT. Use raw inputs
for oblique-tree ablations if the oblique split functions are sensitive to
feature scales. Do not regenerate or reshuffle either representation when
comparing BDTs and oblique trees.

## Tasks

Three fixed tasks are included:

| Task | Standardized file | Raw file | Metadata | Labels | Events |
| --- | --- | --- | --- | --- | --- |
| `qg_vs_wzt` | `data/qg_vs_wzt_standardized.npz` | `data/qg_vs_wzt_raw.npz` | `data/qg_vs_wzt_metadata.json` | 0 = quark/gluon background, 1 = W/Z/top signal | 531200 / 132800 / 166000 |
| `qg_vs_top` | `data/qg_vs_top_standardized.npz` | `data/qg_vs_top_raw.npz` | `data/qg_vs_top_metadata.json` | 0 = quark/gluon background, 1 = top signal | 317466 / 79367 / 99209 |
| `multiclass` | `data/multiclass_standardized.npz` | `data/multiclass_raw.npz` | `data/multiclass_metadata.json` | 0 = gluon, 1 = quark, 2 = W, 3 = Z, 4 = top | 531200 / 132800 / 166000 |

`data/metadata.json` is an alias for the primary `qg_vs_wzt` metadata.

The 16 feature names are stored in every metadata file:

```text
zlogz
c1_b0_mmdt
c1_b1_mmdt
c1_b2_mmdt
c2_b1_mmdt
c2_b2_mmdt
d2_b1_mmdt
d2_b2_mmdt
d2_a1_b1_mmdt
d2_a1_b2_mmdt
m2_b1_mmdt
m2_b2_mmdt
n2_b1_mmdt
n2_b2_mmdt
mass_mmdt
multiplicity
```

## Environment

Create a small environment for tree training:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The reference artifacts were checked with:

```text
numpy 1.26.4
scikit-learn 1.7.2
xgboost 3.2.0
PyYAML 6.0.3
matplotlib 3.10.7
```

## Reference BDTs

All included BDT tasks use the hardware-oriented benchmark tree baseline:

```text
n_estimators = 100
max_depth = 4
subsample = 0.8
colsample_bytree = 0.8
tree_method = hist
random_state = 42
```

The binary tasks use:

```text
learning_rate = 0.1
objective = binary:logistic
n_jobs = 1
```

The multiclass task uses the compact hardware-oriented variant:

```text
learning_rate = 0.06
objective = multi:softprob
num_class = 5
early_stopping_rounds = 15
```

Seed `42` is configured for every task:

```text
configs/bdt_qg_vs_wzt_seed42.yaml
configs/bdt_qg_vs_top_seed42.yaml
configs/bdt_multiclass_seed42.yaml
```

`configs/bdt_config.yaml` is a convenience alias for
`configs/bdt_qg_vs_wzt_seed42.yaml`.

Raw-input BDT configs are also included for preprocessing checks:

```text
configs/bdt_qg_vs_wzt_raw_seed42.yaml
configs/bdt_qg_vs_top_raw_seed42.yaml
configs/bdt_multiclass_raw_seed42.yaml
```

The trained seed-42 model pickle is included for all three tasks.

## Reference Results

Seed-42 metrics, probabilities, predicted labels, ROC arrays, and ROC plots are
stored in `results/`.
A compact index is available at:

```text
results/reference_summary.csv
```

The raw-vs-standardized BDT check is stored in:

```text
results/input_representation_comparison.csv
```

Current reference metrics:

| Task | Seed | Accuracy | Macro AUC | Signal efficiency at FPR=0.01 |
| --- | ---: | ---: | ---: | ---: |
| `qg_vs_wzt` | 42 | 0.84748193 | 0.92074118 | 0.56077001 |
| `qg_vs_top` | 42 | 0.90369825 | 0.95864705 | 0.54335587 |
| `multiclass` | 42 | 0.74766867 | 0.93066798 | n/a |

Raw-input BDT checks give nearly identical results for the axis-aligned BDT:

| Task | Representation | Accuracy | Macro AUC | Signal efficiency at FPR=0.01 |
| --- | --- | ---: | ---: | ---: |
| `qg_vs_wzt` | standardized | 0.84748193 | 0.92074118 | 0.56077001 |
| `qg_vs_wzt` | raw | 0.84748193 | 0.92074076 | 0.56077001 |
| `qg_vs_top` | standardized | 0.90369825 | 0.95864705 | 0.54335587 |
| `qg_vs_top` | raw | 0.90369825 | 0.95864705 | 0.54335587 |
| `multiclass` | standardized | 0.74766867 | 0.93066798 | n/a |
| `multiclass` | raw | 0.74778916 | 0.93067354 | n/a |

For the primary task, copied parent-benchmark prediction files are also stored
as `results/reference_qg_vs_wzt_xgboost_bdt_seed42_parent_predictions.npy`.

## Reproduce Evaluation

Evaluate included models:

```bash
python src/evaluate.py --config configs/bdt_qg_vs_wzt_seed42.yaml
python src/evaluate.py --config configs/bdt_qg_vs_top_seed42.yaml
python src/evaluate.py --config configs/bdt_multiclass_seed42.yaml
```

Evaluate the raw-input BDT checks:

```bash
python src/evaluate.py --config configs/bdt_qg_vs_wzt_raw_seed42.yaml
python src/evaluate.py --config configs/bdt_qg_vs_top_raw_seed42.yaml
python src/evaluate.py --config configs/bdt_multiclass_raw_seed42.yaml
```

The evaluator writes:

```text
results/<run_name>_metrics.json
results/<run_name>_probabilities.npy
results/<run_name>_predictions.npy
results/<run_name>_roc.npz
results/<run_name>_roc.png
```

## Reproduce Training

Train any reference BDT from the fixed arrays:

```bash
python src/train_bdt.py --config configs/bdt_qg_vs_wzt_seed42.yaml
python src/train_bdt.py --config configs/bdt_qg_vs_top_seed42.yaml
python src/train_bdt.py --config configs/bdt_multiclass_seed42.yaml
```

Train the raw-input BDT checks:

```bash
python src/train_bdt.py --config configs/bdt_qg_vs_wzt_raw_seed42.yaml
python src/train_bdt.py --config configs/bdt_qg_vs_top_raw_seed42.yaml
python src/train_bdt.py --config configs/bdt_multiclass_raw_seed42.yaml
```

Each run saves:

```text
models/<run_name>.pkl
models/<run_name>.history.json
results/<run_name>_metrics.json
results/<run_name>_probabilities.npy
results/<run_name>_predictions.npy
results/<run_name>_roc.npz
results/<run_name>_roc.png
```

## Shared Evaluation

All tree models must use:

```text
src/evaluate.py
```

The evaluator computes:

- accuracy
- macro ROC AUC
- per-class AUC
- average precision
- cross-entropy
- Brier score
- expected calibration error
- confidence coverage
- confusion matrix
- binary signal efficiency at fixed FPR, default `0.01`
- per-class efficiency at fixed FPR for binary and multiclass one-vs-rest curves
- ROC arrays saved as `.npz`
- ROC plots saved as `.png`
- predicted labels saved as `.npy`

For an alternative model, save either:

- a pickled object with `predict_proba(X)`, or
- a pickled payload containing `{"model": model}`, where `model` has
  `predict_proba(X)`.

Then run:

```bash
python src/evaluate.py \
  --config configs/oblique_template.yaml \
  --model models/your_oblique_model.pkl
```

## Where to Implement an Oblique Tree

Use:

```text
src/train_oblique.py
configs/oblique_template.yaml
configs/oblique_raw_template.yaml
```

The stub already loads the fixed data partitions. Implement the algorithm there
and then call the shared evaluator. To compare against another task or seed,
copy one of the BDT configs and change only the model-specific section and
output paths. Do not change:

- standardized and raw data arrays
- metadata files
- label mapping
- train/validation/test partition
- scaler/preprocessing
- `src/evaluate.py` metric definitions

That keeps the BDT and oblique-tree comparison isolated to the training
algorithm itself.
