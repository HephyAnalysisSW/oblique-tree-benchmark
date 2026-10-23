# Tree Benchmark for OpenML 42468 Jet Classification

This directory benchmarks XGBoost and hard oblique TAO forests on the fixed
jet-classification splits from the low-precision FPGA benchmark. GOLLUM hard and
soft tree sources are archived under `vendor/gollum/`; the hard TAO adapter is
under `src/tao/`. Neural-network training and FPGA synthesis are not yet included.

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

Use a dedicated Conda environment, `tree-benchmark`, with Python 3.10 and
all package versions pinned by `requirements.txt`. Keep this separate from the
existing `hephy-ml-gpu-claude` environment so that its dependencies stay unchanged.

On CLIP, initialize Conda if it is not already available in your shell:

```bash
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
```

Create the environment once and install the requirements from conda-forge:

```bash
cd /users/robert.schoefbeck/CMS/tree_benchmark
conda create -y -n tree-benchmark --override-channels -c conda-forge python=3.10 pip
conda activate tree-benchmark
conda install -y --override-channels -c conda-forge \
  --file requirements.txt "libxgboost=3.2.0=cpu_*"
python -m pip check
```

For subsequent sessions, activate the environment before training or evaluation:

```bash
conda activate tree-benchmark
cd /users/robert.schoefbeck/CMS/tree_benchmark
```

The CPU build of XGBoost is sufficient for the supplied histogram-based BDT
configs. Install it through Conda on CLIP: the login node has glibc 2.17, while
the PyPI Linux x86-64 wheel for XGBoost 3.2.0 requires glibc 2.28. The conda-forge
CPU build supports glibc 2.17.

The reference artifacts were checked with these pinned package versions:

```text
numpy 1.26.4
scikit-learn 1.7.2
xgboost 3.2.0
PyYAML 6.0.3
matplotlib 3.10.7
```

Use the pinned XGBoost version when loading the supplied pickle files. In a
compatibility check, `hephy-ml-gpu-claude` had XGBoost 3.0.5: the models loaded,
but their probabilities and predicted labels differed from the saved references.
Successful imports alone do not establish reproduction of these artifacts.

The dedicated CLIP environment was verified with Python 3.10.22 and the pins
above. `python -m pip check` passed, headless PNG plotting worked, and all six
supplied models (three tasks, standardized and raw inputs) reproduced the saved
test labels, accuracy, and macro AUC. Maximum absolute probability differences
were at most `1.2e-7`. This validates loading and inference; retraining is a
separate check.

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

## Retrain and Compare the Primary BDT

The primary `qg_vs_wzt` reproduction run uses a separate config so that the
supplied model and reference outputs remain available for comparison. Its data,
seed, and XGBoost parameters are identical to the reference config; only the run
name and output paths differ.

On CLIP, submit the training and comparison as a one-CPU Slurm job:

```bash
cd /users/robert.schoefbeck/CMS/tree_benchmark
sbatch scripts/retrain_bdt.slurm
```

The job activates `tree-benchmark`, retrains the BDT, evaluates it with the
unchanged shared evaluator, and overlays the reference and retrained signal ROC
curves. The plot includes a linear view and a logarithmic FPR view, with the
signal efficiency at 1% background acceptance marked.

To run the same steps within an existing CPU allocation:

```bash
conda activate tree-benchmark
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -B src/train_bdt.py --config configs/bdt_qg_vs_wzt_seed42_retrained.yaml
python -B src/compare_binary_runs.py \
  --reference-config configs/bdt_qg_vs_wzt_seed42.yaml \
  --candidate-config configs/bdt_qg_vs_wzt_seed42_retrained.yaml \
  --output-prefix results/qg_vs_wzt_seed42_reference_vs_retrained
```

Retraining outputs use the prefix `qg_vs_wzt_xgboost_bdt_seed42_retrained` in
`models/` and `results/`. Comparison outputs are:

```text
results/qg_vs_wzt_seed42_reference_vs_retrained_comparison.json
results/qg_vs_wzt_seed42_reference_vs_retrained_roc.png
results/qg_vs_wzt_seed42_reference_vs_retrained_roc.svg
```

The comparison recomputes both sets of metrics from their saved probabilities
and the same test labels using `src/evaluate.py`. It also reports changed labels
and absolute probability differences. The Slurm log is saved as
`results/retrain_qg_vs_wzt_seed42.slurm-<job_id>.log`.

The primary retraining was verified on CLIP (Slurm job `3619490`) against all
166,000 test events. Reference and retrained results agree: accuracy
`0.8474819277`, macro AUC `0.9207411790`, and signal efficiency at FPR `0.01`
`0.5607700126`. No predicted labels changed. The maximum absolute probability
difference was `5.96e-8`; the reference and retrained ROC curves overlap.

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

## Hard TAO: correctness and measured optimization

The current implementation is `src/tao/hard.py`, with linear leaves as the
default and constant leaves available for comparison. Original GOLLUM source
snapshots remain unchanged under `vendor/gollum/devel_rs/` and
`vendor/gollum/master/`. `vendor/gollum/PROVENANCE.md` records their commits.
The original checkout remains on `sbi-pdf-paper` and is not modified. Earlier
scalar-only benchmark prototypes under `src/tao/model.py`, `tree.py`, and
`forest.py` are retained for historical comparison; the current trainer does not
use them. Soft-tree code is archived only; no soft-tree training is run.

Each hard leaf outputs K-1 affine logits, with the final class logit fixed to
zero as a reference. The forest sums tree logits before softmax. Binary left/right
routing is independent of the number of classes. Trees are updated alternately
with all other trees' logits held fixed. Leaf fitting minimizes categorical
cross-entropy with those fixed logits included, rather than subtracting class IDs
as regression targets. L1 is retained for split and leaf coefficients; leaf
intercepts are unpenalized. The leaf optimizer represents L1 exactly with
nonnegative positive/negative coefficient parts and uses L-BFGS-B. It is not an
L2 replacement. Split fitting retains L1/liblinear by default. `--split-solver lbfgsb` is an
experimental optimizer for the same L1 objective, with a current-hyperplane warm
start; controlled equivalence tests and real-data timings are saved. It is not
the default because the measured benefit was small and zero-start fitting was
slower. All candidate updates are
checked against the actual objective before being accepted.

The objective is the full-training-weight-normalized cross-entropy plus
`split_alpha * (sum(abs(split weights)) + sum(abs(split biases)))` and
`leaf_alpha * sum(abs(leaf feature coefficients))`. Consequently the numerical
regularization values differ from the original unnormalized split-loss config;
they must not be compared as if they were the same hyperparameter convention.
Both sigmoid/softmax are probability outputs; inference routing remains hard.

The joint leaf refit explicitly receives the complete training partition.
Constant-leaf refits contain no feature terms. Linear-leaf refits preserve all
linear coefficients. Direct CSR construction avoids Python triplet lists and
records the number of training rows and matrix nonzeros. Zero-weight routing
ties are excluded from split fitting. The raw-statistics `InputTransform` takes
the square root of variances and handles zero-variance features. Fixed exported
standardized arrays use identity preprocessing exactly once; no new scaler is
fitted during training or evaluation.

Correctness checks (binary and 3/5 classes, conditional gradients, full-data
refit, tied routing, normalization, serialization, hard boundaries, and
objective descent):

```bash
conda activate tree-benchmark
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

Measured steps, using one CPU and controlled data:

```bash
sbatch scripts/measure_tao.slurm
```

Results are in `results/tao_steps/measurements.json`. Steps 1-5 are isolated
correctness demonstrations. Step 6 times identical fixed models and matrices;
it does not attribute a new loss function's runtime to a performance optimization.
`results/tao_steps/hard_v1.py` preserves the implementation used for that report.
Batched routing is measured separately by `scripts/measure_batched_routing.py`.
Dense traversal remains the default because indexed/batched variants were not
consistently faster at shallow depths. `--routing` selects the measured variants.
The frozen real-data split benchmark is `scripts/measure_split_solver.py`.
The report, raw measurements, full-data pilot profiles, and figures are published
at https://schoef.web.cern.ch/schoef/tree_benchmark/results/tao_steps/.
Use `scripts/summarize_tao_steps.py` to refresh the report from completed epochs.

Example short full-training pilots (validation only):

```bash
sbatch --time=00:15:00 scripts/train_tao.slurm \
  --task qg_vs_wzt --depth 2 --trees 1 --leaf-mode linear \
  --epochs 2 --leaf-max-iter 80 --suffix correctness_pilot
sbatch --time=00:15:00 scripts/train_tao.slurm \
  --task multiclass --depth 2 --trees 1 --leaf-mode linear \
  --epochs 2 --leaf-max-iter 80 --suffix correctness_pilot
```

Use `--task qg_vs_top` for the other binary task and `--leaf-mode constant` for
the leaf-type ablation. Seeds default to 42. The trainer reads only train and
validation arrays. Validation macro AUC selects checkpoints; the test partition
is not loaded. Checkpoints are saved after every validation improvement. Run
configs, per-epoch profiles, actual objective values, optimizer convergence flags,
validation metrics, and structural counts are retained. Iteration-limited pilot
fits must not be presented as converged models or final BDT comparisons.

The first controlled layout scan is defined in `configs/tao_layout_scan_v1.json`:
one linear-leaf tree at depths 2, 3, 4, and 5, and two trees at depth 3. It keeps
both L1 penalties at 1e-5 and seed 42. Each layout gets four alternating epochs
with 100-iteration leaf/split budgets, followed by up to 1,000 additional joint
leaf-refit iterations with the selected checkpoint's splits fixed. Validation
AUC selects checkpoints; efficiency at 1% background acceptance is reported
independently. This is a finite-budget exploration, not a convergence claim.

```bash
tao_scan_job=$(sbatch --parsable scripts/tao_layout_scan.slurm)
sbatch --dependency=afterok:"$tao_scan_job" scripts/summarize_tao_layout_scan.slurm
```

Runs refuse to overwrite models. To repeat training, first give the scan a new
`scan_name` and suffix. The summary times all selected models and XGBoost on the same worker,
with one thread and resident validation inputs, at batch sizes 1, 1,024, and
132,800. Nine warmed repeats are retained. ROC legends include inference time;
additional figures compare AUC/efficiency against inference time and show
training wall time. Figures and raw summaries are published under
`plots/comparisons/layout_scan_*`; individual ROC comparisons also appear in
each layout's flat figure folder. `--allow-partial` on the summary script
publishes completed members while the remaining fits run.

### Full training and exact split audit

The four-epoch layout scan above is historical screening, not a fully trained
comparison. `scripts/audit_tao_splits.py` checks all ten screening TAO checkpoints
and the retrained BDT against the reference configuration, and checks that the
entire NPZ is byte-identical to Dorian's original export. It records SHA-256
hashes of each uncompressed NPY member, including row order, dtype and shape.
The supplied BDT's saved split is train/validation/test = 64/16/20%, seed 42.
All models use the same exported 531,200 / 132,800 / 166,000 rows. Test members
are hashed for integrity only; their labels are not used for training, stopping,
model selection or the current comparison metrics.

`configs/tao_full_training_v1.json` defines seven layouts: one tree at depths
2–5, two depth-3 trees, three depth-2 trees, and three depth-3 trees. Existing
layouts continue from screening checkpoints; the new three-tree layouts start
from seed 42. Both L1 penalties remain 1e-5 and leaves remain linear.

```bash
python scripts/audit_tao_splits.py
sbatch scripts/train_tao_fully.slurm
```

The full trainer has no fixed epoch cap and no validation-patience stop. After
at least five additional passes, it requires three consecutive passes with:

- Relative training objective improvement at most 1e-5.
- At most 1e-4 of training events changing leaf assignment in any tree.
- All local leaf and split solver fits converged.
- Global joint-leaf L1 KKT residual at most 1e-6, checked on all training rows.

Leaf fits use tighter numerical tolerances (`ftol=1e-12`, `gtol=1e-8`). The
1,000-iteration global solver budget is a resumable chunk, never a stopping
criterion. Local solver limits increase when needed. This is numerical
coordinate convergence of a nonconvex hard-tree algorithm, not proof of a
global optimum.

Checkpoints are saved atomically between local/split and global stages. Before
the scheduler's one-hour limit, an unfinished worker saves and automatically
submits a continuation for that layout. `*_latest.pkl` and `*_status.json`
record progress; the terminal `*_full_training_v1.pkl` is written only after
convergence. Best-validation checkpoints are kept separately. The primary full
comparison uses converged terminal checkpoints and reports their validation
metrics. Historical screening files are preserved.

Once all seven layouts pass, the completion barrier automatically submits
`scripts/summarize_tao_full_training.slurm`. It times every model and XGBoost
on the same CPU worker and publishes `plots/comparisons/full_training_*`, with
timings in the ROC legends, performance/time plots and training curves. Each
layout also receives `qg_vs_wzt_full_training_roc.png` in its flat figure folder.
The raw summary includes the split audit, model hashes and convergence checks.
The public `plots/comparisons/full_training_status.md` reports the last saved
state and links the audit; it refreshes at job continuations and completions.

Plotting defaults to every five epochs plus the final epoch. Counts of nodes
and coefficients describe algorithm size, not synthesized FPGA LUT/DSP/BRAM use
or latency. Quantization and hardware estimates come after the floating-point
comparison. SciPy 1.15.2 is now pinned explicitly; it was already installed in
`tree-benchmark` as a scikit-learn dependency.

## Automatic Figure Publishing

`common/syncer.py` is adapted from the existing HEPHY-uncertainty syncer. It keeps
the `xrdfs`/`xrdcp` upload mechanism and `/www/` path mapping. ROOT is optional;
Matplotlib `Figure.savefig` (including `plt.savefig`) and ROOT canvas output
are intercepted. Existing benchmark entry points import the syncer through the
evaluator. Future standalone plotting scripts should `import common.syncer`.

Figures are mirrored locally to:

```text
/groups/hephy/cms/robert.schoefbeck/www/tree_benchmark/
```

Figures are published in descriptive subfolders, inferred from their run configs:

```text
plots/tao_1tree_depth3_linear/
plots/tao_1tree_depth2_linear/
plots/xgboost_100rounds_depth4/
plots/comparisons/
plots/diagnostics/
```

Browse https://schoef.web.cern.ch/schoef/tree_benchmark/plots/.
XGBoost folders specify boosting rounds because multiclass fits build one tree
per class per round. Training artifacts retain their original local paths;
the syncer changes only the public figure destination. Other files retain
their repository-relative paths, including the report in `results/tao_steps/`.
Previously published copies remain available at their existing URLs.
Run `python scripts/publish_figures.py` to publish existing figures in this layout.

For a validation ROC against the reference BDT, CPU probability-API benchmarks,
and explicit operation/storage estimates, run:

```bash
sbatch scripts/compare_tao_inference.slurm --tao-config configs/<TAO-run>.yaml
```

This reads validation arrays only. It benchmarks the existing Python/NumPy TAO
implementation and compiled XGBoost with one CPU thread, using batches from one
event to the full validation partition. Raw timing repeats, model hashes, node
counts, assumptions, and a report are published beside the ROC in the tree's
figure folder. Logical fixed-width payload and operation counts are estimates;
they do not establish FPGA latency, DSP/LUT/BRAM use, or quantized accuracy.

`CERN_USER` defaults to `schoef`. If present, `~/.private/.proxy` is selected as
`X509_USER_PROXY`; Kerberos is not required. Renew the CMS proxy before expiry:

```bash
voms-proxy-init --rfc --voms cms -valid 192:00 -out ~/.private/.proxy
```

Uploads occur automatically on normal process exit; `common.syncer.sync()` can
also be called explicitly. Failed uploads retain a persistent pending-file list
and the staged local figures. Retry with:

```bash
conda activate tree-benchmark
python -m common.syncer
```

Concurrent jobs serialize uploads with a file lock. File-like save targets such
as `BytesIO` are ignored. Override the local plot root with
`TREE_BENCHMARK_PLOT_DIRECTORY` if needed. Keep its path under `/www/` to preserve
the intended EOS mapping.


### Constant-leaf comparison and parameter accounting

`configs/tao_constant_full_training_v1.json` runs the same seven layouts with
constant leaves, from seed 42, on the exact same exported training/validation
arrays. Linear results remain separate. The convergence policy and split L1
penalty are unchanged. Constant leaf values are unpenalized biases, so the
leaf-feature L1 penalty has no effect in this mode.

On CLIP, submit `sbatch scripts/train_tao_constant_fully.slurm`. The worker
accepts `--config` to select a manifest. Checkpoints resume automatically at the
scheduler budget. Once all seven models converge, the summary job produces
`plots/comparisons/constant_full_training_*` and `overall_training_*`, including
ROC curves, CPU timings, and an overall Markdown table with parameter counts.
The overall comparison re-times linear, constant and XGBoost models together
on the same CPU worker. CPU timings still compare Python/NumPy TAO with the
compiled XGBoost implementation; they are not FPGA latency estimates.

Parameter counts include stored scalar coefficients, biases and zeros. Every
oblique split has 16 weights plus one bias. A binary linear leaf has 17
coefficients, a constant leaf one. XGBoost counts one threshold per internal
node, one score per leaf and one global base score. Integer feature indices,
tree topology and shared input preprocessing are excluded; parameter counts
are not complete serialized sizes or identifiable degrees of freedom.

The repository retains the original datasets and reference models already
tracked in its starting commit. New checkpoints, logs, figures and local
artifacts are ignored. The split audit checks the exact NPZ before training.
