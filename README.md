# Tree Benchmark for OpenML 42468 Jet Classification

This directory benchmarks XGBoost and hard oblique TAO forests on the fixed
jet-classification splits from the low-precision FPGA benchmark. GOLLUM hard and
soft tree sources are archived under `vendor/gollum/`; the hard TAO adapter is
under `src/tao/`. Neural-network training and FPGA synthesis are not yet included.

## Current winning strategy and finalized FPGA handoff — v2

**Use the three-tree, depth-4 hard oblique forest with linear leaves as the
current implementation baseline.** Its frozen saved checkpoint reaches
**55.78% signal efficiency at 1% background acceptance**, validation AUC
**0.92758**, with **1,581 scalar parameters**. XGBoost's 100 depth-4 trees reach
55.48% with 3,099 parameters: the TAO snapshot matches/exceeds the reference
working point on validation with approximately 49% fewer scalar parameters.
This is a validation comparison, not a claim of statistically significant
outperformance or a held-out test result.

The compact alternative is **one depth-4 tree with linear leaves**: **55.22%**
efficiency, AUC **0.92683**, and **527 parameters**. It is only 0.26 percentage
points below XGBoost at the working point, with approximately 83% fewer parameters.
The v2 bundle also includes a five-tree depth-4 constant-leaf model (55.11%),
a split-unpenalized three-tree alternative (55.57%), and the previous v1 winner.

The authoritative selection and hashes are in
[`fpga_handoff/CURRENT_MODEL.json`](fpga_handoff/CURRENT_MODEL.json).
Read [`fpga_handoff/README.md`](fpga_handoff/README.md) for the implementation
contract. Current coefficients, vectors and comparison data are under
[`fpga_handoff/v2/`](fpga_handoff/v2/); **v1 remains unchanged** for existing work.

| Setting | Current winning strategy |
|---|---|
| Task | `qg_vs_wzt`: quark/gluon background vs W/Z/top signal |
| Architecture | 3 hard binary oblique trees, depth 4, affine linear leaf logits |
| Initialization | CART seed 42; bootstrap samples of the unchanged training partition for initial forest diversity |
| Forest output | Sum of tree logits; joint cross-entropy leaf refit |
| Split search | Weighted logistic proposal plus exact hard-loss threshold scans along current/proposed directions and 25/50/75% blends; accept only objective improvements |
| Regularization | Split L1 = leaf-feature L1 = `1e-5` in the **mean** cross-entropy convention; split bias penalized, leaf bias unpenalized |
| Selection metric | Validation signal efficiency at **1% background acceptance**; AUC secondary |
| Selected checkpoint | `qg_vs_wzt_hardtao_linear_d4_n3_sa1e-05_la1e-05_seed42_hyper_scan_v1_c32_best_efficiency.pkl`, pass 3 |
| Training status | Still running at freeze; selected snapshot is immutable, but neither snapshot nor run is labelled numerically converged |
| Data | Fixed standardized arrays; seed-42 split: 531,200 train / 132,800 validation / 166,000 held-out test |
| FPGA score | Increasing signal margin = **negative sum of selected leaf outputs**; no sigmoid is needed for threshold decisions |

**Finalization scope:** the floating-point FPGA handoff is finalized, not the
entire research scan. At the status check on 2026-10-24, 11/58 scan cases were
converged and the rolling scan remained active. No jobs were stopped and no
held-out test selection/evaluation was performed. Coefficients in v2 will not
change when those runs progress. A later winner requires a new bundle version.

The results do not support blaming the current L1 penalty for the original gap:
the current `1e-5` split/leaf penalties already reach parity, and turning off
split L1 also reaches it. These runs are not yet converged, so this is not a
settled ranking of regularization strengths. CART initialization is promising:
the compact single-tree result substantially improves the earlier warm-start
strategy at the same architecture. The ongoing controlled scan will refine
these conclusions; stronger compression is a subsequent study.

Current comparisons: [screenshot-ready table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/current_winner_comparison.png),
[ROC comparison](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/fpga_handoff_v2_roc.png),
[versioned data](fpga_handoff/v2/comparison.md), and
[live scan](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/hyper_scan_v1_report.md).

The ten compiled v2 C++ variants passed **3,038 vector checks** with exact
leaf-route agreement and maximum margin difference below **1.1e-14**. The
Python suite passes 64 tests, including frozen-model warm starts, path scheduling, compression-frontier selection and support refitting. These are floating-point references: fixed-point
formats, the target FPGA and synthesis remain implementation choices.

When a better **saved checkpoint** is verified at the primary working point,
freeze a new version and update this section plus `CURRENT_MODEL.json`.
Selection history is kept under `fpga_handoff/selections/`. Preserve previous
coefficients and vectors so firmware results remain traceable. A transient log
value alone is not sufficient to replace the implementation baseline.

The next active study is the [L1 compression path](#l1-compression-path-from-the-handed-over-winner).
It starts from this exact v2 snapshot (877 nonzero coefficients in 1,581 slots),
then increases split/leaf penalties separately and together. The immutable FPGA
handoff is unaffected. [Live compression report](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/l1_path_v1_report.md).
The [efficiency–sparsity frontier](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/compression_frontier.png)
and its [table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/compression_frontier_report.md)
combine the architecture scan, frozen models and L1 paths to identify the
smallest saved models that retain the reference efficiency.
For a focused comparison of the continuing three-tree control and frozen
compact one-tree model, see the [ROC](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/compression_pair_roc.png)
and [table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/compression_pair_report.md).
Regenerate them with `python scripts/publish_compression_pair.py`; the report
records the exact checkpoint hashes used and includes XGBoost as a reference.

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

To monitor the constant-leaf scan, run `python scripts/monitor_tao_constant.py`
(or `--once` for a single check). A lock prevents duplicate monitors. The
lightweight monitor checks Slurm and saved checkpoints every two minutes,
publishes `plots/comparisons/constant_training_status.md` and `.json`, and exits
when both final reports exist. Missing jobs are flagged for attention; the
monitor does not change model settings or silently restart failed training.

### Training improvements: controlled studies

The optional `model.split_search='hard_threshold'` update scans every distinct
training projection threshold along the current hyperplane direction, the
weighted-logistic proposal, and 25/50/75% blends. Equal projections move together;
constant routing and the L1 bias term are included. Every proposal is checked
against the actual hard-tree cross-entropy plus existing penalties before
acceptance. The original surrogate-only update remains the default for earlier
studies and checkpoints. The new mode adds no inference parameters.

`configs/tao_hard_search_v1.json` continues the seven converged linear layouts
using this update. Comparisons are published as `hard_search_*` and
`hard_search_vs_baseline_*`. Warm starts and additional training time are recorded.

The initial controller policy, archived in
`configs/tao_training_improvements_before_hyper_scan.json`, defined this
follow-up sequence (subsequently superseded by the wider factorial scan):

1. Finish and measure the split-search comparison.
2. At one and three depth-3 trees, compare random seeds 42/43/44 with CART
   initialization at seed 42. CART uses training events only; forests use
   bootstrap samples of that partition for initial diversity.
3. Start from the improved single depth-3 tree, append one zero-output tree,
   train the whole forest to the same stopping rule, then repeat for three
   trees. Appending a tree initially preserves the existing predictive function.
4. Warm-start the improved three-tree model with split and leaf L1 penalties
   changed separately by factors of ten. Stored parameter counts and nonzero
   feature-weight counts are reported separately.

`python scripts/monitor_tao_improvements.py` drives these phases, checking every
two minutes and publishing `plots/comparisons/training_improvements_status.md`.
Each phase must converge and produce its measured comparison before the next
starts. Submission receipts prevent duplicate jobs across controller restarts.
Numerical failures are left visible for inspection; convergence settings are
not automatically relaxed. Workers use `scripts/train_tao_study.slurm` and
resume their latest checkpoint at scheduler budgets.

All studies use the immutable exported training and validation arrays and the
same AUC/efficiency benchmarks. No held-out test evaluation is performed during
these studies. More seeds and hyperparameter comparisons increase selection
opportunities, so the final selected configuration still needs a frozen test
assessment. Training and CPU inference times are retained in the figures and
reports. The joint leaf KKT test certifies fixed-routing leaf optimization;
unchanged hard routing indicates exhaustion of the configured proposals,
not a proof of globally optimal hyperplanes.

The primary benchmark and ranking metric is signal efficiency at **1% background
acceptance**. AUC remains secondary. Reports rank terminal models by that
working point and include `*_working_point.png` showing efficiency versus
parameter count and measured CPU inference cost. This preference does not
change cross-entropy training or the numerical stopping rule. Updated workers
also retain `*_best_efficiency.pkl` with AUC as a tie-breaker, alongside the
existing best-AUC archive. Workers already running use their loaded code until
continuation; earlier unsaved checkpoints cannot be reconstructed from history
metrics alone. Discrete scores (especially constant leaves) use the shared
ROC interpolation convention; a deterministic threshold need not attain
exactly 1% background acceptance.

### Performance-first hyperparameter scan

`configs/tao_hyper_scan_v1.json` defines a controlled 54-case factorial scan:
1/3/5 trees, depths 2/3/4, linear/constant leaves, and split/leaf L1 penalties
0, 1e-7 or 1e-5 in the mean-cross-entropy convention. Four additional linear
three-tree cases at depths 3/4 turn off the split or leaf penalty separately.
Every case uses CART initialization at seed 42, exactly the same exported
train/validation arrays, and the same full-training convergence checks.
All 58 cases train to those checks; none is labelled converged by an epoch cap.

This replaces the previously queued initialization/growth/compression sequence
as the next priority. The old queue policy is retained as
`configs/tao_training_improvements_before_hyper_scan.json`; existing split-search
jobs continue independently. The immediate target is XGBoost's signal efficiency
at 1% background acceptance. Compression and stronger sparsity tuning follow a
competitive performance baseline. The general oblique-forest class contains
axis-aligned forests of the same structure; a tiny oblique forest is not
guaranteed to represent a much larger boosted forest.

Run `python scripts/monitor_tao_hyper_scan.py` on CLIP. Its rolling scheduler
keeps at most four scan jobs pending/running, including continuation attempts.
Each worker uses one CPU and 8 GB memory, checkpoints between optimizer stages,
and records its exit status. Only a clean continuation exit (75) is restarted
automatically. Failed or time-limited stages are flagged for inspection.
Unstarted cases are prioritized before subsequent attempts to provide broad
coverage, with larger models prioritized within that group; all cases remain subject to full convergence. The global leaf solver
uses 250-iteration chunks to keep checkpoints frequent.

Progress, a ranked table and architecture/regularization heatmaps are published
as `plots/comparisons/hyper_scan_v1_*`. Unfinished cases are explicitly
provisional; heatmaps show measured training compute time. The terminal scan
report records whether any converged model reaches the XGBoost validation
working point. After all cases converge, the best eight (ranked by efficiency,
AUC tie-breaker) receive common CPU timings and ROC plots as `hyper_scan_top_*`.
Zero-L1 split proposals use unpenalized L-BFGS because LIBLINEAR does not support
that objective. Constant leaves contain no penalized feature weights, so their
leaf L1 coefficient has no effect.

Large scan cases now checkpoint after individual leaf and split updates, rather
than only after a complete forest pass. `tao.epoch` serializes the original
prediction/routing caches and preserves the original node order and numerical
solvers. Tests pickle and resume between every node and reproduce an
uninterrupted epoch for binary/multiclass, constant/linear leaves and multiple
trees. This prevents a multi-tree epoch from repeatedly losing its work at the
one-hour Slurm limit. Existing completed checkpoints and stopping criteria are
unchanged. An inspected failed case can be explicitly approved for one recovery
via `results/hyper_scan_v1/case_XX_recovery.json`, bound to the failed job ID;
uninspected failures remain held. Recovery takes the next available scan slot.

`sbatch scripts/publish_current_winner.slurm` now republishes the pinned frozen
handoff via `CURRENT_MODEL.json`, rather than silently selecting a newer live
checkpoint. Current aliases include `plots/comparisons/current_winner_report.md`
and `current_winner_roc.png`, plus `winner_roc.png` in the selected layout's flat
folder. An explicit `--run` retains the older live-run diagnostic for completed
training runs. `scripts/verify_fpga_handoff.py` defaults to the current pinned
bundle; use `--bundle` to check an older version explicitly.

### L1 compression path from the handed-over winner

The v2 implementation baseline is handed over and remains pinned by
`fpga_handoff/CURRENT_MODEL.json`. The next study is separate and does not
modify any frozen coefficients: `configs/tao_l1_path_v1.json` starts from the
**exact standardized v2 JSON**, checked by SHA-256, rather than the live model
checkpoint that may continue changing.

A same-penalty control first continues this snapshot to numerical convergence
at split L1 = leaf L1 = 1e-5. This matters because the frozen selected snapshot
was saved before its training run converged. Three independent paths then
increase split-only, leaf-only, or both penalties by 3, 10, 30, 100 and 300,
warm-starting each point from its own previous **converged terminal** model.
The other penalty stays at 1e-5 in the one-penalty paths. There are 16 runs in
all (control plus 15 path points), with at most three path-study jobs active.
The existing broad architecture scan continues independently.

This is Lasso-style **L1 sparsity**, implemented for classification using
mean cross-entropy plus L1. Linear leaves use an exact positive/negative-part
L1 parameterization with L-BFGS-B; oblique split proposals use L1-regularized
weighted logistic regression, with actual hard-tree objective acceptance.
Replacing these with `sklearn.linear_model.Lasso` would introduce a squared-error
regression objective, which is not the classifier's current loss. The warm-started
sparsity path follows the idea in the
[original TAO paper, section 4](https://proceedings.neurips.cc/paper/2018/file/185c29dc24325934ee377cfda20e414c-Paper.pdf).

Run `python scripts/monitor_tao_l1_path.py` on CLIP. Submission receipts and a
file lock prevent duplicate controllers/jobs. Workers use resumable node-level
checkpoints and the existing numerical stopping rule; no epoch limit labels a
model converged. Failures remain visible for inspection. The monitor publishes
`plots/comparisons/l1_path_v1_report.md`, `_summary.json` and `_tradeoffs.png`.

The primary comparison remains signal efficiency at 1% background acceptance;
AUC is secondary, and test data remain untouched. Reports distinguish:

- 1,581 dense coefficient slots, which do not change while topology is fixed;
- exact nonzero coefficients, including biases;
- nonzeros in structurally reachable branches (only identically constant split
  decisions permit input-independent branch elimination);
- sparse selected-path multiplication/comparison bounds and training time.

If all weights of a split are zero, preserve its constant routing decision
before quantizing the bias: a tiny positive bias must not silently round to zero
and switch its child. Such nodes can instead be collapsed algebraically.

These are not synthesized FPGA resources or guaranteed latency improvements.
Sparsity must be exploited by the inference/export implementation to realize
savings. Counts do not silently treat small coefficients as zero. The latest
metrics and sparsity belong to the same evaluated checkpoint; best-efficiency
checkpoint sparsity is reported separately. The frozen winner and the completed
same-penalty control are both retained so optimization changes are not mistaken
for the effect of stronger regularization. Nonzero counts need not be monotonic
along a nonconvex tree-training path; measure each point before choosing a
compressed successor for a new handoff version.

The controller also runs `scripts/compression_frontier.py`, combining the
best-efficiency checkpoints from the architecture scan and L1 study with the
frozen v2 candidates. Each point's validation efficiency and exact nonzero
count come from the same saved coefficients. The frontier minimizes nonzero
coefficients while maximizing signal efficiency at 1% background; its table
also reports dense slots and sparse path MAC bounds. Descriptive loss bands
of 0, 0.25, 0.5 and 1 percentage point relative to XGBoost and the frozen winner
show nearby alternatives without selecting an acceptable degradation budget.
Unfinished source runs are marked provisional. Frontier and band-winning
pickle checkpoints are retained by SHA-256 under
`models/compression_frontier/`, so subsequent training cannot overwrite the
reported choices. Those model files remain outside Git, like other training
artifacts. This comparison uses validation data only and does not change the
implementation handoff. The aim is to find the efficiency–sparsity tradeoff
before choosing an inference implementation that exploits the zeros.

### Frozen-support refitting

`src/tao/support.py` implements **select → freeze → refit**. Start from a
SHA-256-pinned **converged terminal parent**, freeze its exact nonzero support,
and remove both L1 penalties. The support covers every split weight, split
bias, leaf feature coefficient and leaf bias, separately for each free class.
Zeros remain zero; surviving values can change magnitude and sign. Small
nonzero values are not silently discarded. An allowed value can become zero
without changing the original mask, so reports distinguish the **allowed
coefficient budget** from the current actual nonzero count.

The two stages are sequential:

| Stage | Fixed | Optimized |
|---|---|---|
| `leaf_only` | Original splits, routes and frozen-zero coordinates | All surviving leaf coefficients jointly, without L1 |
| `full` | The original support and tree layout | Surviving split and leaf coefficients, without L1 |

The full stage starts from the converged leaf-only result but keeps the
**parent's original support mask**. It does not reselect support. Every local
fit, joint fit, split proposal and threshold search enforces that mask.
Original splits with no nonzero feature weights keep their entire decision
fixed, including a tiny positive bias. The current implementation preserves
these nodes in the stored layout; physical branch compaction is a separate
export step. No branches are declared dead from validation or test events.

The constrained leaf KKT check considers only allowed coordinates. A frozen
zero may have a nonzero gradient because it is an equality constraint. The
worker requires KKT <= 1e-6, at least five epochs per stage and three stable
passes under the configured objective/routing/proposal checks. It rejects an
increased training objective. This certifies numerical optimization within the
support and configured split proposals, not global tree optimality. Lower
training cross-entropy does not guarantee better validation efficiency.

Removing L1 can expose complete or quasi separation, where logistic loss has
no finite coefficient optimum. A sparse linear-programming recession-direction
check runs before fitting a new routing configuration; its result is cached
while routes stay unchanged. Separation or an inconclusive/time-limited LP
blocks completion. No hidden L2 penalty or coefficient bound is substituted.
A separating **router** is different: if a split already sends every active
training event to its lower-loss child, its reduced hard objective is optimal,
even if an auxiliary logistic surrogate would prefer infinite margin. This
case is explicitly recorded as a pointwise-lower-bound certificate.

Copy `configs/tao_support_refit_template.json` to a new config, fill its unique
run name, terminal parent path and parent SHA-256, then run in `tree-benchmark`:

```bash
python scripts/train_tao_support_refit.py --config configs/your_refit.json --wall-seconds 2700
```

This runs one explicit parent study; it submits no jobs. Resume using the same
command/config. The worker checks the audited dataset, parent provenance,
feature/class metadata, identity preprocessing and implementation hashes.
Changes to the resume policy or implementation require a new run name. A file
lock prevents two CLI workers writing the same run. Both stages use the fixed
training/validation arrays and do not load test arrays.

Outputs are `models/<run>_latest.pkl`, converged `<run>_leaf_only.pkl` and
`<run>_full.pkl`, and `results/<run>_status.json`. Status includes the exact
parent's metrics and hash, selection penalties, immutable support hash,
allowed/actual counts, stage histories and stopping diagnostics. Earlier
best-efficiency parent snapshots are not interchangeable with terminal parents:
for example the both-L1=1e-4 three-tree terminal has 520 nonzeros, whereas its
previously reported best-efficiency snapshot has 577.

| Exit code | Meaning |
|---:|---|
| 0 | All requested stages numerically converged |
| 75 | Scheduler budget reached; checkpoint can resume |
| 78 | Complete/quasi separation detected; incomplete refit |
| 79 | Separation check inconclusive; incomplete refit |
| 80 | Numerical/proposal iteration safeguard reached; incomplete refit |

An inconclusive separation check can be retried explicitly with
`--retry-separation`; other failed checks are not silently bypassed. The LP's
box bounds only normalize its search direction; fitted coefficients themselves
are unconstrained within the support. Large-data LP time/memory must be measured
in a pilot before choosing the scan.

[Validation record](docs/support_refit_validation.md): **56 tests pass**, including
17 new support/refit checks and all existing tests. Freezing the saved real
one-tree and three-tree terminal parents preserves logits and routes exactly
on all 132,800 validation vectors each. These correctness checks do not claim new physics performance; the full-data
pilot and scan are documented below.

The [full-data pilot](docs/support_refit_pilot.md) is now running from the
converged 520-coefficient three-tree parent, using
`configs/tao_support_refit_pilot_v1.json`. Its first separation check passed in
52.6 seconds with 3.40 GiB process peak RSS. Unpenalized leaf fitting is underway;
this is not yet a completed refit result or a broader scan. The pilot Slurm
wrapper records per-allocation resource usage and resumes only scheduler-budget
exits; separation/numerical failures remain held for inspection.

### Autonomous support-refit scan

The [ten-parent support-refit scan](docs/support_refit_scan.md) is active, with
at most three refit jobs including the pilot. It compares mild and aggressive
L1-selected supports at 889/837/777/675/561/520/418/394/317 coefficients for
three D4 linear-leaf trees, plus a 403-coefficient single D4 tree. Every parent
is a pinned converged terminal checkpoint. Each case removes L1, refits leaves
with fixed splits, then refits the full model within exactly the same support.

[Live table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_report.md)
and [tradeoff figure](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_tradeoffs.png).
The controller will publish coefficient-verified final tables and ROCs once
all cases have terminal outcomes. Separation/stall outcomes stay explicit.
The pilot's leaf-only stage converged at 55.82% efficiency at 1% background,
versus its exact parent's 55.58%, with all 520 retained coefficients unchanged
in number. Full refitting remains provisional. Test data are not used.

Publishing automatically falls back to `results/publish_cache/www/tree_benchmark`
if the shared `/groups` cache is full; public URLs do not change. The suite now
passes **64 tests**, including scan scheduling, exact evaluated-checkpoint
archiving, final-report metric checks and cache failover.
