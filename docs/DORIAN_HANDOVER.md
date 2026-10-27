# Dorian: implement Compact refit from FPGA bundle v3

**Start with `fpga_handoff/v3/compact_refit_3t_d4/`. This is the implementation default chosen by Robert.** The higher-efficiency model is an alternative, not the default. `fpga_handoff/CURRENT_MODEL.json` is authoritative. v1 and v2 are historical immutable snapshots.

Repository: https://github.com/HephyAnalysisSW/oblique-tree-benchmark (branch `main`). Reference CLIP checkout: `/users/robert.schoefbeck/CMS/tree_benchmark`. Use your own checkout for training; the reference checkout contains archived checkpoints and ongoing independent studies.

| Role | v3 candidate directory | Architecture | Dense / nonzero coefficients | Reachable nonzero | Sparse path MAC bound | Signal efficiency at 1% background | AUC |
|---|---|---|---:|---:|---:|---:|---:|
| **DEFAULT: Compact refit** | `compact_refit_3t_d4` | 3 hard oblique trees, depth 4, linear leaves | 1581 / 418 | 379 | 80 | **56.20%** | 0.92733 |
| Higher efficiency alternative | `highest_efficiency_3t_d4` | 3 hard oblique trees, depth 4, linear leaves | 1581 / 837 | 837 | 163 | 57.12% | 0.92967 |
| One-tree alternative | `single_tree_1t_d4` | 1 hard oblique tree, depth 4, linear leaves | 527 / 452 | 452 | 65 | 55.63% | 0.92704 |
| XGBoost reference | `xgboost_reference` | 100 axis-aligned trees, depth 4, constant leaves | 3099 / not counted | — | — | 55.48% | 0.92056 |

All metrics in this handover use the unchanged validation split. The oblique study did not evaluate or select models on test data; the earlier XGBoost reproduction check did evaluate the test partition. Nonzero counts include biases, refer to standardized inputs, and use exact zeros. Reachability removes only input-independent constant decisions. MAC bounds are scalar arithmetic counts, not synthesized FPGA resources or latency. Raw folding can change bias zero counts.

## Status rechecked on 26 October 2026

Compact refit remains the pinned v3 default; its coefficients and the two v3 alternatives are unchanged. The L1 path study is complete (16/16), the support-refit scan is terminal (6 converged full refits and 4 held outcomes), and the broader architecture scan has 37/58 converged cases with three jobs running in the checked snapshot. The provisional five-tree 57.13% result is not a new handover model. See the [dated research update and checkpoint provenance](handover_status_2026-10-26.md).

## Implementation files and verification

For the default, use either `fpga_handoff/v3/compact_refit_3t_d4/standardized/` **or** `raw/` consistently:

- `model.json`: exact coefficients, feature order, score/routing convention, scaler, and this input space's strict working-point threshold.
- `model_coefficients.h`: the same coefficients as C++ hexadecimal double constants.
- `vectors.tsv` and `expected.tsv`: golden inputs and expected `signal_margin leaf0 leaf1 leaf2`.
- `tree_free_logits.tsv`: individual tree outputs before summation/sign reversal.
- `ranges.json`: conservative bounds within the observed training feature box, not physical input limits.
- Candidate-level `validation_row_indices.tsv`: row numbers in the original validation partition (not test rows).

The shared `fpga_handoff/v3/input_contract.json` gives all 16 features in exact order and training-only scaler/ranges. The other two oblique candidates use the identical file layout. XGBoost is its native JSON plus standardized vectors and expected signal probabilities.

Inference: right iff `dot(w,x)+bias > 0`; equality goes left. Heap root 0, left `2*j+1`, right `2*j+2`; leaf index is `node-(2**depth-1)`. Leaf row is `[16 weights,bias]`. **Signal margin is the negative sum of the selected leaf outputs**, with no averaging. Compare the margin directly; sigmoid is unnecessary. Standardized inputs are `(raw-mean)/scale`. The raw variant already folds this into its coefficients; do not standardize again. Raw here means high-level features, not detector inputs.

Each input-space JSON has its own implementable strict `>` threshold. Do not copy a threshold between raw and standardized variants or substitute the interpolated ROC point. The JSON records accepted validation background counts; quantized hardware needs a new validation of its operating point.

No Conda, datasets, pickles or CERN proxy are needed for the portable reference checks. A Python 3 interpreter and C++17 compiler suffice:

```bash
git clone https://github.com/HephyAnalysisSW/oblique-tree-benchmark.git
cd oblique-tree-benchmark
python3 scripts/verify_fpga_handoff.py --compiler c++
# Default comes from CURRENT_MODEL.json; to be explicit:
python3 scripts/verify_fpga_handoff.py --bundle fpga_handoff/v3 --compiler c++
c++ -std=c++17 -O2 -ffp-contract=off \
  -I fpga_handoff/v3/compact_refit_3t_d4/standardized \
  fpga_handoff/reference.cpp -o /tmp/compact_refit_reference
/tmp/compact_refit_reference < fpga_handoff/v3/compact_refit_3t_d4/standardized/vectors.tsv
```

All six C++ variants passed 1,594 vector evaluations with exact routes and maximum margin error below 2.5e-14 (Apple Clang 17); see `fpga_handoff/v3/cpp_verification.json`. CLIP's default GCC 4.8 cannot run this C++17 check; use a modern compiler on your implementation machine. Re-running the verifier preserves the frozen report; `--report /tmp/new_check.json` writes a separate report.

## Exact selected training artifacts on CLIP

All paths below are relative to `/users/robert.schoefbeck/CMS/tree_benchmark`. Pickles and datasets are deliberately outside Git; the portable bundle is sufficient for implementation. The checkpoint paths by hash are immutable copies selected during publication.

**Compact refit (default)**

- Frozen pickle: `models/support_refit_scan_v1_selected/e89b8923fba66f5df02046e04a1828a8ea4c56f60c1d7b201a2a2f92a0ab00fa.pkl`; SHA-256 is the filename stem.
- Original saved snapshot: `models/qg_vs_wzt_support_scan_v1_split300_full_best_efficiency.pkl`.
- Selected **full-refit pass 8**, not the terminal iterate. The full source stage subsequently converged at pass 15 (terminal efficiency 55.98%, selected snapshot 56.20%). The selected pass itself is not labelled stationary.
- Refit configuration: `configs/support_refit_scan_v1/split300.json`; wrapper `scripts/run_support_refit_scan_case.py`; worker `scripts/train_tao_support_refit.py`; batch wrapper `scripts/train_tao_support_refit_scan.slurm`.
- Exact L1-selected parent: `models/qg_vs_wzt_hardtao_linear_d4_n3_sa0.003_la1e-05_seed42_l1_path_v1_split_05.pkl`; SHA-256 `b1ef49948984cb2702a3d493a74741816863e2ffe61efdfb106ec0ca68c23560`.
- Parent support-selection configuration: `configs/l1_path_v1/split_05.json`: split L1 **0.003**, leaf-feature L1 **1e-5**. This is the end of a continuation path, not a fresh single-penalty fit: frozen v2 model → `configs/l1_path_v1/control.json` → `split_01.json` → `split_02.json` → `split_03.json` → `split_04.json` → `split_05.json` in that directory. Path manifest: `configs/tao_l1_path_v1.json`; parent-training worker: `scripts/train_tao_fully.py`.
- L1 strengths use the mean per-event cross-entropy convention. Split biases are penalized; linear-leaf biases are unpenalized during support selection.
- Refit first optimizes surviving leaf coefficients with splits fixed, then surviving split and leaf coefficients with **both L1 strengths zero** and immutable masks. Zeros cannot regrow. Implementation: `src/tao/support.py`, `src/tao/hard.py`, `src/tao/epoch.py`, `src/tao/convergence.py`, `src/tao/split_search.py`.

**Higher-efficiency alternative**: `configs/support_refit_scan_v1/leaf3.json`, saved `models/qg_vs_wzt_support_scan_v1_leaf3_leaf_only.pkl`, immutable pickle SHA `80b0cd6a23d0528a31a691b27d4b006cc9acd12dca4d56022f7f4e046f744700`. Support selected with split/leaf L1 `1e-5/3e-5` (`configs/l1_path_v1/leaf_01.json` after the control), then leaf-only refit without L1. The leaf-only stage converged; the subsequent full stage was held after a separation-diagnostic timeout. The diagnostic 57.24% full snapshot is **not** in the FPGA bundle.

**One-tree alternative**: `configs/hyper_scan_v1/case_12.json` and `configs/qg_vs_wzt_hardtao_linear_d4_n1_sa0_la0_seed42_hyper_scan_v1_c12.yaml`; worker `scripts/train_tao_fully.py`. Original unpenalized CART-initialized model, not a support-refit result. Selected pass 6 from a subsequently converged run. Immutable pickle SHA `a06140db35833c3ebae2d0f41fb1e87bb6b76005556173e903aa6994fe0f2c05`, under `models/support_refit_scan_v1_selected/`.

**XGBoost reference**: `configs/bdt_qg_vs_wzt_seed42.yaml`, trainer `src/train_bdt.py`, archived pickle `models/qg_vs_wzt_xgboost_bdt_seed42.pkl`, portable native JSON `fpga_handoff/v3/xgboost_reference/model.json`.

All model JSON/file hashes, source paths, selected stages and convergence distinctions are recorded in `fpga_handoff/v3/manifest.json` and candidate `candidate.json`. Original research provenance remains in `research_handoff/support_refit_scan_v1/selection.json`. Training implementation is based on commit `720493800102471b82eb7078cde9b31a349f42f4`; later handover changes do not modify the fitting code.

## Training setup and controlled reproduction

On CLIP, use an isolated Python 3.10 environment with **exactly `requirements.txt`** (NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.7.2, XGBoost 3.2.0, PyYAML 6.0.3, matplotlib 3.10.7). The existing reference environment is named `tree-benchmark`; create your own if working under another account:

```bash
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda create -y -n tree-benchmark --override-channels -c conda-forge python=3.10 pip
conda activate tree-benchmark
conda install -y --override-channels -c conda-forge \
  --file requirements.txt 'libxgboost=3.2.0=cpu_*'
python -m pip check
export PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Use Conda-forge's CPU build on CLIP (glibc 2.17); the PyPI XGBoost 3.2.0 Linux wheel requires newer glibc. Training needs no GPU. A CERN proxy is only needed for optional figure uploads, not inference/training; do not copy Robert's proxy into your setup.

Primary dataset: `data/qg_vs_wzt_standardized.npz`, metadata `data/qg_vs_wzt_metadata.json`. Exact NPZ SHA-256:

`f5ec6f7ed442010656a904700f8f6421cc34b674a1ec810c5cf6b2de003ed926`

It contains 531,200 train / 132,800 validation / 166,000 test events, seed 42, with the scaler fitted on training only. Do not regenerate, reshuffle, rescale or train on validation/test. In a fresh training checkout, obtain the unchanged NPZ and metadata plus `results/tao_split_audit.json` from the reference checkout; workers independently verify the dataset hash. Git alone supplies neither datasets nor pickles.

To repeat **only the default refit**, copy the exact parent pickle above into your own checkout at the configured path, then make a new config/run name so existing research results cannot be overwritten:

```bash
mkdir -p models results
python - <<'PYCONFIG'
import json
from pathlib import Path
p=json.loads(Path('configs/support_refit_scan_v1/split300.json').read_text())
p['run_name']='qg_vs_wzt_support_split300_dorian_reproduction'
Path('configs/dorian_compact_refit.json').write_text(json.dumps(p,indent=2)+'\n')
PYCONFIG
python -B scripts/run_support_refit_scan_case.py \
  --config configs/dorian_compact_refit.json --wall-seconds 2700
```

Exit 75 means a clean compute-budget checkpoint: repeat the identical command/config until completion. Exit 0 is convergence; 78/79/80 are separated/inconclusive/stalled outcomes, not success. Best evaluated snapshots are archived as `models/<new_run_name>_full_best_efficiency.pkl`; terminal coefficients are `<new_run_name>_full.pkl`. Do not replace the handed-over pass-8 coefficients with a terminal or newly retrained file. The exact masks, minimum 5 passes, KKT ≤1e-6, relative loss gain ≤1e-5, routing change ≤1e-4, and 3 stable passes are pinned by the config.

To rebuild the **L1-selected parent** rather than using its archive, execute `scripts/train_tao_fully.py --config <path_config> --index 0 --wall-seconds 2700` for the control and each split-path config listed above, in order, repeating a stage on exit 75 and advancing only on exit 0. The starting portable v2 JSON and its hash are pinned in the control config. Generated `.yaml` configs connect each completed parent to the next stage. Re-serialized/retrained parent pickles can have a different hash; pin that new hash and a new run name in a separate refit config. Do not weaken the original hash check to label a new parent as the archived one.

For batch work, wrappers request partition `c`, QOS `c_rapid`, account `cms`, one CPU and one hour (refits: 12 GiB). The supplied wrappers contain Robert's checkout/environment/upload paths; adapt those operational paths in your own copy before submitting. The direct commands above avoid those assumptions. Never resume with modified fitting code or stopping policy under an existing run name.

## Reproduce the handover and inspect results

`configs/fpga_handoff_v3.json` pins the exact research-selection file hash, checkpoint hashes and explicit default. `scripts/freeze_support_handoff.py` consumes the verified research JSON/vector exports; it does not train or choose the highest-AUC model. To make another export, use a **new directory/version**:

```bash
python -B scripts/freeze_support_handoff.py --output fpga_handoff/v4
python3 scripts/verify_fpga_handoff.py --bundle fpga_handoff/v4 --compiler c++
# Only after verification and an intentional new selection:
# python -B scripts/freeze_support_handoff.py --output fpga_handoff/v4 --promote
```

Existing bundle directories are never overwritten. Promotion rejects changed files or incomplete C++ checks. `src/tao/portable.py` is the independent Python evaluator/export contract; `fpga_handoff/reference.cpp` is the standalone C++ reference. No fixed-point formats, target device, clock or synthesis results have been chosen yet.

[ROC](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/support_refit_scan_v1_roc.png) · [timing/parameter table](https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/current_winner_comparison.png) · [full scan results](support_refit_scan_results.md) · [training validation and protocol](support_refit_scan.md).

## Mathematical writeup and actual tree equations

See the [research paper](../paper/main.pdf) and its [annotated edition](../paper/annotated.pdf), with shared [LaTeX source](../paper/main.tex) and [build instructions](../paper/README.md). The annotated edition adds optional blue implementation and reproduction notes and prints every cut and affine leaf of Compact refit. `TreePrinter.print_tree` in `src/tao/printing.py` and `scripts/print_tao_tree.py` provide the same read-only inspection; default 17-digit output retains every nonzero coefficient, including tiny nonzeros. The PDF's rounded display listings do not replace the authoritative model JSON or hexadecimal headers.

## XGBoost conifer emulation reference

The [XGBoost emulator study](conifer_xgboost_v1/results.md) now tests the native
model against conifer conversion, the actual CMSSW external and generated HLS
C++ emulation, including fixed-point precision scans. [Setup and version
constraints](conifer_xgboost_v1/README.md) are explicit; the CMSSW 1.7 bridge is
restricted to axis-aligned cuts and constant leaves. The compact converted
reference is under `fpga/conifer/reference/xgboost_v1/`. No FPGA resource or
latency measurements are implied, and the pinned Compact refit default is unchanged.
