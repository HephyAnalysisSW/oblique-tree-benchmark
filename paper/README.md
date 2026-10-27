# Research paper and optional implementation notes

- **`main.pdf`** is the scientific paper, with implementation boxes hidden.
- **`annotated.pdf`** contains the identical scientific text plus blue, page-breakable
  implementation boxes, reproducibility instructions, and complete tree printouts.
- **`main.tex`** is the common source. `annotated.tex` enables the boxes and includes it.
- **`references.bib`** contains the cited references; each edition builds its own bibliography.

Compile with TeX Live and `latexmk` from this directory:

```bash
make
# Or build either edition:
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error annotated.tex
```

To show the blue notes directly in `main.tex`, change
`\implementationdetailsfalse` to `\implementationdetailstrue`. The environment
is named `implementation`. With it disabled, its content is excluded before
LaTeX processes it. Scientific definitions, algorithms, numerical settings,
results and qualifications remain in the paper. All file paths, code/API names,
hashes, shell commands and storage conventions belong in the blue notes.

All inputs are included here. Both PDFs build without datasets, model pickles,
a Python training environment or a network connection. PDFs are intentionally
committed; intermediate TeX files are ignored.

## Data-derived inputs

The frozen v3 results and actual model are recorded in `sources/`; the original
implementation-note generator and its provenance remain available. From the
repository root, using the pinned training environment:

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/build_paper_inputs.py
# Regenerate the native-model ROCs as well (requires CLIP archives):
python scripts/build_paper_inputs.py --with-roc
```

The scientific table, path-labelled example equations and sparsity figure are
derived solely from the checked-in frozen snapshots:

```bash
python3 paper/build_scientific_assets.py
# Regenerate the sparsity figure too; requires matplotlib, but no datasets:
python3 paper/build_scientific_assets.py --plot
```

`research_sfr_results.tex` compares eligible leaf/full refits, excluding held
full-stage diagnostics. `sources/scientific_assets.json` records input/output
and generator hashes. The scientific figure uses model names instead of run IDs.
`research_cuts.tex` and `research_tiny_cut.tex` use feature coordinates and
root-to-node paths, with coefficients rounded to eight significant digits.

The read-only printer is `tao.printing.TreePrinter`. For a full-precision record:

```bash
python scripts/print_tao_tree.py \
  --model fpga_handoff/v3/compact_refit_3t_d4/standardized/model.json \
  --precision 17 --output /tmp/compact_forest.txt
```

`generated/compact_forest_full_precision.txt` contains all affine cuts and leaf
functions at 17 significant digits. The annotated supplement uses nine display
digits. Exact-zero terms are omitted; small nonzero terms are retained. The
portable JSON and hexadecimal headers are the inference authority.

The paper distinguishes the validation-only oblique study from the historical
XGBoost reproduction check, which evaluated the test split. The frozen model
selection, coefficients and numerical training code are unchanged by this revision.

## Constant leaves and fixed-point reference (27 October 2026)

The constant-leaf section is an explicitly provisional snapshot: **26/27**
controlled CART-initialized configurations have converged. Case 52 (five trees,
depth four, split L1 `1e-7`) is still training and is excluded. The best eligible
constant model is separate from the frozen v3 Compact refit handover default.

`constant_leaf_study.tex` reads tables/macros from `generated/` and provenance
from `sources/constant_leaf_study.json`. With the archived checkpoints and fixed
dataset available on CLIP, regenerate using the pinned training environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python scripts/build_constant_leaf_study.py --require-complete
```

Run this from the repository root. The final gate fails while any source case
is incomplete. It records selected and terminal checkpoints separately, checks
validation metrics against saved metrics, and archives selected pickles by hash.
ROC timing labels use nine uncached prediction repeats; XGBoost uses
`Booster.inplace_predict` on preconverted float32 inputs, TAO uses its NumPy API
on float64 inputs. They measure current batch APIs, not FPGA latency.

The focused continuation monitor `scripts/prioritize_constant_scan.py` coordinates
clean continuations through the existing scheduler. It regenerates numerical
assets after all 27 constant cases converge, but does not compile, commit or
publish a new paper automatically. Review the regenerated results, rebuild both
editions and inspect their rendered pages before publishing an updated snapshot.

`xgboost_emulation.tex` documents the fixed-point quantizer, range bounds and
precision comparison. Regenerate its table with `python3 paper/build_precision_table.py`.
Its numbers are frozen from
`docs/conifer_xgboost_v1/validation.json`; the full reproduction commands are in
`docs/conifer_xgboost_v1/README.md`. No FPGA synthesis result is claimed.
