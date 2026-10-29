# Three tree families versus XGBoost: paper and implementation edition

`main.pdf` is the self-contained research paper. `annotated.pdf` contains the
same scientific text plus optional blue implementation boxes and the complete
Compact forest printout. Both end with the **same ten-model comparison table**,
after references, supporting appendices and optional printouts.

The main narrative compares axis/linear, oblique/constant and oblique/linear
forests with the fixed XGBoost reference. Detailed objective, leaf and cut
updates remain in the main text, including the exhaustive axis search.
Support-refit, all four axis layouts, all nine constant layouts, earlier
fixed-point checks, conditional uncertainty and actual cuts are supporting
appendices. The discussion separates scalar storage, selected-path and all-node
work, common C++ throughput, and unmeasured FPGA resources.

## Build either edition

From this directory with TeX Live and `latexmk`:

```bash
make
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error annotated.tex
```

All inputs are included under `paper/`: neither datasets, pickles, a training
environment nor network access are needed to compile the PDFs. PDFs are
intentionally committed; auxiliary TeX files are ignored. `annotated.tex`
defines `ShowImplementation`; alternatively change `implementationdetailsfalse`
to `implementationdetailstrue` in `main.tex`. The environment is named
`implementation`; with it disabled, its contents are excluded before LaTeX
processes them. Scientific definitions and algorithms do not depend on the boxes.

## Authoritative data and handover

- `sources/ten_model_comparison.json` freezes the measured ten-row report.
- `sources/axis_linear_study.json` freezes the four-case axis cohort.
- `sources/constant_leaf_study.json` freezes all 27 completed constant cases.
- `sources/results_snapshot.json` retains the earlier support/refit study.
- `../fpga_handoff/CURRENT_COMPARISON.json` pins all ten portable models and
  their verification, including the three family winners and Compact alternative.

The final rows were independently checked on the full 132,800-event validation
partition. All leaf assignments match canonical TAO/native XGBoost inference.
The common C++ kernel uses binary64 arithmetic and XGBoost binary32 input
semantics; its XGBoost sum differs slightly from native floating-point
accumulation. No new test evaluation, ten-model quantization or FPGA synthesis
is claimed. The axis 3×D4 unpenalized refit was held for separation; its
converged regularized source remains selected. The 837-coefficient oblique
model is a completed leaf-only stage, not its held full-stage diagnostic.

## Regenerate the final table and figures

From the repository root:

```bash
# Standard library only: regenerate scientific table and complete axis cohort.
python3 paper/build_comparison_assets.py
# Requires matplotlib, archived validation scores and fixed dataset on CLIP:
python3 paper/build_comparison_assets.py --plot
# Re-run the actual benchmark in the pinned EL9 environment:
scripts/run_ten_model_comparison.sh
```

The benchmark may produce different CPU times on another host; the paper's
frozen report preserves its measured snapshot. Regenerating plots verifies the
dataset hash. `sources/final_comparison_assets.json` records source/generator
and generated asset hashes. `generated/final_comparison.tex` is included only
by `final_comparison.tex`, which is the last input in both editions.

The earlier supporting inputs retain their original generators:
`build_paper_inputs.py`, `build_constant_leaf_study.py`,
`publish_leaf_competition.py`, `build_scientific_assets.py` and
`build_precision_table.py`. Rebuilding a historical subset does not replace the
final measured table. Exact training/environment and reproduction pointers are
in `docs/DORIAN_HANDOVER.md` at repository level and the optional blue notes.

## Actual model equations

`generated/compact_forest_full_precision.txt` records all Compact cuts and
leaves at 17 significant digits; the optional supplement displays nine digits.
`tao.printing.TreePrinter` and `scripts/print_tao_tree.py` are read-only. Only
exact-zero terms are omitted; tiny nonzero cuts remain. Binary portable JSON
and hexadecimal headers are the inference authority, not rounded printed text.

Before publishing an edited edition, compile both, check references and layout,
render and inspect their pages, and verify that the ten-model table is the
last page in each. Keep historical model bundles immutable.
