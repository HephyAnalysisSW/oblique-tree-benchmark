# Standalone LaTeX implementation note

`main.tex` and the checked-in `main.pdf` describe the implemented oblique forests,
TAO training, **Select–Freeze–Refit (SFR)**, the fixed benchmark and exact printed
tree functions. The cited bibliography is `references.bib`.

Compile from this directory with a normal TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# or: make
```

All included figures, tables, examples and tree listings are in this folder;
compilation needs no datasets, trained pickles, network connection or Python.
`main.pdf` is intentionally committed; only intermediate TeX files are ignored.

To regenerate data-derived inputs, use the repository's pinned Python environment
from the repository root (see `../docs/DORIAN_HANDOVER.md`):

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/build_paper_inputs.py
# Also regenerate ROC/tradeoff figures, requiring CLIP's archived NPZ/pickles:
python scripts/build_paper_inputs.py --with-roc
```

The generator reads only training provenance and validation arrays, not test
arrays. A results snapshot and actual compact model JSON are included under
`sources/`; source hashes and generated-asset hashes are in `sources/provenance.json`.
The manuscript's frozen comparison is v3, not a moving best score from an
unfinished architecture scan. The dated provisional comparison is identified
separately in the text.

The read-only print API is `tao.printing.TreePrinter.print_tree`. For example:

```bash
python scripts/print_tao_tree.py \
  --model fpga_handoff/v3/compact_refit_3t_d4/standardized/model.json \
  --precision 17 --output /tmp/compact_forest.txt
```

`generated/compact_forest_full_precision.txt` records every affine cut and leaf
with 17 significant digits. The PDF appendix uses nine display digits; the
main-text worked equations use eight and are explicitly marked as rounded.
Only exact-zero terms are omitted. No magnitude pruning or cut normalization
is applied. The JSON/hexadecimal headers remain the firmware authority.

The printing utility is separate from the pinned fitting files, so writing this
note does not change mathematical training code, current coefficients or the
ongoing jobs' implementation identities.
