# Response to comments in main.tex (GitHub commit 0de8171)

- Rewrote the abstract for readers unfamiliar with the project. It introduces
  the method and ends with the application and compact model result.
- Replaced the implementation-note framing with a research-paper introduction,
  self-contained model, algorithm, experimental protocol, results and discussion.
  Removed repeated statements about what the work does not address.
- Introduced a single optional `implementation` environment with blue background
  and page breaks. Code names, repository links, hashes, commands, checkpoint
  identities and handover details occur only there. The first page has no boxes.
- Made `main.pdf` the clean paper and supplied `annotated.pdf` with the blue notes
  and full printed forest. Both are built from the same scientific source.
- Replaced storage-index notation in the science by root-to-node paths, defined
  every model component, and explained why a constant cut creates unreachable
  descendants. Heap and array indices are explained only in the blue supplement.
- Replaced the blackboard indicator glyph by a bold numeral 1 and explicitly
  defined its meaning. Checked its rendering in both editions.
- Kept numerical algorithm settings and statistical limitations in the scientific
  text, so hiding implementation notes removes no essential method or evidence.
- Regenerated the SFR comparison with best eligible refits (not held diagnostics)
  and the sparsity figure with scientific model labels rather than run IDs.
  Example equations now use defined feature coordinates and path labels.

No numerical training or inference code, frozen model coefficients, model
selection or ongoing training jobs were changed.
