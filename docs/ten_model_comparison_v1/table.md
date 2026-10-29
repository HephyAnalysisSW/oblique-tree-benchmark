| Model | Layout | Learned scalar nnz | Extra feature indices | Efficiency [%] | Realizable [%] | AUC | C++ µs/event | MUL path / all | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| XGBoost reference | 100 × D4 | 3099 | 1499 | 55.48 | 55.48 | 0.92056 | 3.675 | 0 / 0 | Frozen reference |
| Axis / linear | 3 × D4 | 479 | 45 | 56.32 | 56.32 | 0.92576 | 0.101 | 35 / 386 | Regularized; refit held |
| Axis / linear | 1 × D4 | 228 | 15 | 54.74 | 54.74 | 0.92387 | 0.048 | 16 / 197 | Leaf refit converged |
| Axis / linear | 1 × D3 | 132 | 7 | 50.10 | 50.10 | 0.92020 | 0.049 | 16 / 117 | Leaf refit converged |
| Oblique / constant | 5 × D4 | 1250 | 0 | 55.40 | 54.75 | 0.92610 | 0.965 | 272 / 1088 | Regularized; source converged |
| Oblique / constant | 3 × D4 | 753 | 0 | 53.54 | 53.30 | 0.92318 | 0.542 | 160 / 656 | Regularized; source converged |
| Oblique / constant | 1 × D4 | 256 | 0 | 45.36 | 45.33 | 0.91247 | 0.171 | 64 / 224 | Regularized; source converged |
| Oblique / linear | 5 × D4 | 1275 | 0 | 57.73 | 57.73 | 0.93090 | 0.412 | 212 / 1096 | Regularized; source converged |
| Oblique / linear | 3 × D4 | 837 | 0 | 57.12 | 57.12 | 0.92967 | 0.320 | 160 / 734 | Leaf refit converged |
| Oblique / linear | 3 × D4 | 418 | 0 | 56.20 | 56.20 | 0.92733 | 0.167 | 78 / 325 | Full refit converged |
