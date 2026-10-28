# Constant-leaf factorial study

Converged source runs: 27/27.

| Trees | Depth | Split L1 | Dense / nonzero | Interpolated signal efficiency [%] | Realizable signal efficiency [%] | Training compute [min] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1e-07 | 55 / 55 | 14.978 | 0.000 | 57.6 |
| 1 | 3 | 1e-05 | 127 / 126 | 38.607 | 0.000 | 104.0 |
| 1 | 4 | 0 | 271 / 256 | 45.355 | 45.332 | 17.3 |
| 3 | 2 | 1e-05 | 165 / 129 | 30.722 | 0.000 | 136.8 |
| 3 | 3 | 0 | 381 / 336 | 49.012 | 38.807 | 90.2 |
| 3 | 4 | 0 | 813 / 753 | 53.536 | 53.300 | 195.4 |
| 5 | 2 | 1e-07 | 275 / 275 | 47.774 | 47.013 | 297.8 |
| 5 | 3 | 1e-07 | 635 / 615 | 50.940 | 50.310 | 636.5 |
| 5 | 4 | 0 | 1355 / 1250 | 55.400 | 54.755 | 344.9 |

Every selected snapshot is archived by SHA-256. Selected and terminal epochs, convergence flags, configuration and metrics are separate in the JSON.

Pending cases: none.

Constant leaves have no penalized feature weights. The nominal leaf penalty does not affect their objective.

Validation only. This controlled CART-initialized cohort is separate from the older seven-layout random-initialization study.
