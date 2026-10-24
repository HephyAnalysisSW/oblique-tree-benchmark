# Frozen FPGA handoff comparison

| Model | Parameters | Signal efficiency at 1% background | AUC | Bulk CPU us/event |
|---|---:|---:|---:|---:|
| TAO: 3 trees, depth 4, linear (winner) | 1581 | 55.78% | 0.92758 | 2.143 |
| TAO: 3 trees, depth 4, linear | 1581 | 55.57% | 0.92498 | 2.133 |
| XGBoost: 100 trees, depth 4 | 3099 | 55.48% | 0.92056 | 1.336 |
| TAO: 1 trees, depth 4, linear | 527 | 55.22% | 0.92683 | 0.790 |
| TAO: 5 trees, depth 4, constant | 1355 | 55.11% | 0.92529 | 3.155 |
| TAO: 3 trees, depth 3, linear | 765 | 54.28% | 0.92740 | 1.480 |

Validation only; test remains held out. Saved checkpoint hashes and training status are frozen in the handoff manifest. CPU compares the current software APIs, not FPGA latency. Source training status is recorded per candidate; some runs may still be active. Snapshot selection does not certify convergence.
