# Frozen FPGA handoff comparison

| Model | Parameters | Signal efficiency at 1% background | AUC | Bulk CPU us/event |
|---|---:|---:|---:|---:|
| XGBoost: 100 trees, depth 4 | 3099 | 55.48% | 0.92056 | 1.342 |
| TAO: 3 trees, depth 3 (winner) | 765 | 54.28% | 0.92740 | 1.163 |
| TAO: 3 trees, depth 2 | 357 | 53.75% | 0.92496 | 0.874 |
| TAO: 2 trees, depth 3 | 510 | 53.61% | 0.92624 | 0.832 |
| TAO: 1 trees, depth 5 | 1071 | 53.13% | 0.92639 | 0.882 |
| TAO: 1 trees, depth 4 | 527 | 52.60% | 0.92364 | 0.649 |

Validation only; test remains held out. Saved checkpoint hashes and training status are frozen in the handoff manifest. CPU compares the current software APIs, not FPGA latency. The three depth-2 tree training run was still active when its snapshot was frozen.
