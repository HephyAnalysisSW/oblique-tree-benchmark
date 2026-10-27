# XGBoost fixed-point emulation results

Full validation: 132,800 events; the test partition was not evaluated.

Native raw-margin ROC: signal efficiency at 1% background **55.4821%**, AUC **0.9205588**.

Fixed-point columns give total input/threshold and score widths. All use `AP_RND_CONV, AP_SAT`; integer widths and complete type strings are in the report.

| Emulator | Input / score bits | Signal eff. at 1% bkg [%] | AUC | Realizable signal eff. [%] | Accepted bkg / 525 | CMSSW/HLS mismatches |
|---|---:|---:|---:|---:|---:|---:|
| double | native double | 55.4821 | 0.9205588 | 55.4821 | 525 | 0 |
| float | native float | 55.4821 | 0.9205588 | 55.4821 | 525 | 0 |
| fixed_f18 | 23 / 24 | 55.4821 | 0.9205592 | 55.4821 | 525 | 0 |
| fixed_f12 | 17 / 18 | 55.4911 | 0.9205559 | 55.4908 | 525 | 0 |
| fixed_f10 | 15 / 16 | 55.4571 | 0.9205569 | 55.4547 | 525 | 0 |
| fixed_f8 | 13 / 14 | 55.5578 | 0.9204776 | 55.5306 | 524 | 0 |
| fixed_f6 | 11 / 12 | 55.4725 | 0.9204009 | 55.2865 | 522 | 0 |
| fixed_f4 | 9 / 10 | 52.0484 | 0.9175962 | 50.8451 | 477 | 0 |

The realizable column uses a strict score threshold and at most 525 background events. It differs from interpolation when quantization creates tied scores.

Native/exported routing checks: **13,280,000** validation leaf IDs and **482,100** boundary leaf IDs agree exactly in the independent conversion walker.
Boundary tests include 324 frozen handover vectors and 4497 predecessor/equality/successor inputs across 1499 internal nodes.

Each HLS C++ profile was compared on 512 inputs, including every handover vector. The reference scores exercise the actual CMSSW conifer external; conversion leaf checks are separately identified.

The converter export is reconciled against native binary32 coefficients and actual topology. Actual maximum depth is 4; the loaded booster configuration reported 6. Both upstream and reconciled diagnostic results are retained in the JSON.

**No FPGA synthesis was run.** CPU emulation timings include I/O and are not hardware latency. Conifer’s export part/clock settings remain placeholders until a device and throughput target are chosen.

[Complete validation/provenance](validation.json) · [ROC](roc.pdf) · [Reproduction instructions](README.md)
