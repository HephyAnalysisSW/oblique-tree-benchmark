"""Publish reviewable results from a completed two-candidate emulator run."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'artifacts/conifer/leaf_competition_v1';out=ROOT/'docs/leaf_competition_v1'
    report=json.loads((source/'report.json').read_text())
    assert report['complete'] and report['scope']=='full_validation' and report['events']==132800
    assert not report['test_evaluated'] and not report['synthesis_run']
    assert len(report['cpp_fixture_checks'])==16 and all(c['passed'] for c in report['cpp_fixture_checks'])
    for path,digest in report['source_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    out.mkdir(parents=True,exist_ok=True)
    for name in ['report.json','roc.pdf','roc.png']:shutil.copyfile(source/name,out/('validation.json' if name=='report.json' else name))
    cpu={c['leaf_mode']:next(r for r in c['profiles'] if r['name']=='double')['median_microseconds_per_event'] for c in report['candidates']}
    (ROOT/'paper/generated/leaf_competition_macros.tex').write_text('\\newcommand{\\LeafConstantCPU}{'+f"{cpu['constant']:.3f}"+'}\n'+'\\newcommand{\\LeafLinearCPU}{'+f"{cpu['linear']:.3f}"+'}\n')
    lines=['# Constant versus linear leaf implementation candidates','',
        'All 27 constant-leaf scan configurations converged. The winner is unchanged.',
        'Joint model entry point: [`CURRENT_CANDIDATES.json`](../../fpga_handoff/CURRENT_CANDIDATES.json).',
        'The [manifest](../../fpga_handoff/leaf_competition_v1/manifest.json) pins model/checkpoint hashes, training setups and selected epochs.',
        '', '| Candidate | Layout | Dense / nonzero | Eff. interpolated / realizable [%] | Sparse path MUL / comparisons | All reachable MUL | C++ [µs/event] |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in report['candidates']:
        a=c['arithmetic'];r=next(r for r in c['profiles'] if r['name']=='double');m=r['metrics']
        layout='5 × D4 constant' if c['leaf_mode']=='constant' else '3 × D4 linear'
        lines.append(f"| {c['id']} | {layout} | {a['dense_slots']} / {a['nonzero_coefficients']} | {100*m['signal_efficiency_at_1pct_background']:.3f} / {100*m['strict_signal_efficiency']:.3f} | {a['sparse_path_multiplications_bound']} / {a['sparse_path_comparisons_bound']} | {a['reachable_all_node_multiplications']} | {r['median_microseconds_per_event']:.3f} |")
    lines+=['', f"**Measured CPU gain:** the linear candidate is about {cpu['constant']/cpu['linear']:.1f} times faster in this matched sparse C++ traversal. This is not an FPGA latency result.",
        '', 'Both use the same single-thread C++ sparse path evaluator, compiler flags and input batch; one warmup and nine timed repeats. Input conversion, JSON parsing and I/O are excluded. Input/coefficient/output types are binary64; no prediction cache is used. Times are for this batch and host, not independent hardware-algorithm rankings.',
        '', f"Host: `{report['host']}`; compiler: `{report['compiler']}`. Full validation: 132,800 events; no test arrays loaded.",
        '', '## Storage and arithmetic',
        '', 'At 64 bits per nonzero coefficient, value payload alone is 10,000 bytes (constant) versus 3,344 bytes (linear), a factor 2.99. These are exact-zero compressed coefficient values, not the allocated dense arrays or actual process memory; topology, sparse indices, padding and preprocessing are excluded. Reachability can further reduce the linear payload from 418 to 379 coefficients.',
        '', 'Path counts bound sequential traversal. All-reachable counts describe computing every surviving affine function, closer to an unrolled implementation. Stock conifer HLS currently emits dense weighted sums for all nodes; synthesis must establish zero/constant elimination and the eventual DSP/LUT/BRAM mapping. Scalar multiplication counts are not DSP counts. No resource-utilization or hardware-latency winner is claimed.',
        '', '## Common reduced-precision check',
        '', 'Cuts are scaled by a positive power of two per node before quantization, including the bias. Downward scaling is constrained not to flush any nonzero binary64 coefficient. This preserves canonical full-precision routing on every validation event; canonical handover JSON/headers remain unchanged. Leaves are not rescaled.',
        '', 'All fixed-point types use nearest-even rounding and saturation (`AP_RND_CONV, AP_SAT`). Fractional width F is common; input/threshold integer width is 10, weight width 6, score/leaf-bias width 16, each including sign. Dot products accumulate in the input/threshold type; affine leaf values accumulate in the score type. Input and coefficient casts precede multiplication; each accumulation is cast back to its accumulator type. Scores are summed without a sigmoid. Training-box absolute partial-sum bounds and guard margins are checked before evaluation. These deliberately conservative common formats have not been minimized for either candidate.',
        '', '| Candidate | Fractional bits | Input / weight / score bits | Interpolated eff. [%] | Realizable eff. [%] | AUC |',
        '|---|---:|---:|---:|---:|---:|']
    table=[]
    for c in report['candidates']:
        for r in c['profiles']:
            if r['name']=='double':continue
            f=int(r['name'][1:]);m=r['metrics']
            lines.append(f"| {c['id']} | {f} | {f+10} / {f+6} / {f+16} | {100*m['signal_efficiency_at_1pct_background']:.3f} | {100*m['strict_signal_efficiency']:.3f} | {m['auc']:.5f} |")
            label='Constant' if c['leaf_mode']=='constant' else 'Linear'
            table.append(f"{label} & {f} & {100*m['signal_efficiency_at_1pct_background']:.2f} & {100*m['strict_signal_efficiency']:.2f} "+r'\\')
    (ROOT/'paper/generated/leaf_competition_precision.tex').write_text('\n'.join(table)+'\n'+r'\bottomrule'+'\n')
    lines+=['', 'The constant fixed-point scores agree exactly between the independent sparse reference and **unmodified conifer 1.11 C++ on all 132,800 validation events**. Standard conifer HLS C++ array-interface checks pass on 512 vectors at double and F=12 precision; F=12 is bit-exact, double differs only by floating-point summation order. No vendor synthesis tool was invoked. The modern package header is used, not the older axis-only CMSSW 1.7 header.',
        '', 'The linear model is evaluated by our independent affine-leaf C++ reference. This is a tested specification for a future conifer extension, not a claim that stock conifer supports affine leaves. The toy C++ checks exercise tiny cut coefficients, left equality, rounding, input-independent cuts and both leaf types at all four precisions. Full-precision leaf assignments match the canonical model for both candidates on every validation event.',
        '', '## Reproduce',
        '', 'Portable golden checks need only Python standard library and a C++17 compiler:',
        '', '```bash','python3 scripts/verify_fpga_handoff.py --bundle fpga_handoff/leaf_competition_v1 \\',
        '  --report /tmp/leaf_competition_verification.json','```',
        '', 'The constant winner has 458 golden vectors in each input space; Compact refit retains its 270 v3 vectors. This checks canonical coefficients/headers, raw and standardized predictions and all leaf indices. See [`portable_verification.json`](portable_verification.json).',
        '', 'For full emulation, first use the [existing EL9/CMSSW conifer setup](../conifer_xgboost_v1/README.md). Its pinned Python environment and CMSSW arbitrary-precision/JSON headers suffice; no Vivado/Vitis license is needed:',
        '', '```bash','scripts/setup_conifer_workspace.sh','scripts/run_leaf_competition.sh',
        '# Or submit on CLIP:','scripts/submit_leaf_competition.sh',
        '# Publish only after the full report passes:','python scripts/publish_leaf_competition.py','```',
        '', 'Generated projects are in ignored `artifacts/conifer/leaf_competition_v1/`. The runner uses only training-range metadata and validation inputs/labels. `fpga/leaf_competition/models.py` implements the export adapter, positive cut scaling and arithmetic accounting; `fpga/leaf_competition/emulate.cc` implements the shared sparse reference.',
        '', 'The frozen constant bundle was created once with `PYTHONPATH=src:. python scripts/freeze_leaf_competition.py` in the training environment. It refuses to overwrite an existing bundle. Training reproduction is specified by the configuration and checkpoint fields in `constant_5t_d4/candidate.json`; linear training/refit setup remains in the v3 handover.',
        '', 'Training pointers (run in your own checkout with the original fixed NPZ and pinned training environment):',
        '', '- Constant winner: `configs/hyper_scan_v1/case_51.json`, `scripts/train_tao_fully.py --config configs/hyper_scan_v1/case_51.json --index 0`. The run uses five D4 trees, CART seed 42, hard-threshold search and zero split/leaf L1. Re-run the same command after clean wall-time continuations until the convergence marker is complete; the selected snapshot is epoch 23 of a source run converged at epoch 36.',
        '- Linear Compact refit: `configs/support_refit_scan_v1/split300.json`, `scripts/run_support_refit_scan_case.py`, `scripts/train_tao_support_refit.py` and `configs/tao_support_refit_scan_v1.json`. Support selection uses split L1 0.003 and leaf L1 1e-5; frozen-support refit sets both to zero. The exact selected full-refit checkpoint is epoch 8, while its source stage converged later. Use the archived v3 checkpoint/hash for bitwise reproduction of this winner; rerunning optimizers is not promised to produce identical coefficients.',
        '', '[ROC comparison](roc.pdf) · [full provenance and per-profile errors](validation.json)',
        '', 'Next hardware decision: implement affine leaves in the conifer HLS backend, then synthesize both at a matched device, clock, interface, throughput and validated precision/working point.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
