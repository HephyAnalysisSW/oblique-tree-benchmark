"""Publish a completed validation report and portable conifer reference in Git.

This copies small reviewable artifacts only; workspaces, event arrays, compiled
bridges and HLS build trees remain outside Git. Run from the repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='artifacts/conifer/xgboost_v1')
    args = parser.parse_args()
    source = ROOT/args.source
    report = json.loads((source/'report.json').read_text())
    assert report['complete'] and report['scope']=='full_validation'
    assert report['validation_events']==132800 and not report['test_evaluated']
    assert report['native_leaf_conversion_checks']['all_exact']
    assert len(report['profiles'])==8
    for row in report['profiles']:
        assert 'hls_cpp_emulation' in row
        if 'fractional_bits' in row:
            assert row['hls_cpp_emulation']['error_against_cmssw']['exact_mismatches']==0
    destination=ROOT/'docs/conifer_xgboost_v1'
    reference=ROOT/'fpga/conifer/reference/xgboost_v1'
    destination.mkdir(parents=True,exist_ok=True);reference.mkdir(parents=True,exist_ok=True)
    converted=json.loads((source/'model.json').read_text())
    converted={k:v for k,v in converted.items() if k not in ['config','metadata']}
    target=reference/'model.json'
    if target.exists():
        assert json.loads(target.read_text())==converted,'Reference changed: publish a new version explicitly'
    target.write_text(json.dumps(converted,separators=(',',':'))+'\n')
    shutil.copy2(source/'report.json',destination/'validation.json')
    for name in ['roc.pdf','roc.png']:
        shutil.copy2(source/name,destination/name)
    manifest={'native_model_sha256':report['model_sha256'],'conifer_json_sha256':sha(target),
              'validation_report_sha256':sha(destination/'validation.json'),
              'finite_binary32_inputs_only':True,'score':'increasing raw signal margin',
              'source':'fpga_handoff/v3/xgboost_reference/model.json',
              'configuration':'configs/conifer_xgboost_emulation.json',
              'versions':report['versions'],'ranges':report['ranges'],
              'precision_profiles':report['precision_profiles'],
              'note':'A separate XGBoost reference export. The v3 Compact refit default is unchanged.'}
    (reference/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    base=report['native_margin_metrics']; rows=report['profiles']
    lines=['# XGBoost fixed-point emulation results','',
           'Full validation: 132,800 events; the test partition was not evaluated.',
           '','Native raw-margin ROC: signal efficiency at 1% background '
           f"**{100*base['signal_efficiency_at_1pct_background']:.4f}%**, AUC **{base['auc']:.7f}**.",
           '', 'Fixed-point columns give total input/threshold and score widths. All use '
           '`AP_RND_CONV, AP_SAT`; integer widths and complete type strings are in the report.', '',
           '| Emulator | Input / score bits | Signal eff. at 1% bkg [%] | AUC | Realizable signal eff. [%] | Accepted bkg / 525 | CMSSW/HLS mismatches |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['metrics'];h=row['hls_cpp_emulation']
        bits=f"{row['input_bits']} / {row['score_bits']}" if 'fractional_bits' in row else 'native '+row['name']
        lines.append(f"| {row['name']} | {bits} | {100*m['signal_efficiency_at_1pct_background']:.4f} | {m['auc']:.7f} | {100*m['strict_signal_efficiency']:.4f} | {m['accepted_background']} | {h['error_against_cmssw']['exact_mismatches']} |")
    leaf=report['native_leaf_conversion_checks'];edge=report['boundary_tests'];cor=report['conversion_reconciliation']
    lines+=['','The realizable column uses a strict score threshold and at most 525 background events. '
            'It differs from interpolation when quantization creates tied scores.', '',
            f"Native/exported routing checks: **{leaf['validation_comparisons']:,}** validation leaf IDs and "
            f"**{leaf['boundary_comparisons']:,}** boundary leaf IDs agree exactly in the independent conversion walker.",
            f"Boundary tests include {edge['handover_vectors']} frozen handover vectors and {edge['node_edge_vectors']} "
            f"predecessor/equality/successor inputs across {edge['covered_internal_nodes']} internal nodes.", '',
            f"Each HLS C++ profile was compared on {rows[0]['hls_cpp_emulation']['events']} inputs, including every handover vector. "
            'The reference scores exercise the actual CMSSW conifer external; conversion leaf checks are separately identified.', '',
            'The converter export is reconciled against native binary32 coefficients and actual topology. '
            f"Actual maximum depth is {cor['max_depth_after']}; the loaded booster configuration reported {cor['max_depth_before']}. "
            'Both upstream and reconciled diagnostic results are retained in the JSON.', '',
            '**No FPGA synthesis was run.** CPU emulation timings include I/O and are not hardware latency. '
            'Conifer’s export part/clock settings remain placeholders until a device and throughput target are chosen.', '',
            '[Complete validation/provenance](validation.json) · [ROC](roc.pdf) · [Reproduction instructions](README.md)']
    (destination/'results.md').write_text('\n'.join(lines)+'\n')
    print(destination/'results.md')


if __name__=='__main__':main()
