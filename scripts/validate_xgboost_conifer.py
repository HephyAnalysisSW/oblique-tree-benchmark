"""Validate frozen XGBoost -> conifer -> CMSSW/HLS C++ fixed-point emulation.

Run via run_xgboost_conifer.sh. Reads X_train for range selection and X_val/y_val
for evaluation; never reads a test array. Generated projects stay untracked.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import conifer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
from fpga.conifer.compat import reconcile
from evaluate import compute_metrics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(content, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def metrics(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    bg = np.sort(score[y == 0])[::-1]
    budget = int(.01*len(bg))
    threshold = float(bg[budget])
    accepted = score > threshold
    return {'signal_efficiency_at_1pct_background': float(np.interp(.01, fpr, tpr)),
            'auc': float(roc_auc_score(y, score)), 'strict_threshold': threshold,
            'accepted_background': int(accepted[y == 0].sum()), 'background_budget': budget,
            'strict_background_efficiency': float(accepted[y == 0].mean()),
            'strict_signal_efficiency': float(accepted[y == 1].mean())}


def write_inputs(path, x):
    with open(path, 'wb') as stream:
        stream.write(struct.pack('<8sQI', b'TBXGB01\0', len(x), x.shape[1]))
        np.asarray(x, dtype='<f8').tofile(stream)


def boundaries(native, x, leaf_indices):
    rows, cases = [], []
    for t, tree in enumerate(native['learner']['gradient_booster']['model']['trees']):
        parent = {}
        for node, (left, right) in enumerate(zip(tree['left_children'], tree['right_children'])):
            if left >= 0:
                parent[left] = node; parent[right] = node
        representative = {}
        leaves, first = np.unique(leaf_indices[:, t], return_index=True)
        for leaf, event in zip(leaves, first):
            node = int(leaf)
            while node in parent:
                node = parent[node]
                representative.setdefault(node, int(event))
        for node in range(len(tree['left_children'])):
            if tree['left_children'][node] < 0 or node not in representative:
                continue
            feature = tree['split_indices'][node]
            threshold = np.float32(tree['split_conditions'][node])
            for name, value in [('below', np.nextafter(threshold, np.float32(-np.inf))),
                                ('equal', threshold),
                                ('above', np.nextafter(threshold, np.float32(np.inf)))]:
                row = x[representative[node]].copy(); row[feature] = value
                rows.append(row); cases.append({'tree': t, 'node': node, 'side': name})
    return np.asarray(rows, dtype=np.float32), cases


def error(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return {'max_abs': float(np.max(np.abs(a-b), initial=0)),
            'mean_abs': float(np.mean(np.abs(a-b))), 'exact_mismatches': int(np.count_nonzero(a != b))}


def run_external(binary, profile, model, inputs, output, trace=None):
    command = [str(binary), profile, str(model), str(inputs), str(output)]
    if trace:
        command.append(str(trace))
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    scores = np.fromfile(output, dtype='<f8')
    return scores, {'wall_seconds_including_io': time.perf_counter()-started,
                    'stderr': completed.stderr.strip()}


def integer_bits(bound):
    return max(2, 1+int(math.ceil(math.log2(max(1., bound+1.)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/conifer_xgboost_emulation.json')
    parser.add_argument('--output', default='artifacts/conifer/xgboost_v1')
    parser.add_argument('--max-events', type=int, help='Smoke test only; omit for the full validation partition')
    parser.add_argument('--hls-events', type=int)
    parser.add_argument('--reuse-hls', help='Reuse compiled HLS C++ bridges from a completed compatible report directory')
    parser.add_argument('--skip-hls', action='store_true', help='Diagnostic only: not a completed chain validation')
    args = parser.parse_args()
    cfg = json.loads((ROOT/args.config).read_text())
    out = (ROOT/args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    report = {'complete': False, 'test_evaluated': False, 'configuration': cfg,
              'versions': {p: importlib.metadata.version(p) for p in ['conifer','xgboost-cpu','numpy','scikit-learn','pandas','pybind11']},
              'python': sys.version, 'cmssw': os.environ.get('CMSSW_VERSION'),
              'scram_arch': os.environ.get('SCRAM_ARCH'),
              'compiler': subprocess.check_output(['g++','--version'], text=True).splitlines()[0],
              'model_sha256': digest(ROOT/cfg['model']), 'dataset_sha256': digest(ROOT/cfg['dataset']),
              'score_semantics': 'increasing signal raw margin; no sigmoid in emulator',
              'scope': 'smoke' if args.max_events else 'full_validation', 'profiles': []}
    save(out/'report.json', report)
    manifest = json.loads((ROOT/cfg['manifest']).read_text())
    assert report['model_sha256'] == manifest['files']['xgboost_reference/model.json']
    assert report['dataset_sha256'] == cfg['dataset_sha256']
    native = json.loads((ROOT/cfg['model']).read_text())
    with np.load(ROOT/cfg['dataset'], allow_pickle=False) as data:
        train = data['X_train']
        train_bound = float(np.max(np.abs(train)))
        assert np.isfinite(train).all()
        del train
        x = np.asarray(data['X_val'], dtype=np.float32)
        y = data['y_val'].astype(int)
    assert len(x) == cfg['validation_events'] and x.shape[1] == cfg['features']
    assert np.isfinite(x).all(), 'NaN/missing routing is outside this finite-input contract'
    if args.max_events:
        x, y = x[:args.max_events], y[:args.max_events]
    report['validation_events'] = len(x)
    booster = xgb.Booster(); booster.load_model(ROOT/cfg['model']); booster.set_param({'nthread': 1})
    matrix = xgb.DMatrix(x)
    native_margin = booster.predict(matrix, output_margin=True)
    native_probability = booster.predict(matrix)
    native_leaves = booster.predict(matrix, pred_leaf=True).astype(np.uint16)
    report['native_margin_metrics'] = metrics(y, native_margin)
    report['native_probability_metrics'] = metrics(y, native_probability)
    shared, _ = compute_metrics(y, np.column_stack([1-native_probability,native_probability]), ['quark/gluon','W/Z/top'])
    report['shared_evaluation_probability_metrics'] = shared
    assert abs(shared['signal_efficiency_at_fixed_fpr']-report['native_probability_metrics']['signal_efficiency_at_1pct_background']) < 1e-12
    official = conifer.converters.convert_from_xgboost(booster, {'Backend': 'python', 'ProjectName': 'xgb_python', 'OutputDir': str(out/'python')})
    official.save(str(out/'upstream_conversion.json'))
    upstream = json.loads((out/'upstream_conversion.json').read_text())
    converted, corrections = reconcile(native, upstream)
    assert converted['n_trees'] == cfg['trees'] and converted['max_depth'] == cfg['max_depth']
    save(out/'model.json', converted)
    report['conversion_reconciliation'] = corrections
    sample = min(256, len(x))
    report['upstream_python_margin_error_sample'] = error(official.decision_function(x[:sample]), native_margin[:sample])
    corrected_py = conifer.model.make_model(converted, {'Backend': 'python', 'ProjectName': 'xgb_python', 'OutputDir': str(out/'python')})
    report['reconciled_python_margin_error_sample'] = error(corrected_py.decision_function(x[:sample]), native_margin[:sample])
    assert report['reconciled_python_margin_error_sample']['max_abs'] <= cfg['native_margin_absolute_tolerance']
    # Training-derived bounds only: validation does not choose the type ranges.
    thresholds = [abs(v) for group in converted['trees'] for tree in group for f,v in zip(tree['feature'],tree['threshold']) if f >= 0]
    leaf_sum_bound = abs(converted['init_predict'][0]) + sum(max(abs(v) for f,v in zip(g[0]['feature'],g[0]['value']) if f < 0) for g in converted['trees'])
    ib_input = integer_bits(max(train_bound, max(thresholds)))
    # Include worst-case per-tree rounding at the coarsest scanned fraction.
    ib_score = integer_bits(leaf_sum_bound+(len(converted['trees'])+1)*2**(-min(cfg['fractional_bits'])-1))
    report['ranges'] = {'training_max_abs_input': train_bound, 'max_abs_threshold': max(thresholds),
                        'conservative_float_score_bound': leaf_sum_bound,
                        'input_integer_bits': ib_input, 'score_integer_bits': ib_score,
                        'selection': 'training inputs, model thresholds, conservative leaf-sum bound; no validation tuning'}
    profiles = [{'name':'double','input':'double','score':'double','weight':'double'},
                {'name':'float','input':'float','score':'float','weight':'float'}]
    for fraction in cfg['fractional_bits']:
        profiles.append({'name':f'fixed_f{fraction}', 'fractional_bits':fraction,
                         'input':f'ap_fixed<{ib_input+fraction},{ib_input},AP_RND_CONV,AP_SAT>',
                         'score':f'ap_fixed<{ib_score+fraction},{ib_score},AP_RND_CONV,AP_SAT>',
                         'weight':'ap_fixed<2,2,AP_TRN,AP_WRAP>'})
    report['precision_profiles'] = profiles
    workspace = Path(os.environ['CMSSW_BASE'])
    package = workspace/'src/TreeBenchmark/ConiferValidation'
    shutil.copytree(ROOT/'cmssw/TreeBenchmark/ConiferValidation', package, dirs_exist_ok=True)
    # Remove the obsolete generated root build file from early setup attempts.
    if (package/'BuildFile.xml').exists():
        (package/'BuildFile.xml').unlink()
    (package/'bin/PrecisionProfiles.inc').write_text('\n'.join(
        f'if (name == "{p["name"]}") return evaluate<{p["input"]}, {p["score"]}>(args);' for p in profiles)+'\n')
    print('Building CMSSW emulator', flush=True)
    build = subprocess.run(['scram','b','-j','2'], cwd=workspace/'src', text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out/'cmssw_build.log').write_text(build.stdout)
    assert build.returncode == 0, 'See cmssw_build.log'
    binary = workspace/'bin'/os.environ['SCRAM_ARCH']/'validateXGBoostConifer'
    assert binary.is_file()
    report['emulator_sha256'] = digest(binary)
    report['tools'] = {t: subprocess.check_output(['scram','tool','info',t], cwd=workspace/'src', text=True) for t in ['conifer','hls','json']}
    write_inputs(out/'validation.bin', x)
    edge_x, edge_cases = boundaries(native, x, native_leaves)
    golden = np.atleast_2d(np.loadtxt(ROOT/'fpga_handoff/v3/xgboost_reference/vectors.tsv')).astype(np.float32)
    golden_expected = np.loadtxt(ROOT/'fpga_handoff/v3/xgboost_reference/expected_signal_probability.tsv')
    np.testing.assert_allclose(booster.predict(xgb.DMatrix(golden)), golden_expected, rtol=0, atol=1e-7)
    boundary_x = np.concatenate([golden, edge_x])
    boundary_matrix = xgb.DMatrix(boundary_x)
    boundary_native = booster.predict(boundary_matrix, output_margin=True)
    print('Checking upstream conversion on boundary vectors',flush=True)
    report['upstream_python_boundary_margin_error'] = error(official.decision_function(boundary_x),boundary_native)
    boundary_leaves = booster.predict(boundary_matrix, pred_leaf=True).astype(np.uint16)
    write_inputs(out/'boundaries.bin', boundary_x)
    report['boundary_tests'] = {'handover_vectors':len(golden), 'node_edge_vectors':len(edge_x),
                               'covered_internal_nodes':len(edge_cases)//3, 'total':len(boundary_x)}
    save(out/'boundary_cases.json', edge_cases)
    hls_n = max(len(golden), args.hls_events or cfg['hls_emulation_events'])
    if hls_n >= len(boundary_x):
        hls_indices = np.arange(len(boundary_x))
    else:
        hls_indices = np.r_[np.arange(len(golden)), np.linspace(len(golden),len(boundary_x)-1,hls_n-len(golden),dtype=int)]
    hls_x = boundary_x[hls_indices]
    previous = None
    if args.reuse_hls:
        previous = json.loads((ROOT/args.reuse_hls/'report.json').read_text())
        assert previous['complete'] and previous['versions']['conifer']==report['versions']['conifer']
        assert previous['compiler']==report['compiler'] and previous['tools']['hls']==report['tools']['hls']
        old_model = json.loads((ROOT/args.reuse_hls/'model.json').read_text())
        for key in ['trees','n_features','n_classes','n_trees','max_depth','norm','init_predict','splitting_convention']:
            assert old_model[key]==converted[key], 'Incompatible cached HLS model: '+key
    predictions = {'XGBoost': native_margin}
    for p in profiles:
        print('Evaluating',p['name'], flush=True)
        trace = out/'validation_leaves.bin' if p['name']=='double' else None
        score, timing = run_external(binary, p['name'], out/'model.json', out/'validation.bin', out/(p['name']+'.bin'), trace)
        assert len(score) == len(x)
        boundary_trace = out/'boundary_leaves.bin' if p['name']=='double' else None
        boundary_score, _ = run_external(binary, p['name'], out/'model.json', out/'boundaries.bin', out/(p['name']+'_boundaries.bin'), boundary_trace)
        result = dict(p, metrics=metrics(y, score), native_margin_error=error(score,native_margin),
                      boundary_native_margin_error=error(boundary_score,boundary_native), execution=timing)
        if p['name']=='double':
            trace_array = np.fromfile(trace,dtype='<u2').reshape(native_leaves.shape)
            boundary_array = np.fromfile(boundary_trace,dtype='<u2').reshape(boundary_leaves.shape)
            assert np.array_equal(trace_array,native_leaves), 'Exported/native validation leaf mismatch'
            assert np.array_equal(boundary_array,boundary_leaves), 'Exported/native boundary leaf mismatch'
            report['native_leaf_conversion_checks'] = {'validation_comparisons':int(native_leaves.size),
                  'boundary_comparisons':int(boundary_leaves.size),'all_exact':True,
                  'method':'independent converted-tree walker, compared with native XGBoost pred_leaf'}
            assert result['native_margin_error']['max_abs'] <= cfg['native_margin_absolute_tolerance']
            assert result['boundary_native_margin_error']['max_abs'] <= cfg['native_margin_absolute_tolerance']
        if 'fractional_bits' in p:
            low, high = -2**(ib_input-1), 2**(ib_input-1)-2**(-p['fractional_bits'])
            result['validation_input_values_outside_representable_range'] = int(np.count_nonzero((x<low)|(x>high)))
            result['input_bits'] = ib_input+p['fractional_bits']; result['score_bits'] = ib_score+p['fractional_bits']
        if not args.skip_hls:
            print('Compiling HLS C++ emulation',p['name'], flush=True)
            hcfg = conifer.backends.xilinxhls.auto_config()
            hcfg.update(ProjectName='xgb_'+p['name'], OutputDir=str(out/'hls'/p['name']),
                        InputPrecision=p['input'], ThresholdPrecision=p['input'],
                        WeightPrecision=p['weight'], ScorePrecision=p['score'])
            reuse_record = None
            if previous:
                cached = next(r for r in previous['profiles'] if r['name']==p['name'])['hls_cpp_emulation']
                old_cfg = cached['configuration']
                for key in ['InputPrecision','ThresholdPrecision','WeightPrecision','ScorePrecision','Unroll']:
                    assert old_cfg[key]==hcfg[key], 'Cached precision/config differs: '+key
                hcfg = old_cfg
                hls = conifer.model.make_model(converted,hcfg)
                libraries = list(Path(hcfg['OutputDir']).glob('conifer_bridge_*.so'))
                assert len(libraries)==1, 'Expected exactly one compiled bridge in cache'
                library = libraries[0]
                module_spec = importlib.util.spec_from_file_location(library.stem,library)
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                hls.bridge = module
                reuse_record = {'from_report':str((ROOT/args.reuse_hls/'report.json').resolve()),'library_sha256':digest(library)}
            else:
                hls = conifer.model.make_model(converted,hcfg)
                hls.compile()
            hscore = np.asarray(hls.decision_function(hls_x)).reshape(-1)
            comparison = error(hscore,boundary_score[hls_indices])
            result['hls_cpp_emulation'] = dict(events=len(hls_x), error_against_cmssw=comparison,
                    configuration=hcfg, synthesis_run=False, reused_compiled_bridge=reuse_record,
                    target_note='Conifer default part/clock are export placeholders, not a chosen resource benchmark')
            if 'fractional_bits' in p:
                assert comparison['exact_mismatches'] == 0, 'Fixed-point CMSSW/HLS C++ mismatch'
            else:
                assert comparison['max_abs'] <= cfg['floating_backend_absolute_tolerance']
        predictions[p['name']] = score
        report['profiles'].append(result)
        save(out/'report.json', report)
    report['complete'] = not args.skip_hls
    report['synthesis'] = {'run':False,'tools_on_path':{t:shutil.which(t) for t in ['vitis_hls','vivado_hls','vitis-run','vivado']},
                           'note':'C++ emulation is validated; no FPGA resources or hardware latency are inferred.'}
    report['environment_freeze'] = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    report['source_hashes'] = {p:digest(ROOT/p) for p in ['scripts/validate_xgboost_conifer.py','fpga/conifer/compat.py','cmssw/TreeBenchmark/ConiferValidation/bin/validateXGBoost.cc',args.config]}
    conifer_base = next(line.split('=',1)[1] for line in report['tools']['conifer'].splitlines() if line.startswith('CONIFER_BASE='))
    report['header_hashes'] = {str(p):digest(p) for p in [Path(conifer_base)/'include/conifer.h',Path(os.environ['XILINX_AP_INCLUDE'])/'ap_fixed.h',Path(os.environ['JSON_ROOT'])/'nlohmann/json.hpp']}
    report['hls_stream_header_available'] = (Path(os.environ['XILINX_AP_INCLUDE'])/'hls_stream.h').is_file()
    np.savez_compressed(out/'validation_scores.npz', y=y, **{k.replace(' ','_'):v for k,v in predictions.items()})
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for name, values in predictions.items():
        if name in ('double','float'):continue
        label=name
        if name.startswith('fixed_'):
            r=next(r for r in report['profiles'] if r['name']==name)
            label=f"fixed {r['input_bits']}/{r['score_bits']} bits"
        fpr,tpr,_ = roc_curve(y,values)
        for ax in axes: ax.plot(fpr,tpr,label=label,lw=1.1)
    axes[0].set_xlim(0,1); axes[0].set_ylim(0,1); axes[0].set_title('Validation ROC')
    axes[1].set_xscale('log');axes[1].set_xlim(1e-4,.1); axes[1].set_ylim(0,1);axes[1].set_title('Low background acceptance')
    for ax in axes:ax.set_xlabel('Background efficiency');ax.set_ylabel('Signal efficiency');ax.grid(alpha=.2);ax.axvline(.01,color='gray',ls=':',lw=.7)
    axes[1].legend(fontsize=7,loc='lower right');fig.tight_layout();fig.savefig(out/'roc.pdf');fig.savefig(out/'roc.png',dpi=170);plt.close(fig)
    save(out/'report.json', report)
    print(json.dumps({'complete':report['complete'],'events':len(x),'report':str(out/'report.json')},indent=2),flush=True)


if __name__ == '__main__':
    main()
