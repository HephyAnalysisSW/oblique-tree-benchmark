"""Validation ROC, current CPU inference, and explicit hardware work estimates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import time
import numpy as np
from data_utils import load_config, load_metadata, resolve, write_json
from evaluate import compute_metrics, safe_label
import common.syncer as syncer
from common.plot_layout import publication_path
import matplotlib.pyplot as plt


def benchmark(fn, x):
    """Repeated API calls; inputs already resident, no IO or model loading."""
    for _ in range(3):
        fn(x)
    start = time.perf_counter()
    fn(x)
    estimate = max(time.perf_counter() - start, 1e-6)
    calls = max(1, min(128, math.ceil(.025 / estimate)))
    wall, cpu = [], []
    for _ in range(9):
        tic, cpu_tic = time.perf_counter(), time.process_time()
        for _ in range(calls):
            fn(x)
        cpu.append((time.process_time() - cpu_tic) / calls)
        wall.append((time.perf_counter() - tic) / calls)
    return dict(batch=len(x), calls_per_repeat=calls, repeats=9,
                wall_seconds=wall, cpu_seconds=cpu,
                median_wall_seconds=float(np.median(wall)),
                median_cpu_seconds=float(np.median(cpu)),
                median_microseconds_per_event=float(np.median(wall) * 1e6 / len(x)))


def bdt_counts(model):
    trees = [json.loads(t) for t in model.get_booster().get_dump(dump_format='json')]
    def walk(node, depth=0):
        if 'leaf' in node:
            return 0, 1, depth
        children = [walk(c, depth + 1) for c in node['children']]
        return 1 + sum(c[0] for c in children), sum(c[1] for c in children), max(c[2] for c in children)
    counts = [walk(t) for t in trees]
    return dict(trees=len(trees), internal_nodes=sum(c[0] for c in counts),
                leaves=sum(c[1] for c in counts), maximum_depth=max(c[2] for c in counts),
                path_comparisons_upper_bound=sum(c[2] for c in counts),
                tree_score_additions=max(0, len(trees) - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tao-config', required=True)
    parser.add_argument('--roc-only', action='store_true', help='Refresh the validation ROC without repeating timing measurements.')
    args = parser.parse_args()
    cfg = load_config(args.tao_config)
    bcfg = load_config('configs/bdt_qg_vs_wzt_seed42.yaml')
    assert cfg['data']['dataset_path'] == bcfg['data']['dataset_path']
    metadata = load_metadata(cfg['data']['metadata_path'])
    with np.load(resolve(cfg['data']['dataset_path'])) as d:
        x, y = np.ascontiguousarray(d['X_val'], dtype=np.float64), d['y_val']
    with resolve(cfg['model']['output_model_path']).open('rb') as f:
        tao = pickle.load(f)['model']
    with resolve(bcfg['model']['output_model_path']).open('rb') as f:
        bdt = pickle.load(f)['model']
    # Change the booster thread count directly. The sklearn set_params wrapper
    # also re-sends the saved list of metrics, which this XGBoost build rejects.
    bdt.get_booster().set_param({'nthread': 1})
    assert tao.n_classes_ == 2
    prefix = resolve('results/' + cfg['run_name'] + '_vs_xgboost')
    metrics, curves = [], []
    for model in [bdt, tao]:
        m, roc = compute_metrics(y, model.predict_proba(x), metadata['class_names'])
        metrics.append(m); curves.append(roc)
    signal = safe_label(metadata['class_names'][1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    labels = ['XGBoost: 100 trees, depth 4', f'Hard TAO: {tao.ntrees} tree, depth {tao.depth}, linear leaves']
    for label, m, roc in zip(labels, metrics, curves):
        for ax in axes:
            line, = ax.plot(roc[signal+'_fpr'], roc[signal+'_tpr'], lw=1.7,
                           label=f"{label}\nAUC {m['signal_auc']:.5f}")
            ax.plot(.01, m['signal_efficiency_at_fixed_fpr'], 'o', color=line.get_color(), ms=4)
    for ax in axes:
        ax.set_xlabel('Background acceptance'); ax.grid(alpha=.2); ax.set_ylim(0, 1.02)
        ax.axvline(.01, color='grey', ls=':', lw=.8)
    axes[0].set_xlim(0, 1); axes[0].set_ylabel('Signal efficiency'); axes[0].legend(fontsize=8)
    axes[1].set_xscale('log'); axes[1].set_xlim(1e-4, 1)
    axes[1].text(.04,.95, 'At 1% background acceptance:\n' + '\n'.join(
        f"{name}: {m['signal_efficiency_at_fixed_fpr']:.2%}"
        for name,m in zip(['XGBoost','TAO'],metrics)), transform=axes[1].transAxes, fontsize=9,
        va='top', bbox=dict(facecolor='white', edgecolor='0.85', alpha=.95))
    fig.suptitle('q/g vs W/Z/top | validation partition (132,800 events)')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(str(prefix)+'_roc.'+ext, dpi=180)
    plt.close(fig)
    if args.roc_only:
        print('Validation ROC refreshed; timing artifacts unchanged.', flush=True)
        return
    functions = [('xgboost_predict_proba', bdt.predict_proba)]
    reference = tao.predict_proba(x)
    agreement = {}
    for mode in ['dense','indexed','batched']:
        candidate = copy.deepcopy(tao); candidate.indexed = mode
        prediction = candidate.predict_proba(x)
        agreement[mode] = float(np.max(np.abs(reference-prediction)))
        assert np.allclose(reference, prediction, atol=1e-6, rtol=1e-6)
        functions.append(('tao_'+mode+'_predict_proba', candidate.predict_proba))
    timings = {}
    # Alternate model order between batch sizes to reduce a fixed order bias.
    for index, size in enumerate([1,32,1024,16384,len(x)]):
        for name, fn in (functions if index % 2 == 0 else functions[::-1]):
            row = benchmark(fn,x[:size]); timings.setdefault(name,[]).append(row)
            print(json.dumps(dict(model=name, **{k:v for k,v in row.items() if not k.endswith('_seconds')})), flush=True)
    for rows in timings.values(): rows.sort(key=lambda r:r['batch'])
    tc, bc = tao.complexity(), bdt_counts(bdt)
    features=x.shape[1]; free_logits=tao.n_classes_-1
    stored_coefficients=sum(t.split_w.size+t.split_b.size+t.leaf_parameters.size for t in tao.trees)
    path_multiplications=tao.ntrees*(tao.depth*features+features*free_logits)
    allnode_multiplications=sum(t.internal*features+t.leaves*features*free_logits for t in tao.trees)
    work=dict(tao=tc, xgboost=bc, features=features,
        tao_stored_coefficients=stored_coefficients,
        tao_selected_path_multiplications=path_multiplications,
        tao_selected_path_additions_including_bias=path_multiplications,
        tao_selected_path_comparisons=tao.ntrees*tao.depth,
        tao_fully_unrolled_all_nodes_multiplications=allnode_multiplications,
        tao_fully_unrolled_all_nodes_additions_including_bias=allnode_multiplications,
        tao_fully_unrolled_all_nodes_comparisons=tc['internal_nodes'],
        bdt_fully_unrolled_comparisons=bc['internal_nodes'])
    memory=[]
    for bits in [8,12,16,32]:
        # Logical payload only: omit topology, alignment, accumulators and routing.
        memory.append(dict(scalar_bits=bits, tao_payload_bits=stored_coefficients*bits,
            bdt_payload_bits=(bc['internal_nodes']+bc['leaves'])*bits + bc['internal_nodes']*math.ceil(math.log2(features))))
    report=dict(partition='validation',events=len(x),test_evaluated=False,
        config=cfg, reference_config=bcfg, metrics=dict(xgboost=metrics[0],tao=metrics[1]),
        timings=timings,routing_max_probability_difference=agreement,work=work,payload_estimates=memory,
        environment=dict(host=platform.node(),cpu=platform.processor(),platform=platform.platform(),
            numpy=np.__version__,threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
            cpu_info=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'')),
        model_sha256={name:hashlib.sha256(resolve(c['model']['output_model_path']).read_bytes()).hexdigest()
                      for name,c in [('tao',cfg),('xgboost',bcfg)]},
        limitations=['Python/NumPy TAO versus compiled XGBoost: timings measure current APIs, not intrinsic algorithms.',
            'Inputs are resident float64 arrays; conversions and probability transforms inside predict_proba are included.',
            'No preprocessing, model loading, transfers, training or plot/upload time is included.',
            'FPGA payload assumes equal coefficient/threshold/leaf bit widths; quantized accuracy has not been tested.',
            'Logical operation counts are not DSP/LUT/BRAM, clocks, latency or initiation interval.',
            'Comparator, multiplier and adder costs differ; operation counts cannot be divided into an FPGA speedup.',
            'Payload omits topology, control, alignment, activation storage and widened accumulators.'])
    write_json(str(prefix)+'_inference.json',report)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,rows in timings.items():
        ax.loglog([r['batch'] for r in rows],[r['median_microseconds_per_event'] for r in rows], 'o-', label=name.replace('_predict_proba',''))
    ax.set_xlabel('Batch size'); ax.set_ylabel('Median wall time per event [microseconds]')
    ax.set_title('Current probability APIs | one CPU thread | resident inputs')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(str(prefix)+'_cpu.png',dpi=180);plt.close(fig)
    directory=publication_path(Path(str(prefix)+'_cpu.png'),resolve('.')).parent
    # Keep the reviewable estimates beside the ROC and timing figures.
    summary=['# One hard TAO tree versus XGBoost','',
        'Validation data only; these classifiers have different performance. The TAO checkpoint follows two alternating epochs and a converged leaf refit with its hyperplanes fixed; convergence of the full alternating training is not established.','',
        '| Model | Validation AUC | Signal efficiency at 1% FPR |','|---|---:|---:|']
    for name,m in zip(['XGBoost','TAO'],metrics):
        summary.append(f"| {name} | {m['signal_auc']:.6f} | {m['signal_efficiency_at_fixed_fpr']:.6f} |")
    summary+=['','## Current CPU inference','',f"CPU: {report['environment']['cpu_info']}. One thread; 9 repeats, three warmup calls.",
        'Median wall microseconds per event; model loading and IO excluded.','',
        '| Batch | XGBoost | TAO dense | TAO indexed | TAO batched |','|---:|---:|---:|---:|---:|']
    for i,size in enumerate([1,32,1024,16384,len(x)]):
        summary.append('| '+str(size)+' | '+' | '.join(f"{timings[name][i]['median_microseconds_per_event']:.3f}" for name,_ in functions)+' |')
    summary+=['','## Hardware work and payload estimates','',
        f"Actual XGBoost: {bc['trees']} trees, {bc['internal_nodes']} internal nodes, {bc['leaves']} leaves, maximum depth {bc['maximum_depth']}.",
        f"TAO: {tc['trees']} tree, {tc['internal_nodes']} internal nodes, {tc['leaves']} leaves, {stored_coefficients} stored scalar coefficients.",
        f"Selected-path TAO: {path_multiplications} multiplications, {path_multiplications} additions including biases, {tao.ntrees*tao.depth} comparisons per event. Dense counts retain zero weights.",
        f"Selected-path XGBoost: at most {bc['path_comparisons_upper_bound']} scalar comparisons, {bc['tree_score_additions']} tree-score additions per event.",
        f"Fully unrolled TAO (all splits and all leaves evaluated): {allnode_multiplications} multiplications and additions, {tc['internal_nodes']} comparisons, then selection. Fully unrolled XGBoost: {bc['internal_nodes']} comparisons, leaf selection and tree summation.",
        'These are alternative architectures. For binary classification a final score threshold can replace the probability transform in either model.',
        'For 16 features, a parallel dot product has four balanced reduction-adder levels, plus bias handling. A path-based design has three split dot products followed by one selected linear-leaf dot product; computing every node in parallel trades more arithmetic for selection logic. Neither description determines clock cycles without scheduling.',
        'As an accumulator-sizing example, signed 16-bit inputs and weights give 32-bit products. Summing 16 products needs up to 36 bits before additional bias/forest headroom, assuming no saturation. Quantization ranges and binary points still need calibration.',
        '', '| Assumed scalar bits | TAO logical payload bits | XGBoost logical payload bits |','|---:|---:|---:|']
    for row in memory:summary.append(f"| {row['scalar_bits']} | {row['tao_payload_bits']} | {row['bdt_payload_bits']} |")
    summary+=['','## Interpretation and limitations','']+['- '+s for s in report['limitations']]
    summary+=['','No FPGA speedup is claimed. A target device, fixed-point ranges, accumulator widths, clock constraint and synthesis are needed for a resource/latency comparison.',
              '', 'The raw repeated timings, model hashes and count definitions are in inference.json.']
    for name,content in [('report.md','\n'.join(summary)+'\n'),('inference.json',json.dumps(report,indent=2)+'\n')]:
        path=resolve(directory/name);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content);syncer.register_file(path)
    print('Published under',directory,flush=True)


if __name__=='__main__':main()
