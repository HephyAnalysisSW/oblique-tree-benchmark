"""Compare selected layouts on validation data and time them on one worker."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
import hashlib
import json
import os
import pickle
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_utils import resolve, load_config, load_metadata, write_json
from evaluate import compute_metrics, safe_label
from scripts.compare_tao_inference import benchmark, bdt_counts
import common.syncer as syncer


def roc_figure(items, signal, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 10 if len(items)>8 else 7), sharey=True)
    for item in items:
        roc, metrics = item['roc'], item['metrics']
        bulk = item['timings'][-1]['median_microseconds_per_event']
        label = (f"{item['label']} | AUC {metrics['signal_auc']:.5f}\n"
                 f"eff@1% {metrics['signal_efficiency_at_fixed_fpr']:.2%} | {bulk:.3f} us/event")
        for ax in axes:
            line, = ax.plot(roc[signal+'_fpr'], roc[signal+'_tpr'],
                           color=item['color'], lw=1.7, label=label)
            ax.plot(.01, metrics['signal_efficiency_at_fixed_fpr'], 'o', color=line.get_color(), ms=4)
    for ax in axes:
        ax.set_xlabel('Background acceptance'); ax.set_ylim(0, 1.02); ax.grid(alpha=.2)
        ax.axvline(.01, ls=':', color='.5', lw=.7)
    axes[0].set_xlim(0, 1); axes[0].set_ylabel('Signal efficiency')
    axes[1].set_xscale('log'); axes[1].set_xlim(1e-4, 1)
    axes[0].set_title('Full ROC'); axes[1].set_title('Low background acceptance')
    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(.5, .015))
    bottom = .39 if len(items)>8 else (.12 if len(items) <= 2 else .23)
    fig.tight_layout(rect=(0, bottom, 1, .95))
    fig.savefig(path, dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    scan = json.loads(resolve('configs/tao_layout_scan_v1.json').read_text())
    members = []
    for i in range(len(scan['layouts'])):
        marker = resolve(f"results/{scan['scan_name']}_member_{i}.json")
        if marker.exists():
            members.append(json.loads(marker.read_text()))
    if not members or (not args.allow_partial and len(members) != len(scan['layouts'])):
        raise RuntimeError(f'Completed {len(members)}/{len(scan["layouts"])} layouts')
    bcfg = load_config(f"configs/bdt_{scan['task']}_seed42.yaml")
    metadata = load_metadata(bcfg['data']['metadata_path'])
    with np.load(resolve(bcfg['data']['dataset_path'])) as d:
        x, y = np.ascontiguousarray(d['X_val'], dtype=np.float64), d['y_val']
    with resolve(bcfg['model']['output_model_path']).open('rb') as f:
        bdt = pickle.load(f)['model']
    bdt.get_booster().set_param({'nthread': 1})
    bc = bdt_counts(bdt)
    items = [dict(label=f"XGBoost: {bc['trees']} trees, depth {bc['maximum_depth']}",
                  run_name=bcfg['run_name'], config=bcfg, model=bdt, color='black',
                  complexity=bc, training_wall_seconds=None)]
    for member in members:
        config = load_config('configs/'+member['selected_run']+'.yaml')
        if config['data']['dataset_path'] != bcfg['data']['dataset_path']:
            raise ValueError('Dataset mismatch')
        with resolve(config['model']['output_model_path']).open('rb') as f:
            model = pickle.load(f)['model']
        model.indexed = 'dense'
        alternating = json.loads(resolve('results/'+member['base_run']+'_history.json').read_text())
        refinement = json.loads(resolve('results/'+member['selected_run']+'_history.json').read_text())
        validation = json.loads(resolve('results/'+member['selected_run']+'_validation.json').read_text())
        base_validation = json.loads(resolve('results/'+member['base_run']+'_validation.json').read_text())
        complexity = model.complexity()
        complexity['stored_coefficients'] = sum(t.split_w.size+t.split_b.size+t.leaf_parameters.size for t in model.trees)
        complexity['selected_path_multiplications'] = model.ntrees*(model.depth*x.shape[1]+x.shape[1]*(model.n_classes_-1))
        tree_word = 'tree' if model.ntrees == 1 else 'trees'
        items.append(dict(label=f"TAO: {model.ntrees} {tree_word}, depth {model.depth}",
            run_name=member['selected_run'], config=config, model=model, complexity=complexity,
            color=plt.get_cmap('tab10')(member['index']),
            training_wall_seconds=sum(h['profile']['seconds'] for h in alternating)+sum(h['seconds'] for h in refinement),
            alternating_epochs=len(alternating), best_alternating_epoch=base_validation['best_epoch'],
            selected_refinement_stage=validation['best_stage'],
            final_leaf_refit_converged=bool(refinement[-1].get('refit',{}).get('converged',False)),
            alternating_leaf_fits_at_limit=sum(not s['converged'] for h in alternating for s in h['profile']['leaf_fits']),
            alternating_global_refits_at_limit=sum(not h['profile']['global_refit']['converged'] for h in alternating),
            alternating_history=alternating, refinement_history=refinement))
    for item in items:
        item['metrics'], item['roc'] = compute_metrics(y,item['model'].predict_proba(x),metadata['class_names'])
        if item is not items[0]:
            validation = json.loads(resolve('results/'+item['run_name']+'_validation.json').read_text())
            assert abs(item['metrics']['signal_auc']-validation['validation']['signal_auc'])<1e-8
        item['timings'] = []
    # Same worker, one thread, identical resident inputs and timing procedure.
    for batch_index, size in enumerate([1,1024,len(x)]):
        for item in (items if batch_index % 2 == 0 else items[::-1]):
            timing = benchmark(item['model'].predict_proba,x[:size])
            item['timings'].append(timing)
            print(json.dumps(dict(model=item['label'],batch=size,
                us_per_event=timing['median_microseconds_per_event'])),flush=True)
    for item in items:
        item['timings'].sort(key=lambda t:t['batch'])
        item['model_sha256'] = hashlib.sha256(resolve(item['config']['model']['output_model_path']).read_bytes()).hexdigest()
    root = resolve('plots/comparisons')
    root.mkdir(parents=True,exist_ok=True)
    signal = safe_label(metadata['class_names'][1])
    cpu = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'')
    title = f"q/g vs W/Z/top | validation | {len(members)}/{len(scan['layouts'])} layouts complete"
    subtitle = f"CPU timing: 1 thread, batch {len(x):,}, resident inputs; probability APIs"
    roc_figure(items,signal,root/'layout_scan_roc.png', title+'\n'+subtitle)
    for item in items[1:]:
        c=item['complexity']; folder=resolve(f"plots/tao_{c['trees']}tree_depth{c['maximum_depth']}_{c['leaf_mode']}")
        roc_figure([items[0],item],signal,folder/f"{scan['task']}_layout_scan_roc.png",title+'\n'+subtitle)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for item in items:
        latency=item['timings'][-1]['median_microseconds_per_event']
        samples=np.asarray(item['timings'][-1]['wall_seconds'])*1e6/len(x)
        lower,upper=np.percentile(samples,[25,75])
        for ax,key in zip(axes,['signal_auc','signal_efficiency_at_fixed_fpr']):
            ax.scatter(latency,item['metrics'][key],color=item['color'],s=65,marker='*' if item is items[0] else 'o')
            ax.errorbar(latency,item['metrics'][key],xerr=np.array([[latency-lower],[upper-latency]]),
                        fmt='none',color=item['color'],capsize=3,alpha=.7)
            label='BDT' if item is items[0] else f"{item['complexity']['trees']}T d{item['complexity']['maximum_depth']}"
            ax.annotate(label,(latency,item['metrics'][key]),xytext=(5,5),textcoords='offset points',fontsize=8)
            ax.axhline(items[0]['metrics'][key],color='grey',ls=':',lw=.8)
    for ax,ylabel in zip(axes,['Validation AUC','Signal efficiency at 1% background acceptance']):
        ax.set_xlabel('CPU inference [microseconds/event, full validation batch]');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        ax.margins(x=.2,y=.2)
    fig.suptitle(title+'\nOne CPU thread; linear leaves, fixed L1; timing bars: interquartile range')
    fig.tight_layout();fig.savefig(root/'layout_scan_tradeoffs.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for item in items:
        ax.loglog([r['batch'] for r in item['timings']],
                  [r['median_microseconds_per_event'] for r in item['timings']],
                  'o-',color=item['color'],label=item['label'])
    ax.set_xlabel('Batch size');ax.set_ylabel('CPU inference [microseconds/event]')
    ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_title(title+'\n'+cpu+' | one thread')
    fig.tight_layout();fig.savefig(root/'layout_scan_cpu.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for item in items[1:]:
        history=item['alternating_history']; epochs=[r['epoch'] for r in history]
        for ax,key in zip(axes[:2],['signal_auc','signal_efficiency_at_fixed_fpr']):
            ax.plot(epochs,[h['validation'][key] for h in history],'o-',color=item['color'],label=item['label'])
            ax.plot([epochs[-1],5],[history[-1]['validation'][key],item['metrics'][key]],
                    ':',color=item['color'],alpha=.5)
            ax.scatter(5,item['metrics'][key],marker='*',s=80,color=item['color'])
        seconds=np.cumsum([h['profile']['seconds'] for h in history])
        final_seconds=seconds[-1]+sum(h['seconds'] for h in item['refinement_history'])
        axes[2].plot(epochs+[5],np.r_[seconds,final_seconds]/60,'o-',color=item['color'],label=item['label'])
    for ax,key in zip(axes[:2],['signal_auc','signal_efficiency_at_fixed_fpr']):
        ax.axhline(items[0]['metrics'][key],color='black',ls=':',label='Reference BDT')
    for ax,ylabel in zip(axes,['Validation AUC','Signal efficiency at 1% background','Cumulative training wall time [min]']):
        ax.set_xlabel('Alternating epoch / final selection and leaf refit');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        ax.set_xticks([1,2,3,4,5],['1','2','3','4','Refit'])
    axes[2].plot([],[],color='black',ls=':',label='Reference BDT (left panels)')
    axes[2].legend(fontsize=7)
    fig.suptitle('Matched training budget | Refit: best-AUC checkpoint + fixed-routing leaf fit')
    fig.tight_layout();fig.savefig(root/'layout_scan_training.png',dpi=180);plt.close(fig)
    serial=[]
    for item in items:
        serial.append({k:v for k,v in item.items() if k not in ['model','roc','color','alternating_history','refinement_history']})
    report=dict(scan=scan,completed_layouts=len(members),planned_layouts=len(scan['layouts']),
        partition='validation',events=len(x),test_evaluated=False,models=serial,
        environment=dict(host=platform.node(),cpu=cpu,numpy=np.__version__,
            threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),
        procedure='Four alternating epochs, choose best validation-AUC checkpoint; then up to 1000 additional joint leaf-refit iterations at fixed routing, retaining best validation AUC.',
        caveats=['One seed and finite training budget; no claim of full alternating convergence.',
            'Current Python/NumPy TAO and compiled XGBoost probability APIs; model loading and IO excluded.',
            'CPU timings do not establish FPGA latency. Inference uses dense coefficients, including zeros.',
            'Training time is measured compute wall time, including the final refit; validation, IO and upload are excluded.'])
    write_json(root/'layout_scan_summary.json',report)
    lines=['# Depth and two-tree scan','',report['procedure'],'',
        f"Completed {len(members)}/{len(scan['layouts'])} layouts. Validation only; test data unused.",
        '',f'CPU: {cpu}; one thread. Median of nine warmed timing repeats.',
        '', '| Model | AUC | Efficiency at 1% FPR | CPU us/event (batch 1) | CPU us/event (full batch) | Training min | Leaf refit converged |',
        '|---|---:|---:|---:|---:|---:|---|']
    for item in items:
        m=item['metrics']; training='—' if item['training_wall_seconds'] is None else f"{item['training_wall_seconds']/60:.1f}"
        lines.append(f"| {item['label']} | {m['signal_auc']:.6f} | {m['signal_efficiency_at_fixed_fpr']:.6f} | {item['timings'][0]['median_microseconds_per_event']:.3f} | {item['timings'][-1]['median_microseconds_per_event']:.3f} | {training} | {item.get('final_leaf_refit_converged','—')} |")
    lines+=['','## Caveats','']+['- '+s for s in report['caveats']]
    matched={}
    for item in items[1:]:
        c=item['complexity']; matched[c['trees'],c['maximum_depth']]=c
    if (1,4) in matched and (2,3) in matched:
        lines+=['','## Similar-storage comparison','',
            f"The depth-4 single tree stores {matched[1,4]['stored_coefficients']} scalar coefficients, versus {matched[2,3]['stored_coefficients']} for two depth-3 trees, including split and leaf biases. Both retain linear leaves. Their overlapping partitions differ; each intersection still has an affine logit. Storage similarity does not imply equal computation or latency.",
            f"Selected-path dense multiplication counts are {matched[1,4]['selected_path_multiplications']} and {matched[2,3]['selected_path_multiplications']} per event, respectively; forest trees may be evaluated in parallel in hardware."]
    lines+=['','Split and leaf L1 penalties both remain 1e-5; all TAO leaves are linear. Final refit convergence refers only to leaf coefficients with hyperplanes fixed.',
        '', 'Selection uses AUC; efficiency at 1% background acceptance is reported independently.',
        '', 'Figures: layout_scan_roc.png, layout_scan_tradeoffs.png, layout_scan_cpu.png, layout_scan_training.png.',
        'Raw timing repeats, model hashes, configurations and convergence flags: layout_scan_summary.json.']
    (root/'layout_scan_report.md').write_text('\n'.join(lines)+'\n')
    for name in ['layout_scan_report.md','layout_scan_summary.json']:
        syncer.register_file(root/name)
    print('Summary complete:',len(members),'layouts',flush=True)


if __name__=='__main__':main()
