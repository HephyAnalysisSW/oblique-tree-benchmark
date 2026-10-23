"""Publish only numerically converged terminal models, with shared CPU timings."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,hashlib,json,pickle,platform
import numpy as np
from data_utils import resolve,load_config,load_metadata,write_json
from evaluate import compute_metrics,safe_label
from scripts.compare_tao_inference import benchmark,bdt_counts
from scripts.summarize_tao_layout_scan import roc_figure
import common.syncer as syncer
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/tao_full_training_v1.json')
    parser.add_argument('--include-linear',action='store_true')
    parser.add_argument('--extra-config',action='append',default=[])
    parser.add_argument('--output-prefix')
    parser.add_argument('--focus-layouts',help='Comma-separated trees:depth pairs')
    args=parser.parse_args()
    scan=json.loads(resolve(args.config).read_text())
    prefix=(scan['publication_prefix']+('_vs_baseline' if args.include_linear else '')) if 'publication_prefix' in scan else ('overall_training' if args.include_linear else ('constant_full_training' if scan['leaf_mode']=='constant' else 'full_training'))
    if args.output_prefix:prefix=args.output_prefix
    scans=[scan]
    if args.include_linear:scans.insert(0,json.loads(resolve('configs/tao_full_training_v1.json').read_text()))
    scans.extend(json.loads(resolve(path).read_text()) for path in args.extra_config)
    focus={tuple(map(int,pair.split(':'))) for pair in args.focus_layouts.split(',')} if args.focus_layouts else None
    audit=json.loads(resolve('results/tao_split_audit.json').read_text())
    reference=load_config('configs/bdt_'+scan['task']+'_seed42.yaml')
    metadata=load_metadata(reference['data']['metadata_path'])
    with np.load(resolve(reference['data']['dataset_path'])) as d:
        x,y=np.ascontiguousarray(d['X_val'],dtype=float),d['y_val']
    with resolve(reference['model']['output_model_path']).open('rb') as f:bdt=pickle.load(f)['model']
    bdt.get_booster().set_param({'nthread':1})
    items=[dict(label='XGBoost: 100 trees, depth 4',run_name=reference['run_name'],config=reference,
        model=bdt,color='black',complexity=bdt_counts(bdt),training_seconds=None,history=[])]
    for scan,i,layout in [(policy,i,layout) for policy in scans for i,layout in enumerate(policy['layouts'])]:
        if focus and (layout['trees'],layout['depth']) not in focus:continue
        member=json.loads(resolve(f"results/{scan['scan_name']}_member_{i}.json").read_text())
        assert member['converged'] and member['complete']
        cfg=load_config('configs/'+member['run_name']+'.yaml')
        assert cfg['dataset_sha256']==audit['dataset_sha256'] and cfg['data']==reference['data']
        with resolve(cfg['model']['output_model_path']).open('rb') as f:payload=pickle.load(f)
        assert payload['training_converged'] and payload['convergence']['converged']
        model=payload['model'];model.indexed='dense'
        c=model.complexity();c['stored_coefficients']=sum(t.split_w.size+t.split_b.size+t.leaf_parameters.size for t in model.trees)
        tree_word='tree' if model.ntrees==1 else 'trees'
        variant=('improved ' if scan.get('split_search')=='hard_threshold' else 'baseline ') if args.include_linear and prefix.startswith('hard_search') else ''
        if scan.get('label'):variant=scan['label']+' '
        items.append(dict(label=f'TAO {variant}{model.leaf_mode}: {model.ntrees} {tree_word}, depth {model.depth}',run_name=member['run_name'],
            config=cfg,model=model,complexity=c,color=plt.get_cmap('tab20')(len(items)-1),history=payload['history'],
            convergence=payload['convergence'],training_seconds=member['training_seconds']+member['source_training_seconds']))
    for item in items:
        c=item['complexity']
        if item is items[0]:
            c['split_parameter_count']=c['internal_nodes']
            c['leaf_parameter_count']=c['leaves']
            c['global_bias_parameters']=1
            c['feature_indices']=c['internal_nodes']
        else:
            c['split_parameter_count']=sum(t.split_w.size+t.split_b.size for t in item['model'].trees)
            c['leaf_parameter_count']=sum(t.leaf_parameters.size for t in item['model'].trees)
            c['global_bias_parameters']=0
        c['parameter_count']=c['split_parameter_count']+c['leaf_parameter_count']+c['global_bias_parameters']
        item['metrics'],item['roc']=compute_metrics(y,item['model'].predict_proba(x),metadata['class_names'])
        item['timings']=[]
        item['model_sha256']=hashlib.sha256(resolve(item['config']['model']['output_model_path']).read_bytes()).hexdigest()
    for j,size in enumerate([1,1024,len(x)]):
        for item in (items if j%2==0 else items[::-1]):
            row=benchmark(item['model'].predict_proba,x[:size]);item['timings'].append(row)
            print(item['label'],size,row['median_microseconds_per_event'],flush=True)
    for item in items:item['timings'].sort(key=lambda r:r['batch'])
    root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    signal=safe_label(metadata['class_names'][1])
    title='Converged hard TAO | '+('split proposal comparison' if prefix.startswith('hard_search') else ('linear + constant leaves' if args.include_linear else scan['leaf_mode']+' leaves'))+' | q/g vs W/Z/top | validation\nCPU: one thread, batch 132,800, current probability APIs'
    roc_figure(items,signal,root/(prefix+'_roc.png'),title)
    for item in items[1:]:
        c=item['complexity'];folder=resolve(f"plots/tao_{c['trees']}tree_depth{c['maximum_depth']}_{c['leaf_mode']}")
        roc_figure([items[0],item],signal,folder/('qg_vs_wzt_'+prefix+'_roc.png'),title)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for item in items:
        timing=item['timings'][-1];latency=timing['median_microseconds_per_event']
        samples=np.asarray(timing['wall_seconds'])*1e6/len(x);lower,upper=np.percentile(samples,[25,75])
        label='BDT' if item is items[0] else f"{item['complexity']['leaf_mode'][0].upper()} {item['complexity']['trees']}T d{item['complexity']['maximum_depth']}"
        for ax,key in zip(axes,['signal_auc','signal_efficiency_at_fixed_fpr']):
            value=item['metrics'][key]
            ax.errorbar(latency,value,xerr=np.array([[latency-lower],[upper-latency]]),fmt='o',color=item['color'],capsize=3)
            ax.annotate(label,(latency,value),xytext=(5,5),textcoords='offset points',fontsize=8)
            ax.axhline(items[0]['metrics'][key],color='grey',ls=':',lw=.7)
    for ax,label in zip(axes,['Validation AUC','Signal efficiency at 1% background']):
        ax.set_xlabel('CPU inference [microseconds/event, full batch]');ax.set_ylabel(label);ax.grid(alpha=.2);ax.margins(.2)
    fig.suptitle('Converged terminal checkpoints | timing bars: interquartile range')
    fig.tight_layout();fig.savefig(root/(prefix+'_tradeoffs.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for item in items[1:]:
        times=[(item['config']['source_training_seconds']+r['cumulative_training_seconds'])/60 for r in item['history']]
        for ax,key in zip(axes,['signal_auc','signal_efficiency_at_fixed_fpr']):
            ax.plot(times,[r['validation'][key] for r in item['history']],'o-',color=item['color'],label=item['label'],ms=3)
            ax.axhline(items[0]['metrics'][key],color='grey',ls=':',lw=.7)
    for ax,label in zip(axes,['Validation AUC','Signal efficiency at 1% background']):
        ax.set_xlabel('Total training compute wall time [min, including source training]');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Training to numerical convergence | fixed partitions, L1=1e-5')
    fig.tight_layout();fig.savefig(root/(prefix+'_learning.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for item in items:
        ax.loglog([r['batch'] for r in item['timings']],[r['median_microseconds_per_event'] for r in item['timings']],
                  'o-',color=item['color'],label=item['label'])
    ax.set_xlabel('Batch size');ax.set_ylabel('CPU inference [microseconds/event]');ax.grid(alpha=.2);ax.legend(fontsize=7)
    ax.set_title('Converged models | one CPU thread | warmed probability APIs')
    fig.tight_layout();fig.savefig(root/(prefix+'_cpu.png'),dpi=180);plt.close(fig)
    ranking=sorted(items,key=lambda item:(item['metrics']['signal_efficiency_at_fixed_fpr'],item['metrics']['signal_auc']),reverse=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for item in items:
        eff=item['metrics']['signal_efficiency_at_fixed_fpr']
        parameters=item['complexity']['parameter_count'];cpu=item['timings'][-1]['median_microseconds_per_event']
        for ax,value in zip(axes,[parameters,cpu]):
            ax.scatter(value,100*eff,color=item['color'],label=item['label'],s=35)
            ax.axhline(100*items[0]['metrics']['signal_efficiency_at_fixed_fpr'],color='grey',ls=':',lw=.7)
    axes[0].set_xlabel('Stored scalar parameters');axes[0].set_xscale('log')
    axes[1].set_xlabel('CPU inference [microseconds/event, full batch]')
    for ax in axes:ax.set_ylabel('Signal efficiency at 1% background [%]');ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=7)
    fig.suptitle('Primary benchmark: signal efficiency at 1% background acceptance | validation')
    fig.set_figheight(8 if len(items)>8 else 6.5)
    fig.tight_layout(rect=(0,.35 if len(items)>8 else .25,1,.95))
    fig.savefig(root/(prefix+'_working_point.png'),dpi=180);plt.close(fig)
    result=dict(primary_metric='signal_efficiency_at_fixed_fpr',fixed_fpr=.01,secondary_metric='signal_auc',
        ranked_models=[item['run_name'] for item in ranking],scan=scan,split_audit=audit,models=[{k:v for k,v in item.items() if k not in ['model','roc','color','history']} for item in items],
        completed_layouts=len(items)-1,scans=scans,all_training_converged=True,partition='validation',test_evaluated=False,
        host=platform.node(),cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),''),
        caveats=['Fixed-routing leaf optimality and stable hard updates do not certify a local or global tree optimum.',
            'Terminal converged models are compared. Best-efficiency and best-AUC checkpoints are separate archives when saved; ranking does not change the numerical stopping rule.',
            'Training seeds: '+', '.join(map(str,sorted({item['config']['seed'] for item in items[1:]})))+'. CPU timings compare the current implementations, not FPGA latency.'])
    write_json(root/(prefix+'_summary.json'),result)
    lines=['# Fully trained TAO comparison','',
        'All terminal checkpoints pass the recorded convergence criteria. Exact exported partitions match XGBoost; test data remain unused. Models are ranked by signal efficiency at 1% background acceptance, with AUC as a secondary metric.','',
        '| Model | Parameters | Nonzero weights | Split coefficients / thresholds | Leaf coefficients | Efficiency at 1% FPR | AUC | CPU us/event, batch 1 | CPU us/event, full batch | Total training min |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in ranking:
        m=item['metrics'];c=item['complexity'];training='—' if item['training_seconds'] is None else f"{item['training_seconds']/60:.1f}"
        nonzeros='—' if item is items[0] else str(c['split_nonzeros']+c['leaf_nonzeros'])
        lines.append(f"| {item['label']} | {c['parameter_count']} | {nonzeros} | {c['split_parameter_count']} | {c['leaf_parameter_count']} | {m['signal_efficiency_at_fixed_fpr']:.6f} | {m['signal_auc']:.6f} | {item['timings'][0]['median_microseconds_per_event']:.3f} | {item['timings'][-1]['median_microseconds_per_event']:.3f} | {training} |")
    lines+=['', 'Parameter counts are stored scalar model coefficients, including biases and zeros. TAO splits include all 16 weights plus bias; each binary leaf has 17 coefficients (linear) or one (constant). XGBoost counts one threshold per split, one score per leaf and one global base score; integer feature IDs and topology are excluded. Shared input preprocessing is excluded. These are storage counts, not identifiable degrees of freedom. Nonzero weights count TAO feature weights in splits and leaves, excluding biases; they are reported separately to assess L1 sparsity.', '', '## Convergence and interpretation','']+['- '+s for s in result['caveats']]
    lines+=['- Stop after at least five additional full passes and three consecutive passes with relative training-objective gain <=1e-5, changed training routes <=1e-4, all local leaf/split fits converged, and leaf L1 KKT violation <=1e-6.',
        '- Inner leaf solver uses ftol=1e-12, gtol=1e-8. Iteration limits or scheduler time limits alone never mark a model converged.',
        '', 'Figures use the same prefix as this report; ROC legends include CPU timings.']
    (root/(prefix+'_report.md')).write_text('\n'.join(lines)+'\n')
    for name in [prefix+'_report.md',prefix+'_summary.json']:syncer.register_file(root/name)
    print('All converged comparisons published.',flush=True)


if __name__=='__main__':main()
