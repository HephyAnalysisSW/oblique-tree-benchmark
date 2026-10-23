"""Publish the exact best-efficiency checkpoint and its converged terminal model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,hashlib,pickle
import numpy as np
from datetime import datetime,timezone
from data_utils import resolve,load_config,load_metadata,write_json
from evaluate import compute_metrics,safe_label
from scripts.compare_tao_inference import benchmark
from scripts.summarize_tao_layout_scan import roc_figure
import common.syncer as syncer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='qg_vs_wzt_hardtao_linear_d3_n3_sa1e-05_la1e-05_seed42_hard_search_v1')
    args=parser.parse_args();cfg=load_config('configs/'+args.run+'.yaml')
    ref=load_config('configs/bdt_qg_vs_wzt_seed42.yaml');meta=load_metadata(ref['data']['metadata_path'])
    with np.load(resolve(ref['data']['dataset_path'])) as data:x,y=data['X_val'].astype(float),data['y_val']
    best_path=resolve('models/'+args.run+'_best_efficiency.pkl')
    with best_path.open('rb') as f:best=pickle.load(f)
    with resolve(cfg['model']['output_model_path']).open('rb') as f:terminal=pickle.load(f)
    assert terminal['training_converged']
    with resolve(ref['model']['output_model_path']).open('rb') as f:bdt=pickle.load(f)['model']
    bdt.get_booster().set_param({'nthread':1})
    n=best['model'].ntrees;d=best['model'].depth;mode=best['model'].leaf_mode
    items=[dict(label='XGBoost: 100 trees, depth 4',model=bdt,color='black'),
        dict(label=f'TAO: {n} trees, depth {d}, best efficiency (pass {best["epoch"]})',model=best['model'],color='tab:blue'),
        dict(label=f'TAO: converged terminal (pass {len(terminal["history"])})',model=terminal['model'],color='tab:orange')]
    for item in items:
        item['metrics'],item['roc']=compute_metrics(y,item['model'].predict_proba(x),meta['class_names'])
        item['timings']=[benchmark(item['model'].predict_proba,x[:size]) for size in [1,len(x)]]
        print(item['label'],item['metrics']['signal_efficiency_at_fixed_fpr'],flush=True)
    root=resolve(f'plots/tao_{n}tree_depth{d}_{mode}');root.mkdir(parents=True,exist_ok=True)
    title='Current winning saved checkpoint | validation | signal efficiency at 1% background\nOne CPU thread; bulk timing in legend; selected before terminal convergence'
    for path in [root/'winner_roc.png',resolve('plots/comparisons/current_winner_roc.png')]:
        roc_figure(items,safe_label(meta['class_names'][1]),path,title)
    parameters=sum(t.split_w.size+t.split_b.size+t.leaf_parameters.size for t in best['model'].trees)
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),run_name=args.run,selected_epoch=best['epoch'],
        checkpoint_sha256=hashlib.sha256(best_path.read_bytes()).hexdigest(),parameters=parameters,
        training_run_converged=True,selected_checkpoint_converged=best['training_converged_at_save'],test_evaluated=False,
        models=[{k:v for k,v in item.items() if k not in ['model','color','roc']} for item in items])
    lines=['# Current winning saved oblique checkpoint','',f'{n} depth-{d} trees with {mode} leaves; {parameters} stored scalar parameters.',
        '', 'Validation only, unchanged exported partitions. The winning checkpoint is selected by efficiency at 1% background acceptance. Its training run completed; the selected checkpoint precedes terminal convergence.',
        '', '| Model | Efficiency at 1% background | AUC | Bulk CPU us/event |','|---|---:|---:|---:|']
    for item in items:
        m=item['metrics'];lines.append(f"| {item['label']} | {m['signal_efficiency_at_fixed_fpr']:.2%} | {m['signal_auc']:.6f} | {item['timings'][-1]['median_microseconds_per_event']:.3f} |")
    lines+=['','CPU timings compare current NumPy TAO and compiled XGBoost probability APIs. They do not estimate FPGA latency.',
        '', '![ROC](current_winner_roc.png)']
    report=resolve('plots/comparisons/current_winner_report.md');report.write_text('\n'.join(lines)+'\n');syncer.register_file(report)
    for path in [root/'winner_metrics.json',resolve('plots/comparisons/current_winner_summary.json')]:write_json(path,result);syncer.register_file(path)


if __name__=='__main__':main()
