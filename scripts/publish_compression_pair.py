"""Snapshot and compare the continuing three-tree control and frozen compact tree."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import hashlib,json,pickle
from datetime import datetime,timezone
import numpy as np
from data_utils import resolve,load_config,load_metadata,write_json
from evaluate import compute_metrics,safe_label
from tao.portable import restore_model
from tao.sparsity import sparsity_profile
from scripts.compare_tao_inference import benchmark,bdt_counts
from scripts.summarize_tao_layout_scan import roc_figure
import common.syncer as syncer
import matplotlib.pyplot as plt


def main():
    cfg=load_config('configs/bdt_qg_vs_wzt_seed42.yaml');meta=load_metadata(cfg['data']['metadata_path'])
    audit=json.loads(resolve('results/tao_split_audit.json').read_text())
    dataset=resolve(cfg['data']['dataset_path']);h=hashlib.sha256()
    with dataset.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==audit['dataset_sha256']
    with np.load(dataset) as data:x,y=np.ascontiguousarray(data['X_val'],dtype=float),data['y_val']
    bdt=pickle.loads(resolve(cfg['model']['output_model_path']).read_bytes())['model']
    bdt.get_booster().set_param({'nthread':1});bc=bdt_counts(bdt)
    items=[dict(label='XGBoost: 100 trees, depth 4',model=bdt,color='black',dense_slots=bc['internal_nodes']+bc['leaves']+1,
                nonzeros=None,source=cfg['model']['output_model_path'],run_converged=True)]
    sources=list(resolve('models').glob('*_l1_path_v1_control_best_efficiency.pkl'))
    assert len(sources)==1,sources
    blob=sources[0].read_bytes();digest=hashlib.sha256(blob).hexdigest();payload=pickle.loads(blob)
    assert payload['config']['dataset_sha256']==audit['dataset_sha256']
    snapshot=resolve('models/compression_frontier')/(digest+'.pkl');snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():snapshot.write_bytes(blob)
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest
    status=json.loads(resolve('results/'+payload['config']['run_name']+'_status.json').read_text())
    compact_path=resolve('fpga_handoff/v2/compact_1t_d4/standardized/model.json')
    manifest=json.loads(resolve('fpga_handoff/v2/manifest.json').read_text())
    compact=next(c for c in manifest['candidates'] if c['id']=='compact_1t_d4')
    for label,model,color,source,sha,epoch,converged,expected in [
        ('TAO: 3 trees, depth 4, linear',payload['model'],'tab:blue',str(snapshot.relative_to(resolve('.'))),digest,payload['epoch'],bool(status.get('converged')),payload['validation']),
        ('TAO: 1 tree, depth 4, linear',restore_model(json.loads(compact_path.read_text())),'tab:orange',str(compact_path.relative_to(resolve('.'))),hashlib.sha256(compact_path.read_bytes()).hexdigest(),compact['selected_epoch'],compact['training_run_converged'],compact['metrics'])]:
        model.indexed='dense';sp=sparsity_profile(model)
        items.append(dict(label=label,model=model,color=color,dense_slots=sp['dense_parameter_slots'],nonzeros=sp['nonzero_coefficients'],
            source=source,sha256=sha,selected_epoch=epoch,run_converged=converged,expected_metrics=expected,
            sparse_mac_bound=sp['selected_path_multiplications_upper_bound']))
    for item in items:
        item['metrics'],item['roc']=compute_metrics(y,item['model'].predict_proba(x),meta['class_names'])
        if 'expected_metrics' in item:
            for key in ['signal_efficiency_at_fixed_fpr','signal_auc']:
                assert abs(item['metrics'][key]-item['expected_metrics'][key])<1e-7,(item['label'],key)
        item['timings']=[benchmark(item['model'].predict_proba,x)]
    root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    roc_figure(items,safe_label(meta['class_names'][1]),root/'compression_pair_roc.png',
        'Three-tree and compact one-tree comparison | validation\nFixed saved coefficients; training continues')
    cells=[]
    for item in items:
        m=item['metrics'];cells.append([item['label'].replace('TAO: ',''),str(item['dense_slots']),str(item['nonzeros']) if item['nonzeros'] is not None else '—',
            f"{m['signal_efficiency_at_fixed_fpr']:.2%}",f"{m['signal_auc']:.5f}",f"{item['timings'][-1]['median_microseconds_per_event']:.3f}"])
    columns=['Model','Dense parameters','Nonzero coefficients','Eff. at 1% bkg','AUC','CPU us/event']
    fig,ax=plt.subplots(figsize=(14,3.5));ax.axis('off')
    table=ax.table(cellText=cells,colLabels=columns,cellLoc='center',colWidths=[.32,.14,.16,.16,.10,.12],bbox=[0,.27,1,.55])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d9dee7')
        if row==0:cell.set_facecolor('#23334d');cell.get_text().set_color('white')
        else:cell.set_facecolor('#f1f5fa' if row%2 else 'white')
    fig.suptitle('q/g vs W/Z/top | same validation split | saved checkpoints',fontsize=15)
    ax.text(0,.14,'TAO source runs remain unconverged. Dense counts include biases; nonzero counts are exact.',transform=ax.transAxes,fontsize=9)
    ax.text(0,.04,'CPU: single thread, full validation batch; NumPy TAO vs compiled XGBoost, not FPGA latency.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(root/'compression_pair_table.png',dpi=180);plt.close(fig)
    records=[{k:v for k,v in item.items() if k not in ['model','roc','color','expected_metrics']} for item in items]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),dataset_sha256=audit['dataset_sha256'],validation_events=len(y),test_evaluated=False,models=records)
    write_json(root/'compression_pair_summary.json',result)
    lines=['# Three-tree versus compact one-tree comparison','',result['created_utc'],'',
           '| '+' | '.join(columns)+' |','|'+'|'.join(['---']+['---:']*5)+'|']
    lines+=['| '+' | '.join(row)+' |' for row in cells]
    lines+=['','Validation only; fixed saved checkpoints. TAO source runs remain unconverged. CPU measurements compare NumPy TAO with compiled XGBoost, not FPGA latency.',
            '', '![ROC](compression_pair_roc.png)','','![Table](compression_pair_table.png)']
    (root/'compression_pair_report.md').write_text('\n'.join(lines)+'\n')
    for name in ['compression_pair_summary.json','compression_pair_report.md']:syncer.register_file(root/name)
    print('\n'.join(lines[:9]),flush=True)


if __name__=='__main__':main()
