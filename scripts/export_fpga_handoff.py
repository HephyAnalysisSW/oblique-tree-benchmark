"""Freeze selected oblique checkpoints, portable coefficients, vectors and comparison."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,hashlib,json,pickle,subprocess
from datetime import datetime,timezone
import numpy as np
import matplotlib.pyplot as plt
from data_utils import resolve,load_config,load_metadata,write_json
from evaluate import compute_metrics,safe_label
from tao.portable import export_model,predict_margin,deterministic_working_point,operation_counts,cpp_header
from scripts.compare_tao_inference import benchmark,bdt_counts
from scripts.summarize_tao_layout_scan import roc_figure
import common.syncer as syncer

CANDIDATES=[('winner_3t_d3',3,3),('compact_3t_d2',3,2),('runnerup_2t_d3',2,3),('single_1t_d5',1,5),('compact_single_1t_d4',1,4)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def probability(margin):
    e=np.exp(-np.abs(margin));return np.where(margin>=0,1/(1+e),e/(1+e))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='fpga_handoff/v1');parser.add_argument('--selection',help='JSON list of explicit run snapshots');args=parser.parse_args()
    output=resolve(args.output)
    if (output/'manifest.json').exists():raise FileExistsError('Handoff version is frozen; choose a new output version')
    output.mkdir(parents=True,exist_ok=True)
    ref=load_config('configs/bdt_qg_vs_wzt_seed42.yaml');meta=load_metadata(ref['data']['metadata_path'])
    audit=json.loads(resolve('results/tao_split_audit.json').read_text())
    assert sha(resolve(ref['data']['dataset_path']))==audit['dataset_sha256']
    with np.load(resolve(ref['data']['dataset_path'])) as data:
        x,y=data['X_val'].astype(float),data['y_val'];mean=data['scaler_mean'];scale=data['scaler_scale']
        training_min=data['X_train'].min(axis=0).astype(float);training_max=data['X_train'].max(axis=0).astype(float)
    write_json(output/'input_contract.json',dict(features=meta['feature_names'],classes=meta['class_names'],mean=mean.tolist(),scale=scale.tolist(),
        training_standardized_min=training_min.tolist(),training_standardized_max=training_max.tolist(),
        training_raw_min=(training_min*scale+mean).tolist(),training_raw_max=(training_max*scale+mean).tolist(),
        clipping_applied=False,nan_supported=False,range_note='Observed training ranges, not physical guarantees. Choose hardware formats/overflow policy explicitly.'))
    with resolve(ref['model']['output_model_path']).open('rb') as f:bdt=pickle.load(f)['model']
    bdt.get_booster().set_param({'nthread':1});bc=bdt_counts(bdt)
    bdt_dir=output/'xgboost_reference';bdt_dir.mkdir(exist_ok=True)
    bdt.get_booster().save_model(str(bdt_dir/'model.json'))
    bm,br=compute_metrics(y,bdt.predict_proba(x),meta['class_names'])
    items=[dict(label='XGBoost: 100 trees, depth 4',model=bdt,color='black',metrics=bm,roc=br,
        parameters=bc['internal_nodes']+bc['leaves']+1,id='xgboost_reference')]
    selected=json.loads(resolve(args.selection).read_text())['candidates'] if args.selection else [dict(id=tag,run_name=f'qg_vs_wzt_hardtao_linear_d{depth}_n{trees}_sa1e-05_la1e-05_seed42_hard_search_v1') for tag,trees,depth in CANDIDATES]
    frozen=[];rng=np.random.default_rng(42)
    for index,selection in enumerate(selected):
        tag,run=selection['id'],selection['run_name']
        source=resolve('models/'+run+'_best_efficiency.pkl');payload_bytes=source.read_bytes();payload=pickle.loads(payload_bytes)
        model=payload['model'];model.indexed='dense';trees,depth=model.ntrees,model.depth
        status=json.loads(resolve('results/'+run+'_status.json').read_text())
        cfg=load_config('configs/'+run+'.yaml');assert cfg['dataset_sha256']==audit['dataset_sha256']
        specs={space:export_model(model,meta['feature_names'],meta['class_names'],mean,scale,raw=space=='raw') for space in ['standardized','raw']}
        margin,routes,tree_outputs=predict_margin(specs['standardized'],x)
        native=model.decision_function(x)[:,0]
        assert np.max(np.abs(margin+native))<1e-9
        assert np.max(np.abs(probability(margin)-model.predict_proba(x)[:,1]))<1e-6
        working=deterministic_working_point(margin,y)
        raw_inputs=x*scale+mean;raw_margin,raw_routes,_=predict_margin(specs['raw'],raw_inputs)
        assert np.array_equal(routes,raw_routes) and np.max(np.abs(raw_margin-margin))<1e-8
        indices=set(rng.choice(len(x),128,replace=False).tolist())
        indices.update(np.argsort(np.abs(margin-working['signal_margin_threshold']))[:16].tolist())
        for tree in model.trees:
            reaching=tree.routes(x,'dense')
            for node,rows in enumerate(reaching[:tree.internal]):
                if len(rows):
                    distances=np.abs(x[rows]@tree.split_w[node]+tree.split_b[node])
                    indices.update(rows[np.argsort(distances)[:4]].tolist())
        indices=np.asarray(sorted(indices),dtype=int)
        candidate=output/tag;candidate.mkdir(exist_ok=True)
        np.savetxt(candidate/'validation_row_indices.tsv',indices,fmt='%d')
        np.savetxt(candidate/'labels.tsv',y[indices],fmt='%d')
        for space,inputs in [('standardized',x),('raw',raw_inputs)]:
            folder=candidate/space;folder.mkdir(exist_ok=True)
            spec=specs[space];spec['working_point']=working
            (folder/'model.json').write_text(json.dumps(spec,indent=2,allow_nan=False)+'\n')
            (folder/'model_coefficients.h').write_text(cpp_header(spec))
            lower=training_min if space=='standardized' else training_min*scale+mean
            upper=training_max if space=='standardized' else training_max*scale+mean
            bounds=[]
            for tree in spec['trees']:
                w=np.asarray(tree['split_weights']);b=np.asarray(tree['split_bias']);leaf=np.asarray(tree['leaf_parameters'])
                split_low=b+np.minimum(w*lower,w*upper).sum(axis=1)
                split_high=b+np.maximum(w*lower,w*upper).sum(axis=1)
                if spec['leaf_mode']=='linear':
                    leaf_low=leaf[:,-1]+np.minimum(leaf[:,:-1]*lower,leaf[:,:-1]*upper).sum(axis=1)
                    leaf_high=leaf[:,-1]+np.maximum(leaf[:,:-1]*lower,leaf[:,:-1]*upper).sum(axis=1)
                else:leaf_low=leaf_high=leaf[:,0]
                bounds.append(dict(split_min=split_low.tolist(),split_max=split_high.tolist(),leaf_min=leaf_low.tolist(),leaf_max=leaf_high.tolist()))
            write_json(folder/'ranges.json',dict(input_min=lower.tolist(),input_max=upper.tolist(),trees=bounds,
                signal_margin_min=-sum(max(t['leaf_max']) for t in bounds),
                signal_margin_max=-sum(min(t['leaf_min']) for t in bounds),
                note='Conservative independent-feature bounds within observed training feature box; no clipping applied.'))
            expected,leaf,outputs=predict_margin(spec,inputs[indices])
            np.savetxt(folder/'vectors.tsv',inputs[indices],fmt='%.17g')
            np.savetxt(folder/'expected.tsv',np.column_stack([expected,leaf]),fmt=['%.17g']+['%d']*trees)
            np.savetxt(folder/'tree_free_logits.tsv',outputs,fmt='%.17g')
        metrics,roc=compute_metrics(y,model.predict_proba(x),meta['class_names'])
        assert abs(metrics['signal_efficiency_at_fixed_fpr']-payload['validation']['signal_efficiency_at_fixed_fpr'])<1e-7
        counts=operation_counts(specs['standardized'])
        record=dict(id=tag,run_name=run,training_policy=payload['config']['full_training_policy'],checkpoint_path=str(source.relative_to(resolve('.'))),
            checkpoint_sha256=hashlib.sha256(payload_bytes).hexdigest(),selected_epoch=payload['epoch'],
            training_run_converged=status['converged'],selected_checkpoint_converged=payload['training_converged_at_save'],
            metrics=metrics,deterministic_working_point=working,arithmetic=counts,complexity=model.complexity(),
            golden_vector_count=len(indices),raw_fold_max_margin_difference=float(np.max(np.abs(raw_margin-margin))),
            native_max_margin_difference=float(np.max(np.abs(margin+native))))
        write_json(candidate/'candidate.json',record);frozen.append(record)
        label=f'TAO: {trees} trees, depth {depth}, {model.leaf_mode}'
        items.append(dict(label=label,id=tag,model=model,color=plt.get_cmap('tab10')(index),metrics=metrics,roc=roc,parameters=counts['stored_scalar_parameters']))
        print('Frozen',tag,metrics['signal_efficiency_at_fixed_fpr'],flush=True)
    primary=max(frozen,key=lambda item:item['metrics']['signal_efficiency_at_fixed_fpr'])['id']
    next(item for item in items if item['id']==primary)['label']+=' (winner)'
    # Reference XGBoost vectors use the winner's identical standardized rows.
    index=np.loadtxt(output/primary/'validation_row_indices.tsv',dtype=int)
    np.savetxt(bdt_dir/'vectors.tsv',x[index],fmt='%.17g')
    np.savetxt(bdt_dir/'expected_signal_probability.tsv',bdt.predict_proba(x[index])[:,1],fmt='%.17g')
    import xgboost as xgb
    reloaded=xgb.Booster();reloaded.load_model(str(bdt_dir/'model.json'));reloaded.set_param({'nthread':1})
    assert np.max(np.abs(reloaded.inplace_predict(x[index])-bdt.predict_proba(x[index])[:,1]))<1e-7
    for size in [1,len(x)]:
        for item in items:
            item.setdefault('timings',[]).append(benchmark(item['model'].predict_proba,x[:size]))
    version=output.name;prefix='fpga_handoff_'+version
    root=resolve('plots/comparisons');title=f'Frozen FPGA handoff {version} | validation | oblique forests\nBest-efficiency checkpoints; CPU: one thread, current NumPy / XGBoost APIs'
    roc_figure(items,safe_label(meta['class_names'][1]),root/(prefix+'_roc.png'),title)
    ranked=sorted(items,key=lambda item:item['metrics']['signal_efficiency_at_fixed_fpr'],reverse=True)
    fig,ax=plt.subplots(figsize=(13,5));ax.axis('off')
    cells=[]
    for item in ranked:
        m=item['metrics'];cells.append([item['label'],str(item['parameters']),f"{m['signal_efficiency_at_fixed_fpr']:.2%}",f"{m['signal_auc']:.5f}",f"{item['timings'][-1]['median_microseconds_per_event']:.3f}"])
    table=ax.table(cellText=cells,colLabels=['Model','Parameters','Signal eff. at 1% bkg','AUC','CPU us/event'],cellLoc='center',colWidths=[.40,.12,.20,.13,.15],bbox=[0,.20,1,.68])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d9dee7')
        if row==0:cell.set_facecolor('#23334d');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        else:cell.set_facecolor('#edf5ff' if 'winner' in cells[row-1][0] else ('#f5f6f8' if row%2 else 'white'))
    fig.suptitle('XGBoost and frozen oblique-tree candidates',fontsize=19,y=.98)
    ax.text(.5,.94,'q/g vs W/Z/top | same validation split | best saved efficiency checkpoints',ha='center',transform=ax.transAxes,fontsize=12)
    ax.text(.01,.10,'Parameters: scalar coefficients including biases; shared preprocessing and topology excluded.',transform=ax.transAxes,fontsize=9)
    ax.text(.01,.04,'CPU: batch 132,800, one thread; NumPy TAO / compiled XGBoost. Frozen snapshots; some training runs continue.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(root/(prefix+'_comparison.png'),dpi=180);plt.close(fig)
    comparison=[{k:v for k,v in item.items() if k not in ['model','roc','color']} for item in ranked]
    write_json(output/'comparison.json',comparison)
    lines=['# Frozen FPGA handoff comparison','', '| Model | Parameters | Signal efficiency at 1% background | AUC | Bulk CPU us/event |','|---|---:|---:|---:|---:|']
    for row in cells:lines.append('| '+' | '.join(row)+' |')
    lines+=['','Validation only; test remains held out. Saved checkpoint hashes and training status are frozen in the handoff manifest. CPU compares the current software APIs, not FPGA latency. Source training status is recorded per candidate; some runs may still be active. Snapshot selection does not certify convergence.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    public=root/(prefix+'_report.md');public.write_text('\n'.join(lines)+'\n');syncer.register_file(public)
    files={str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest=dict(schema='tree-benchmark.fpga-handoff.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=resolve('.'),text=True).strip(),
        dataset_sha256=audit['dataset_sha256'],partition='validation',validation_events=len(y),test_evaluated=False,
        primary_metric='signal_efficiency_at_fixed_fpr',fixed_fpr=.01,candidates=frozen,
        primary_candidate=primary,
        xgboost=dict(parameters=items[0]['parameters'],metrics=bm,source_checkpoint_sha256=sha(resolve(ref['model']['output_model_path'])),native_json='xgboost_reference/model.json'),
        hardware=dict(quantized=False,synthesized=False,target_device=None,clock_target=None),files=files)
    write_json(output/'manifest.json',manifest)
    print('FPGA handoff frozen at',output,flush=True)


if __name__=='__main__':main()
