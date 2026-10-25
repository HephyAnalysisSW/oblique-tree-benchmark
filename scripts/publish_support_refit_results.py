"""Finalize the support-refit scan using exact saved coefficients and validation only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import hashlib,json,pickle,shutil
from datetime import datetime,timezone
import numpy as np
import matplotlib.pyplot as plt
from data_utils import resolve,load_config,load_metadata,write_json
from evaluate import compute_metrics,safe_label
from scripts.compare_tao_inference import benchmark,bdt_counts
from scripts.summarize_tao_layout_scan import roc_figure
from tao.sparsity import sparsity_profile
from tao.portable import export_model,predict_margin,deterministic_working_point
from tao.uncertainty import paired_intervals
import common.syncer as syncer


def main():
    plan=json.loads(resolve('configs/tao_support_refit_scan_v1.json').read_text())
    ref=load_config('configs/bdt_qg_vs_wzt_seed42.yaml');meta=load_metadata(ref['data']['metadata_path'])
    audit=json.loads(resolve('results/tao_split_audit.json').read_text());h=hashlib.sha256()
    with resolve(ref['data']['dataset_path']).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==audit['dataset_sha256']
    with np.load(resolve(ref['data']['dataset_path'])) as data:
        x,y=data['X_val'].astype(float),data['y_val'];mean,scale=data['scaler_mean'],data['scaler_scale']
    reference=pickle.loads(resolve(ref['model']['output_model_path']).read_bytes())['model'];reference.get_booster().set_param({'nthread':1})
    bc=bdt_counts(reference);bm,br=compute_metrics(y,reference.predict_proba(x),meta['class_names'])
    reference_item=dict(label='XGBoost: 100 trees, D4',color='black',model=reference,metrics=bm,roc=br,
        timings=[benchmark(reference.predict_proba,x)],parameters=bc['internal_nodes']+bc['leaves']+1)
    results=[];eligible=[];root=resolve('plots/comparisons');selected_dir=resolve('models/support_refit_scan_v1_selected');selected_dir.mkdir(parents=True,exist_ok=True)
    for case in plan['cases']:
        cfg=json.loads(resolve(case['config']).read_text());status=json.loads(resolve('results/'+case['run_name']+'_status.json').read_text())
        source=resolve(cfg['source_checkpoint']);blob=source.read_bytes();assert hashlib.sha256(blob).hexdigest()==cfg['source_sha256']
        parent=pickle.loads(blob);assert parent['training_converged'];stages={}
        candidates=[('parent',parent['model'],cfg['source_checkpoint'],cfg['source_sha256'],status['parent']['metrics']['validation'])]
        for stage in ['leaf_only','full']:
            terminal=resolve('models/'+case['run_name']+'_'+stage+'.pkl')
            if terminal.exists():
                raw=terminal.read_bytes();p=pickle.loads(raw);assert p['training_converged'];p['model'].assert_support()
                candidates.append((stage,p['model'],str(terminal.relative_to(resolve('.'))),hashlib.sha256(raw).hexdigest(),p['metrics']['validation']))
            best=resolve('models/'+case['run_name']+'_'+stage+'_best_efficiency.pkl')
            if best.exists():
                raw=best.read_bytes();p=pickle.loads(raw);p['model'].assert_support()
                candidates.append((stage+'_best',p['model'],str(best.relative_to(resolve('.'))),hashlib.sha256(raw).hexdigest(),p['metrics']['validation']))
        for stage,model,path,digest,expected in candidates:
            metrics,roc=compute_metrics(y,model.predict_proba(x),meta['class_names'])
            for key in ['signal_efficiency_at_fixed_fpr','signal_auc']:assert abs(metrics[key]-expected[key])<1e-7,(case['id'],stage,key)
            counts=sparsity_profile(model);assert counts['nonzero_coefficients']<=case['parent_nonzeros']
            timing=benchmark(model.predict_proba,x)
            record=dict(stage=stage,metrics=metrics,sparsity=counts,source=path,sha256=digest,timing=timing)
            stages[stage]=record
            certified=(stage=='parent' or stage.removesuffix('_best') in status.get('stages',{}))
            if certified:
                target=selected_dir/(digest+'.pkl')
                frozen_bytes=resolve(path).read_bytes()
                assert hashlib.sha256(frozen_bytes).hexdigest()==digest,'Checkpoint changed during finalization; retry'
                if not target.exists():target.write_bytes(frozen_bytes)
                assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
                record['immutable_snapshot']=str(target.relative_to(resolve('.')))
                eligible.append(dict(case=case,record=record,model=model,roc=roc))
        amendment_path=resolve('results/'+case['run_name']+'_budget_amendments.json')
        amendments=json.loads(amendment_path.read_text()) if amendment_path.exists() else []
        results.append(dict(case=case,phase=status['phase'],selection_penalties=parent['model'].__dict__.get('parent_penalties',dict(split_alpha=parent['model'].split_alpha,leaf_alpha=parent['model'].leaf_alpha)),
            stages=stages,training_seconds=status['training_seconds'],solver_budget_amendments=amendments))
    references=[]
    for tag,pattern in [('previous_1t_no_l1','*_hyper_scan_v1_c12_best_efficiency.pkl'),('previous_3t_peak','*_l1_path_v1_leaf_01_best_efficiency.pkl')]:
        paths=list(resolve('models').glob(pattern))
        if not paths:continue
        assert len(paths)==1
        raw=paths[0].read_bytes();p=pickle.loads(raw);cfg=p['config'];assert cfg['dataset_sha256']==audit['dataset_sha256']
        source_status=json.loads(resolve('results/'+cfg['run_name']+'_status.json').read_text());assert source_status['converged']
        model=p['model'];metrics,roc=compute_metrics(y,model.predict_proba(x),meta['class_names'])
        for key in ['signal_efficiency_at_fixed_fpr','signal_auc']:assert abs(metrics[key]-p['validation'][key])<1e-7
        counts=sparsity_profile(model);digest=hashlib.sha256(raw).hexdigest();target=selected_dir/(digest+'.pkl')
        if not target.exists():target.write_bytes(raw)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
        case=dict(id=tag,trees=model.ntrees,depth=model.depth,leaf_mode=model.leaf_mode,parent_nonzeros=counts['nonzero_coefficients'])
        record=dict(stage='previous_best',metrics=metrics,sparsity=counts,source=str(paths[0].relative_to(resolve('.'))),sha256=digest,
            immutable_snapshot=str(target.relative_to(resolve('.'))),timing=benchmark(model.predict_proba,x))
        references.append(dict(case=case,record=record));eligible.append(dict(case=case,record=record,model=model,roc=roc))
    winner=max(eligible,key=lambda p:(p['record']['metrics']['signal_efficiency_at_fixed_fpr'],p['record']['metrics']['signal_auc'])) if eligible else None
    matches=[p for p in eligible if p['record']['metrics']['signal_efficiency_at_fixed_fpr']>=bm['signal_efficiency_at_fixed_fpr']]
    compact=min(matches,key=lambda p:(p['record']['sparsity']['nonzero_coefficients'],-p['record']['metrics']['signal_efficiency_at_fixed_fpr'])) if matches else None
    compact_path=min(matches,key=lambda p:(p['record']['sparsity']['selected_path_multiplications_upper_bound'],
        p['record']['sparsity']['nonzero_coefficients'],-p['record']['metrics']['signal_efficiency_at_fixed_fpr'])) if matches else None
    chosen=[]
    for p in [winner,compact,compact_path]:
        if p and not any(q['case']['id']==p['case']['id'] for q in chosen):chosen.append(p)
    count=lambda p:p['record']['sparsity']['nonzero_coefficients']
    efficiency=lambda p:p['record']['metrics']['signal_efficiency_at_fixed_fpr']
    frontier=[p for p in matches if not any(count(q)<=count(p) and efficiency(q)>=efficiency(p)
        and (count(q)<count(p) or efficiency(q)>efficiency(p)) for q in matches)]
    while len(chosen)<5:
        remaining=[p for p in frontier if not any(q['case']['id']==p['case']['id'] for q in chosen)]
        if not remaining:break
        # Cover the useful compression range, rather than plotting only several
        # nearly identical high-parameter checkpoints.
        selected=max(remaining,key=lambda p:(min(abs(np.log(max(1,count(p)))-np.log(max(1,count(q)))) for q in chosen),efficiency(p)))
        chosen.append(selected)
    plot_candidates=list(chosen)
    if winner and winner['record']['stage'] not in ('parent','previous_best'):
        parent_point=next((p for p in eligible if p['case']['id']==winner['case']['id'] and p['record']['stage']=='parent'),None)
        if parent_point:plot_candidates.append(parent_point)
    plot_items=[reference_item]
    for i,p in enumerate(plot_candidates):
        c,r=p['case'],p['record'];plot_items.append(dict(label=f"{c['id']}/{r['stage']}: {c['trees']} x D4 linear, {r['sparsity']['nonzero_coefficients']} nonzeros",
            color=plt.get_cmap('tab10')(i),metrics=r['metrics'],roc=p['roc'],timings=[r['timing']]))
    roc_figure(plot_items,safe_label(meta['class_names'][1]),root/'support_refit_scan_v1_roc.png','Frozen-support refit comparison | validation | completed source stages')
    cells=[]
    fmt=lambda value:'—' if value is None else f'{100*value:.2f}%'
    for result in results:
        c,s=result['case'],result['stages'];metric=lambda stage:s.get(stage,{}).get('metrics',{}).get('signal_efficiency_at_fixed_fpr')
        best=s.get('full_best',s.get('full',{}));auc=best.get('metrics',{}).get('signal_auc');timing=best.get('timing',{}).get('median_microseconds_per_event')
        penalties=result['selection_penalties']
        best_display=fmt(best.get('metrics',{}).get('signal_efficiency_at_fixed_fpr'))
        if best and result['phase']!='complete':best_display+=' *'
        cells.append([c['id'],f"{c['trees']} x D4 linear",f"{penalties['split_alpha']:g}/{penalties['leaf_alpha']:g}",str(c['parent_nonzeros']),
            fmt(metric('parent')),fmt(metric('leaf_only')),fmt(metric('full')),fmt(best.get('metrics',{}).get('signal_efficiency_at_fixed_fpr')),
            '—' if auc is None else f'{auc:.5f}','—' if timing is None else f'{timing:.3f}',result['phase']])
        cells[-1][7]=best_display
    columns=['Selector','Architecture','Split/leaf L1','Support','Parent eff.','Leaf refit','Full terminal','Best saved full','AUC','CPU us/event','Status']
    fig,ax=plt.subplots(figsize=(19,1.9+.42*len(cells)));ax.axis('off')
    table=ax.table(cellText=cells,colLabels=columns,cellLoc='center',bbox=[0,.13,1,.75]);table.auto_set_font_size(False);table.set_fontsize(8)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d9dee7');cell.set_facecolor('#23334d' if r==0 else '#f0f5fb' if r%2 else 'white')
        if r==0:cell.get_text().set_color('white')
    fig.suptitle(f"Frozen-support refits | validation eff. at 1% background | XGBoost {100*bm['signal_efficiency_at_fixed_fpr']:.2f}%",fontsize=15)
    ax.text(0,.04,'3 x D4 = 1,581 dense slots; 1 x D4 = 527. Support includes biases. CPU: one thread, NumPy vs compiled XGBoost; no FPGA latency claim.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(root/'support_refit_scan_v1_final_table.png',dpi=180);plt.close(fig)
    selections={};bootstrap_scores={'XGBoost':reference.predict_proba(x)[:,1]}
    recommendation_rows=[['XGBoost reference',f"{bc.get('trees',100)} x D{bc.get('maximum_depth',4)} constant",str(reference_item['parameters']),
        '—','—','—',f"{bm['signal_efficiency_at_fixed_fpr']:.2%}",f"{bm['signal_auc']:.5f}",
        f"{reference_item['timings'][-1]['median_microseconds_per_event']:.3f}"]]
    for tag,p in [('winner',winner),('compact_match_xgb',compact),('compact_path_match_xgb',compact_path)]:
        if not p:continue
        record=p['record'];selections[tag]=dict(case=p['case'],record=record)
        bootstrap_scores[tag]=p['model'].predict_proba(x)[:,1]
        c,s,m=p['case'],record['sparsity'],record['metrics']
        recommendation_rows.append([{'winner':'Highest efficiency','compact_match_xgb':'Fewest coefficients','compact_path_match_xgb':'Fewest path MACs'}[tag],
            f"{c['trees']} x D{c['depth']} linear",str(s['dense_parameter_slots']),str(s['nonzero_coefficients']),str(s['reachable_nonzero_coefficients']),
            str(s['selected_path_multiplications_upper_bound']),f"{m['signal_efficiency_at_fixed_fpr']:.2%}",f"{m['signal_auc']:.5f}",
            f"{record['timing']['median_microseconds_per_event']:.3f}"])
        folder=selected_dir/tag;folder.mkdir(exist_ok=True)
        standardized=export_model(p['model'],meta['feature_names'],meta['class_names'],mean,scale)
        margin,routes,_=predict_margin(standardized,x)
        assert np.max(np.abs(margin+p['model'].decision_function(x)[:,0]))<1e-9
        working_point=deterministic_working_point(margin,y)
        rng=np.random.default_rng(42);indices=set(rng.choice(len(x),min(128,len(x)),replace=False).tolist())
        indices.update(np.argsort(np.abs(margin-working_point['signal_margin_threshold']))[:16].tolist())
        for tree in p['model'].trees:
            for node,rows in enumerate(tree.routes(x,p['model'].indexed)[:tree.internal]):
                if len(rows):
                    distance=np.abs(x[rows]@tree.split_w[node]+tree.split_b[node])
                    indices.update(rows[np.argsort(distance)[:4]].tolist())
        np.savetxt(folder/'validation_row_indices.tsv',sorted(indices),fmt='%d')
        for raw in [False,True]:
            spec=export_model(p['model'],meta['feature_names'],meta['class_names'],mean,scale,raw=raw)
            if raw:
                raw_margin,raw_routes,_=predict_margin(spec,x*scale+mean)
                assert np.array_equal(routes,raw_routes) and np.max(np.abs(raw_margin-margin))<1e-8
                spec['working_point']=deterministic_working_point(raw_margin,y)
            else:spec['working_point']=working_point
            write_json(folder/('raw_model.json' if raw else 'standardized_model.json'),spec)
    recommendation_columns=['Role','Architecture','Dense slots','Nonzeros','Reachable','MAC bound','Eff. at 1%','AUC','CPU us/event']
    if recommendation_rows:
        fig,ax=plt.subplots(figsize=(16,3.8));ax.axis('off')
        table=ax.table(cellText=recommendation_rows,colLabels=recommendation_columns,cellLoc='center',
            colWidths=[.17,.18,.08,.08,.09,.09,.11,.10,.10],bbox=[0,.24,1,.57]);table.auto_set_font_size(False);table.set_fontsize(10)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#d9dee7');cell.set_facecolor('#23334d' if r==0 else '#f0f5fb' if r%2 else 'white')
            if r==0:cell.get_text().set_color('white')
        fig.suptitle('Selected coefficient-matched checkpoints | validation | XGBoost eff. '+f"{bm['signal_efficiency_at_fixed_fpr']:.2%}",fontsize=15)
        ax.text(0,.12,'Compact roles require at least XGBoost efficiency. Reachability removes only algebraically constant routes; MACs are sparse scalar bounds.',transform=ax.transAxes,fontsize=9)
        ax.text(0,.03,'CPU: one thread, full validation batch, NumPy vs compiled XGBoost. These counts are not synthesized FPGA resources or latency.',transform=ax.transAxes,fontsize=9)
        fig.tight_layout();fig.savefig(root/'support_refit_scan_v1_recommendations.png',dpi=180);plt.close(fig)
    comparisons={}
    if winner and winner['record']['stage'] not in ('parent','previous_best'):
        parent_point=next((p for p in eligible if p['case']['id']==winner['case']['id'] and p['record']['stage']=='parent'),None)
        if parent_point:
            bootstrap_scores['winner_parent']=parent_point['model'].predict_proba(x)[:,1]
            comparisons['winner_vs_parent']=('winner','winner_parent')
    uncertainty=paired_intervals(y,bootstrap_scores,comparisons=comparisons)
    for tag,selected in selections.items():assert abs(uncertainty['models'][tag]['efficiency']-selected['record']['metrics']['signal_efficiency_at_fixed_fpr'])<1e-7
    summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),models=results,selections=selections,reference_models=references,
        xgboost_metrics=bm,xgboost_counts=bc,xgboost_parameters=reference_item['parameters'],xgboost_timing=reference_item['timings'][-1],
        dataset_sha256=audit['dataset_sha256'],test_evaluated=False,uncertainty=uncertainty)
    write_json(root/'support_refit_scan_v1_final_summary.json',summary)
    lines=['# Frozen-support refit scan: coefficient-matched final comparison','',summary['updated_utc'],'',
        '| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+['| '+' | '.join(row)+' |' for row in cells]
    lines+=['','Selected tradeoffs (parent and established reference checkpoints remain eligible):','',
        '| '+' | '.join(recommendation_columns)+' |','|'+'|'.join(['---']*len(recommendation_columns))+'|']+['| '+' | '.join(row)+' |' for row in recommendation_rows]
    lines+=['','Paired class-stratified validation bootstrap, 1,000 replicates (95% percentile intervals):','',
        '| Candidate | Efficiency [95% interval] | Difference vs XGBoost, percentage points [95% interval] |', '|---|---:|---:|']
    for tag,u in uncertainty['models'].items():
        lo,hi=u['interval95'];dl,dh=u['difference_interval95']
        lines.append(f"| {tag} | {u['efficiency']:.2%} [{lo:.2%}, {hi:.2%}] | {100*u['difference_vs_reference']:+.2f} [{100*dl:+.2f}, {100*dh:+.2f}] |")
    for tag,u in uncertainty['comparisons'].items():
        lo,hi=u['interval95'];lines+=['',f"{tag}: {100*u['difference']:+.2f} percentage points [{100*lo:+.2f}, {100*hi:+.2f}]."]
    lines+=['',uncertainty['note']]
    amendments=[(row['case']['id'],a) for row in results for a in row['solver_budget_amendments']]
    if amendments:
        lines+=['','Recorded auxiliary solver budget continuations (all stopping tolerances unchanged):','']
        for case,a in amendments:
            lines.append(f"- `{case}`: split iterations {a['previous_split_max_iter']} -> {a['split_max_iter']}, after {a['completed_history_rows']} evaluated history rows. {a['reason']} Coefficients were bit-identical at the transition; original checkpoint SHA-256 `{a['previous_checkpoint_sha256']}` is retained in the policy archive.")
    if references:
        lines+=['','Previously established reference checkpoints (from completed runs):','', '| Model | Architecture | Nonzeros | Efficiency at 1% background | AUC |','|---|---|---:|---:|---:|']
        for r in references:
            c,m=r['case'],r['record'];lines.append(f"| {c['id']} | {c['trees']} x D{c['depth']} linear | {m['sparsity']['nonzero_coefficients']} | {m['metrics']['signal_efficiency_at_fixed_fpr']:.2%} | {m['metrics']['signal_auc']:.5f} |")
    lines+=['', 'Parent and refit values use the same support. Recommendations consider converged parents, completed refit stages, their archived best snapshots and the established references above. A selected snapshot can precede its source stage terminal. An asterisk marks a diagnostic snapshot from an unsuccessful full stage; it is not eligible for recommendation. Held full refits are not treated as converged; a completed leaf-only stage remains a valid candidate. Parameter and operation counts refer to standardized-input models; folding preprocessing into raw-input coefficients can change bias values and their zero counts. These counts are not synthesized FPGA resource or latency measurements. Test data remain untouched.',
        '', '![Selected table](support_refit_scan_v1_recommendations.png)','','![Full table](support_refit_scan_v1_final_table.png)','','![ROC](support_refit_scan_v1_roc.png)']
    (root/'support_refit_scan_v1_final_report.md').write_text('\n'.join(lines)+'\n')
    for name in ['support_refit_scan_v1_final_summary.json','support_refit_scan_v1_final_report.md']:syncer.register_file(root/name)
    syncer.sync();print('Finalized support-refit results:',len(results),'cases;',len(eligible),'eligible certified-stage snapshots',flush=True)


if __name__=='__main__':main()
