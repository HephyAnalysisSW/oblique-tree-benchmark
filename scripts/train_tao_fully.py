"""Resumable TAO training to an explicit numerical coordinate-convergence test."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,copy,hashlib,json,pickle,time
from datetime import datetime,timezone
import numpy as np
from data_utils import resolve,load_config,load_metadata,write_json
from evaluate import compute_metrics
from tao.hard import HardTAOClassifier
from tao.convergence import leaf_kkt_residual,convergence_check
from tao.initialization import initialize_cart,append_zero_tree
import matplotlib.pyplot as plt


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_pickle(path,payload):
    path=resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def atomic_json(path,payload):
    path=resolve(path);tmp=path.with_suffix(path.suffix+'.tmp')
    write_json(tmp,payload);tmp.replace(path)


def plots(run,state):
    h=state['history']
    if not h:return
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    epochs=[r['epoch'] for r in h]
    for ax,key,label in zip(axes,['signal_auc','signal_efficiency_at_fixed_fpr',None],
        ['Validation AUC','Signal efficiency at 1% background','Training compute wall time [min]']):
        values=[r['validation'][key] for r in h] if key else [r['cumulative_training_seconds']/60 for r in h]
        ax.plot(epochs,values,'o-');ax.set_xlabel('Continuation epoch');ax.set_ylabel(label);ax.grid(alpha=.2)
    c=h[-1]['convergence'];status='CONVERGED' if state['converged'] else 'TRAINING'
    fig.suptitle(f"{run}\n{status} | stable passes {c['stable_epochs']}/3 | leaf KKT {c['leaf_kkt_residual']:.2g}",fontsize=9)
    fig.tight_layout();fig.savefig(resolve('results/'+run+'_learning.png'),dpi=170);plt.close(fig)


def save(run,model,config,metadata,state):
    atomic_pickle('models/'+run+'_latest.pkl',dict(model=model,config=config,metadata=metadata,state=state))
    atomic_json('results/'+run+'_history.json',state['history'])
    atomic_json('results/'+run+'_status.json',dict(run_name=run,phase=state['phase'],converged=state['converged'],
        epochs=len(state['history']),stable_epochs=state['stable_epochs'],
        training_seconds=state['training_seconds'],leaf_max_iter=state['local_leaf_max_iter'],
        last_convergence=state['history'][-1]['convergence'] if state['history'] else None,
        updated_utc=datetime.now(timezone.utc).isoformat(),test_evaluated=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--config',default='configs/tao_full_training_v1.json')
    p.add_argument('--wall-seconds',type=float,default=2700)
    a=p.parse_args();scan=json.loads(resolve(a.config).read_text())
    layout=scan['layouts'][a.index]
    seed=scan['seed']
    run=(f"{scan['task']}_hardtao_{scan['leaf_mode']}_d{layout['depth']}_n{layout['trees']}"
         f"_sa{scan['split_alpha']:g}_la{scan['leaf_alpha']:g}_seed{scan['seed']}_{scan.get('run_suffix','full_training_v1')}")
    latest=resolve('models/'+run+'_latest.pkl')
    audit=json.loads(resolve('results/tao_split_audit.json').read_text())
    reference=load_config('configs/bdt_'+scan['task']+'_seed42.yaml')
    assert sha(resolve(reference['data']['dataset_path']))==audit['dataset_sha256']
    metadata=load_metadata(reference['data']['metadata_path'])
    with np.load(resolve(reference['data']['dataset_path'])) as d:
        x,y=d['X_train'].astype(float),d['y_train'];xv,yv=d['X_val'].astype(float),d['y_val']
    if latest.exists():
        with latest.open('rb') as f:payload=pickle.load(f)
        model,config,state=payload['model'],payload['config'],payload['state']
        assert config['dataset_sha256']==audit['dataset_sha256']
        # Appending another layout does not change an existing member's policy.
        assert {k:v for k,v in config['full_training_policy'].items() if k!='layouts'}=={k:v for k,v in scan.items() if k!='layouts'}
    else:
        marker=resolve(f"results/{scan['source_scan']}_member_{a.index}.json")
        source_run=None;source_sha=None;source_seconds=0
        initial_started=time.perf_counter()
        initialization_records=[]
        if scan.get('source_models'):
            source_run=scan['source_models'][a.index]
            source_cfg=load_config('configs/'+source_run+'.yaml')
            with resolve(source_cfg['model']['output_model_path']).open('rb') as f:source=pickle.load(f)
            assert source['training_converged'] and source_cfg['data']==reference['data']
            model=copy.deepcopy(source['model']);source_sha=sha(resolve(source_cfg['model']['output_model_path']))
            source_seconds=source_cfg['source_training_seconds']+source['history'][-1]['cumulative_training_seconds']
            if scan.get('initialization')=='append_tree':append_zero_tree(model,layout['trees'],seed)
        elif scan.get('source_scan') and marker.exists():
            member=json.loads(marker.read_text())
            if 'layout' in member:assert member['layout']==layout
            source_run=member.get('selected_run',member.get('run_name'));source_cfg=load_config('configs/'+source_run+'.yaml')
            with resolve(source_cfg['model']['output_model_path']).open('rb') as f:source=pickle.load(f)
            model=copy.deepcopy(source['model']);assert source_cfg['data']==reference['data']
            source_sha=sha(resolve(source_cfg['model']['output_model_path']))
            if 'base_run' in member:
                base=json.loads(resolve('results/'+member['base_run']+'_history.json').read_text())
                refit=json.loads(resolve('results/'+source_run+'_history.json').read_text())
                source_seconds=sum(r['profile']['seconds'] for r in base)+sum(r['seconds'] for r in refit)
            else:
                source_seconds=member['training_seconds']+member.get('source_training_seconds',0.)
        else:
            model=HardTAOClassifier(depth=layout['depth'],trees=layout['trees'],leaf_mode=scan['leaf_mode'],
                seed=scan['seed'],split_alpha=scan['split_alpha'],leaf_alpha=scan['leaf_alpha'],
                indexed='dense').initialize(x.shape[1],len(metadata['class_names']))
            if scan.get('initialization')=='cart':initialization_records=initialize_cart(model,x,y)
        initialization_seconds=time.perf_counter()-initial_started
        config=dict(run_name=run,seed=scan['seed'],data=reference['data'],dataset_sha256=audit['dataset_sha256'],
            implementation_sha256=sha(resolve('src/tao/hard.py')),full_training_policy=scan,
            split_search_sha256=sha(resolve('src/tao/split_search.py')),
            initialization=scan.get('initialization','random' if source_run is None else 'checkpoint'),
            initialization_records=initialization_records,initialization_seconds=initialization_seconds,
            source_run=source_run,source_model_sha256=source_sha,source_training_seconds=source_seconds,
            model=dict(type='hard_tao_cross_entropy',output_model_path='models/'+run+'.pkl',
                tao=dict(**layout,leaf_mode=scan['leaf_mode'],split_alpha=scan['split_alpha'],leaf_alpha=scan['leaf_alpha'])),
            evaluation=dict(output_prefix='results/'+run,fixed_fpr=.01))
        state=dict(phase='initial_refit',history=[],stable_epochs=0,converged=False,
            training_seconds=initialization_seconds,local_leaf_max_iter=scan['local_leaf_max_iter'],split_max_iter=scan['split_max_iter'],
            initial_refits=[],best_validation_auc=-float('inf'),step_seconds_estimate=600.)
    model.split_alpha=scan['split_alpha'];model.leaf_alpha=scan['leaf_alpha']
    model.split_solver=scan.get('split_solver','liblinear')
    model.split_search=scan.get('split_search','surrogate')
    model.leaf_ftol=scan['leaf_ftol'];model.leaf_gtol=scan['leaf_gtol'];model.indexed='dense'
    assert model.depth==layout['depth'] and model.ntrees==layout['trees']
    assert model.leaf_mode==scan['leaf_mode']
    assert np.all(model.transform.mean==0) and np.all(model.transform.scale==1)
    atomic_json('configs/'+run+'.yaml',config)
    started=time.perf_counter()
    while not state['converged']:
        elapsed=time.perf_counter()-started
        if elapsed+min(1200,max(300,state['step_seconds_estimate']*1.5))>a.wall_seconds:
            save(run,model,config,metadata,state);plots(run,state)
            print(json.dumps(dict(run=run,phase=state['phase'],continuation_required=True)),flush=True)
            return 75
        tick=time.perf_counter()
        if state['phase']=='epoch':
            state['before_routes']=[t.leaf_indices(x,'dense') for t in model.trees]
            model.leaf_max_iter=state['local_leaf_max_iter'];model.split_max_iter=state['split_max_iter']
            state['pending_profile']=model.fit_epoch(x,y,global_refit=False)
            state['pending_global']=[];state['phase']='global_refit'
            duration=time.perf_counter()-tick;state['training_seconds']+=duration
            state['pending_profile']['seconds']=duration
        else:
            model.leaf_max_iter=scan['global_leaf_chunk_iter']
            before=model.objective(x,y)
            stats=model.global_refit(x,y);kkt=leaf_kkt_residual(model,x,y)
            after=model.objective(x,y)
            assert after<=before+1e-8
            stats['optimizer_converged']=stats['converged']
            stats['leaf_kkt_residual']=kkt
            stats['converged']=bool(kkt<=scan['convergence']['kkt_tol'])
            stats['convergence_test']='normalized_CE_plus_L1_leaf_KKT'
            duration=time.perf_counter()-tick;state['training_seconds']+=duration
            stats['seconds_including_checks']=duration
            dest=state['initial_refits'] if state['phase']=='initial_refit' else state['pending_global']
            dest.append(stats)
            print(json.dumps(dict(run=run,phase=state['phase'],kkt=kkt,global_converged=stats['converged'],
                optimizer_converged=stats['optimizer_converged'],seconds=duration)),flush=True)
            if stats['converged']:
                if state['phase']=='initial_refit':state['phase']='epoch'
                else:
                    profile=state.pop('pending_profile');chunks=state.pop('pending_global')
                    profile['global_refit']=stats;profile['global_refit_chunks']=chunks
                    profile['seconds']+=sum(s['seconds_including_checks'] for s in chunks)
                    profile['objective_after']=after
                    changed=np.zeros(len(y),dtype=bool)
                    for t,previous in zip(model.trees,state.pop('before_routes')):
                        changed|=t.leaf_indices(x,'dense')!=previous
                    check=convergence_check(profile,kkt,float(changed.mean()),state['stable_epochs'],**scan['convergence'])
                    metrics,_=compute_metrics(yv,model.predict_proba(xv),metadata['class_names'])
                    row=dict(epoch=len(state['history'])+1,profile=profile,convergence=check,validation=metrics,
                             cumulative_training_seconds=state['training_seconds'])
                    state['history'].append(row);state['stable_epochs']=check['stable_epochs']
                    state['converged']=bool(check['converged'] and len(state['history'])>=scan['minimum_continuation_epochs'])
                    state['phase']='complete' if state['converged'] else 'epoch'
                    if any(not s['converged'] for s in profile['leaf_fits']):
                        state['local_leaf_max_iter']=min(4000,state['local_leaf_max_iter']*2)
                    if any(not s['converged'] for s in profile['split_fits']):
                        state['split_max_iter']=min(8000,state['split_max_iter']*2)
                    if metrics['signal_auc']>state['best_validation_auc']:
                        state['best_validation_auc']=metrics['signal_auc']
                        atomic_pickle('models/'+run+'_best_validation.pkl',dict(model=model,config=config,metadata=metadata,
                            validation=metrics,epoch=row['epoch'],training_converged_at_save=state['converged']))
                    efficiency=metrics['signal_efficiency_at_fixed_fpr']
                    if (efficiency,metrics['signal_auc'])>(state.get('best_validation_efficiency',-float('inf')),
                                                          state.get('best_efficiency_auc',-float('inf'))):
                        state['best_validation_efficiency']=efficiency
                        state['best_efficiency_auc']=metrics['signal_auc']
                        atomic_pickle('models/'+run+'_best_efficiency.pkl',dict(model=model,config=config,metadata=metadata,
                            validation=metrics,epoch=row['epoch'],training_converged_at_save=state['converged'],
                            selected_by='signal_efficiency_at_fixed_fpr',fixed_fpr=.01,
                            secondary_metric='signal_auc'))
                    print(json.dumps(dict(run=run,epoch=row['epoch'],objective=after,auc=metrics['signal_auc'],
                        efficiency=metrics['signal_efficiency_at_fixed_fpr'],**check)),flush=True)
                    plots(run,state)
        state['step_seconds_estimate']=duration
        save(run,model,config,metadata,state)
    atomic_pickle(config['model']['output_model_path'],dict(model=model,config=config,metadata=metadata,
        history=state['history'],convergence=state['history'][-1]['convergence'],training_converged=True))
    atomic_json('results/'+run+'_validation.json',dict(run_name=run,validation=state['history'][-1]['validation'],
        training_converged=True,convergence=state['history'][-1]['convergence'],complexity=model.complexity(),
        test_evaluated=False,best_validation_auc=state['best_validation_auc'],
        best_validation_efficiency=state.get('best_validation_efficiency'),
        primary_metric='signal_efficiency_at_fixed_fpr'))
    atomic_json(f"results/{scan['scan_name']}_member_{a.index}.json",dict(index=a.index,run_name=run,complete=True,
        converged=True,training_seconds=state['training_seconds'],source_training_seconds=config['source_training_seconds']))
    return 0


if __name__=='__main__':sys.exit(main())
