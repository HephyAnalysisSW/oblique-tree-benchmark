"""Explicit single-parent select/freeze/refit worker; does not submit a scan.

Exit 75: resumable scheduler budget; 78: logistic separation; 79: inconclusive
separation check; 80: numerical/proposal stall. Only exit 0 means both requested
stages completed their configured numerical checks. No test-array access.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,fcntl,hashlib,json,pickle,re,time
from datetime import datetime,timezone
import numpy as np
from data_utils import resolve,load_config,load_metadata,write_json
from evaluate import compute_metrics
from tao.hard import cross_entropy_rows
from tao.support import freeze_support,continue_full_refit,separation_check
from tao.epoch import begin_epoch,step_epoch
from tao.convergence import leaf_kkt_residual,convergence_check
from tao.sparsity import sparsity_profile

IMPLEMENTATION=['src/tao/hard.py','src/tao/support.py','src/tao/epoch.py',
                'src/tao/convergence.py','src/tao/split_search.py','src/tao/sparsity.py',
                'src/data_utils.py','src/evaluate.py','scripts/train_tao_support_refit.py']


def sha(path):
    h=hashlib.sha256()
    with resolve(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_pickle(path,payload):
    path=resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL));temporary.replace(path)


def atomic_json(path,payload):
    path=resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');write_json(temporary,payload);temporary.replace(path)


def evaluate(model,x,y,xv,yv,classes):
    return dict(training_cross_entropy=float(cross_entropy_rows(model.decision_function(x),y).mean()),
        validation=compute_metrics(yv,model.predict_proba(xv),classes)[0],
        sparsity=sparsity_profile(model),support=model.support_counts())


def route_digest(model,x):
    h=hashlib.sha256()
    for tree in model.trees:h.update(tree.leaf_indices(x,model.indexed).tobytes())
    return h.hexdigest()


def persist(prefix,payload):
    model,state=payload['model'],payload['state'];model.assert_support()
    atomic_pickle('models/'+prefix+'_latest.pkl',payload)
    atomic_json('results/'+prefix+'_status.json',dict(run_name=prefix,phase=state['phase'],stage=model.support_mode,
        converged=state['phase']=='complete',history=state['history'],stages=state['stages'],
        parent=state['parent'],support=model.support_counts(),separation=state.get('separation'),
        last_global_fit=state.get('last_global'),global_chunks=state['global_chunks'],
        phase_progress={k:state['work'][k] for k in ['stage','tree_index','leaf_index','node_index'] if k in state.get('work',{})},
        training_seconds=state['training_seconds'],test_evaluated=False,updated_utc=datetime.now(timezone.utc).isoformat()))


def train(plan,wall_seconds=2700.,retry_separation=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',plan['run_name']):raise ValueError('Use a simple unique run_name')
    stages=plan.get('stages',['leaf_only','full'])
    if stages not in (['leaf_only'],['leaf_only','full']):raise ValueError('Refit leaf-only first; optionally continue full')
    if plan.get('exact_zero_support',True) is not True:raise ValueError('Only exact-zero support selection is implemented')
    if plan.get('minimum_epochs',5)<5:raise ValueError('Require at least five epochs per stage')
    reference=load_config(plan.get('reference_config','configs/bdt_qg_vs_wzt_seed42.yaml'))
    audit=json.loads(resolve('results/tao_split_audit.json').read_text())
    if sha(reference['data']['dataset_path'])!=audit['dataset_sha256']:raise ValueError('Dataset differs from audited split')
    metadata=load_metadata(reference['data']['metadata_path'])
    with np.load(resolve(reference['data']['dataset_path'])) as data:
        x,y=data['X_train'].astype(float),data['y_train'];xv,yv=data['X_val'].astype(float),data['y_val']
    code={path:sha(path) for path in IMPLEMENTATION}
    prefix=plan['run_name'];checkpoint=resolve('models/'+prefix+'_latest.pkl')
    if checkpoint.exists():
        payload=pickle.loads(checkpoint.read_bytes())
        if payload['plan']!=plan or payload['implementation_sha256']!=code:raise ValueError('Resume policy or implementation changed; use a new run name')
        if payload['dataset_sha256']!=audit['dataset_sha256']:raise ValueError('Resume dataset changed')
        if payload['metadata']!=metadata:raise ValueError('Resume metadata changed')
        model,state=payload['model'],payload['state'];model.assert_support()
        if state['phase']=='inconclusive' and retry_separation:state['phase']='separation'
        if state['phase'] in ('separated','inconclusive','stalled','complete'):
            return dict(separated=78,inconclusive=79,stalled=80,complete=0)[state['phase']]
    else:
        blob=resolve(plan['source_checkpoint']).read_bytes();digest=hashlib.sha256(blob).hexdigest()
        if digest!=plan['source_sha256']:raise ValueError('Parent checkpoint SHA-256 mismatch')
        parent=pickle.loads(blob)
        if parent.get('training_converged') is not True:raise ValueError('Require a converged terminal parent, not a best-efficiency snapshot')
        if parent['config']['dataset_sha256']!=audit['dataset_sha256'] or parent['config']['data']!=reference['data']:
            raise ValueError('Parent used different data or preprocessing')
        if parent['metadata']!=metadata:raise ValueError('Parent metadata mismatch')
        original=parent['model']
        if original.n_features_in_!=x.shape[1] or original.n_classes_!=len(metadata['class_names']):raise ValueError('Parent dimensions differ')
        if not (np.all(original.transform.mean==0) and np.all(original.transform.scale==1)):
            raise ValueError('Require standardized-data parent with identity transform')
        model=freeze_support(original)
        if not np.array_equal(original.decision_function(xv),model.decision_function(xv)):raise ValueError('Freezing changed predictions')
        model.leaf_ftol=plan.get('leaf_ftol',1e-12);model.leaf_gtol=plan.get('leaf_gtol',1e-8)
        model.indexed='dense';model.leaf_max_iter=plan.get('local_leaf_max_iter',400);model.split_max_iter=plan.get('split_max_iter',800)
        state=dict(phase='separation',history=[],stages={},stable=0,stage_epochs=0,global_chunks=0,training_seconds=0.,
            parent=dict(source_checkpoint=plan['source_checkpoint'],sha256=digest,
                        metrics=evaluate(model,x,y,xv,yv,metadata['class_names']),
                        selection_penalties=model.parent_penalties,training_converged=True))
        payload=dict(model=model,state=state,plan=plan,metadata=metadata,implementation_sha256=code,dataset_sha256=audit['dataset_sha256'])
        persist(prefix,payload)
    started=time.perf_counter()
    while state['phase']!='complete':
        if time.perf_counter()-started+state.get('step_estimate',60.)*1.5>wall_seconds:
            persist(prefix,payload);return 75
        tick=time.perf_counter();phase=state['phase']
        if phase=='separation':
            digest=route_digest(model,x)
            if digest!=state.get('certified_routes'):
                report=separation_check(model,x,y,time_limit=plan.get('separation_time_limit',120.))
                state['separation']=report
                if report['status']!='no_recession_direction':
                    state['phase']='separated' if report['status']=='separated' else 'inconclusive'
                    state['training_seconds']+=time.perf_counter()-tick;persist(prefix,payload)
                    return 78 if state['phase']=='separated' else 79
                state['certified_routes']=digest
            state['phase']='global';state['global_chunks']=0
        elif phase=='epoch':
            model.leaf_max_iter=plan.get('local_leaf_max_iter',400)
            if 'work' not in state:
                state['before_routes']=[t.leaf_indices(x,model.indexed) for t in model.trees]
                state['work']=begin_epoch(model,x,y)
            if step_epoch(model,state['work'],x,y):
                state['profile']=state.pop('work')['profile'];state['phase']='separation'
        elif phase=='global':
            before=model.objective(x,y);model.leaf_max_iter=plan.get('global_leaf_chunk_iter',400)
            stats=model.global_refit(x,y);after=model.objective(x,y);kkt=leaf_kkt_residual(model,x,y)
            if not np.isfinite(after) or after>before+1e-10:raise ValueError('Refit increased unpenalized CE')
            state['global_chunks']+=1
            # KKT is the numerical leaf certificate; optimizer success alone is not sufficient.
            stats['optimizer_converged']=stats['converged'];stats['converged']=bool(kkt<=plan.get('kkt_tol',1e-6))
            stats['leaf_kkt_residual']=kkt
            state['last_global']=stats
            if stats['converged']:
                if 'profile' not in state:state['phase']='epoch'
                else:
                    profile=state.pop('profile');profile['global_refit']=stats;profile['objective_after']=after
                    previous=state.pop('before_routes');changed=np.zeros(len(y),dtype=bool)
                    for t,old in zip(model.trees,previous):changed|=t.leaf_indices(x,model.indexed)!=old
                    check=convergence_check(profile,kkt,float(changed.mean()),state['stable'],
                        kkt_tol=plan.get('kkt_tol',1e-6),loss_rtol=plan.get('loss_rtol',1e-5),route_tol=plan.get('route_tol',1e-4),patience=3)
                    state['stable']=check['stable_epochs'];state['stage_epochs']+=1
                    row=dict(stage=model.support_mode,epoch=state['stage_epochs'],convergence=check,
                        metrics=evaluate(model,x,y,xv,yv,metadata['class_names']),
                        profile=profile,cumulative_training_seconds=state['training_seconds']+time.perf_counter()-tick)
                    state['history'].append(row);state['phase']='epoch'
                    if check['converged'] and state['stage_epochs']>=plan.get('minimum_epochs',5):
                        row['source_run_converged']=True;state['stages'][model.support_mode]=row
                        atomic_pickle('models/'+prefix+'_'+model.support_mode+'.pkl',dict(model=model,plan=plan,metadata=metadata,
                            dataset_sha256=audit['dataset_sha256'],support_sha256=model.support_sha256,
                            parent=state['parent'],metrics=row['metrics'],training_converged=True,convergence=check))
                        if model.support_mode==stages[-1]:state['phase']='complete'
                        else:
                            model=continue_full_refit(model);payload['model']=model
                            state.update(phase='separation',stage_epochs=0,stable=0,global_chunks=0)
                    elif state['stage_epochs']>=plan.get('max_epochs',200):state['phase']='stalled'
                    print(json.dumps(dict(stage=row['stage'],epoch=row['epoch'],phase=state['phase'],
                        training_cross_entropy=after,kkt=kkt,efficiency=row['metrics']['validation']['signal_efficiency_at_fixed_fpr'])),flush=True)
            elif state['global_chunks']>=plan.get('max_global_chunks',100):state['phase']='stalled'
        else:raise ValueError('Unknown refit phase: '+phase)
        model.assert_support();duration=time.perf_counter()-tick
        state['training_seconds']+=duration;state['step_estimate']=max(1.,duration)
        persist(prefix,payload)
        if state['phase']=='stalled':return 80
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--wall-seconds',type=float,default=2700.);parser.add_argument('--retry-separation',action='store_true')
    args=parser.parse_args();plan=json.loads(resolve(args.config).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+',plan['run_name']):raise ValueError('Use a simple unique run_name')
    lock=resolve('results/'+plan['run_name']+'_support.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Support-refit worker already running')
        return train(plan,args.wall_seconds,args.retry_separation)


if __name__=='__main__':sys.exit(main())
