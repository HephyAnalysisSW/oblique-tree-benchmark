"""Archive evaluated refit coefficients before subsequent node updates alter them."""
import copy,json
from data_utils import resolve


def install_archiver(worker,plan):
    run=plan['run_name'];latest=resolve('models/'+run+'_latest.pkl');seen=set()
    if latest.exists():
        import pickle
        payload=pickle.loads(latest.read_bytes())
        seen={(r['stage'],r['epoch']) for r in payload['state']['history']}
    original=worker.persist
    def persist(prefix,payload):
        original(prefix,payload)
        history=payload['state']['history']
        if not history:return
        row=history[-1];key=(row['stage'],row['epoch'])
        if key in seen:return
        # This hook sees the new row at the worker's immediate post-evaluation
        # save, before another node update. Old history on resume is excluded.
        if 'work' in payload['state']:raise ValueError('Evaluated snapshot captured during partial epoch')
        model=copy.deepcopy(payload['model']);model.support_mode=row['stage'];model.assert_support()
        metric=row['metrics']['validation'];score=(metric['signal_efficiency_at_fixed_fpr'],metric['signal_auc'])
        directory=resolve('results/'+prefix+'_archived_best.json')
        book=json.loads(directory.read_text()) if directory.exists() else {}
        old=book.get(row['stage'])
        if old is None or score>tuple(old['score']):
            snapshot=dict(model=model,plan=plan,metadata=payload['metadata'],dataset_sha256=payload['dataset_sha256'],
                parent=payload['state']['parent'],metrics=row['metrics'],stage=row['stage'],epoch=row['epoch'],
                convergence=row['convergence'],selected_by='validation_efficiency_at_1pct_background_then_auc',
                support_sha256=model.support_sha256,training_converged_at_save=bool(row.get('source_run_converged')))
            path='models/'+prefix+'_'+row['stage']+'_best_efficiency.pkl'
            worker.atomic_pickle(path,snapshot)
            import hashlib
            book[row['stage']]=dict(score=score,epoch=row['epoch'],path=path,sha256=hashlib.sha256(resolve(path).read_bytes()).hexdigest())
            worker.atomic_json(directory,book)
        seen.add(key)
    worker.persist=persist
