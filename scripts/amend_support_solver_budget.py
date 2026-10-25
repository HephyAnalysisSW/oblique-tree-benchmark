"""Audit a strictly larger auxiliary-split budget on a quiescent checkpoint.

This is an explicit checkpoint-policy migration, never an implicit resume override.
The original checkpoint is retained by SHA; coefficients and all stopping rules stay fixed.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse, copy, fcntl, hashlib, json, pickle
from datetime import datetime, timezone
import numpy as np
from data_utils import resolve
from scripts import train_tao_support_refit as worker


def coefficient_digest(model):
    h = hashlib.sha256()
    for tree in model.trees:
        for value in (tree.split_w, tree.split_b, tree.leaf_parameters):
            a = np.ascontiguousarray(value)
            h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def prepare(payload, old_plan, new_budget, reason):
    if payload['plan'] != old_plan:
        raise ValueError('Checkpoint and on-disk plan differ')
    old_budget = old_plan.get('split_max_iter', 800)
    if isinstance(new_budget, bool) or not isinstance(new_budget, int) or new_budget <= old_budget:
        raise ValueError('Only a strictly larger integer split iteration budget is allowed')
    if not reason.strip(): raise ValueError('An explicit measured reason is required')
    if payload['state']['phase'] in ('complete', 'separated', 'inconclusive', 'stalled'):
        raise ValueError('Only an unfinished non-held fit may be amended')
    model = payload['model']; model.assert_support()
    if model.split_max_iter != old_budget:
        raise ValueError('Model and plan split budgets differ')
    before = coefficient_digest(model); support = model.support_sha256
    updated = copy.deepcopy(payload)
    updated['plan']['split_max_iter'] = new_budget
    updated['model'].split_max_iter = new_budget
    updated['model'].assert_support()
    assert coefficient_digest(updated['model']) == before
    assert updated['model'].support_sha256 == support
    note = dict(updated_utc=datetime.now(timezone.utc).isoformat(), reason=reason,
                previous_split_max_iter=old_budget, split_max_iter=new_budget,
                coefficient_sha256=before, support_sha256=support,
                dataset_sha256=payload['dataset_sha256'],
                implementation_sha256=payload['implementation_sha256'],
                completed_history_rows=len(payload['state']['history']),
                phase=payload['state']['phase'],
                note='Additional auxiliary solver work only; objective, masks, coefficients, stopping tolerances and saved history unchanged.')
    updated['state'].setdefault('solver_budget_amendments', []).append(note)
    return updated, note


def apply(config, new_budget, reason):
    plan_path = resolve(config); old_plan = json.loads(plan_path.read_text())
    run = old_plan['run_name']
    scan = json.loads(resolve('configs/tao_support_refit_scan_v1.json').read_text())
    if scan['concurrency'] != 0:
        raise ValueError('Pause scan submissions (concurrency=0) before the explicit migration')
    with resolve('results/' + run + '_support.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Worker is active; wait for its clean checkpoint exit')
        latest = resolve('models/' + run + '_latest.pkl'); blob = latest.read_bytes()
        payload = pickle.loads(blob)
        expected = {path: worker.sha(path) for path in worker.IMPLEMENTATION}
        if payload['implementation_sha256'] != expected:
            raise ValueError('Implementation differs from the pinned checkpoint')
        updated, note = prepare(payload, old_plan, new_budget, reason)
        source_sha = hashlib.sha256(blob).hexdigest()
        archive = resolve('models/support_refit_policy_archive/' + source_sha + '.pkl')
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists(): archive.write_bytes(blob)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != source_sha:
            raise ValueError('Original checkpoint archive failed verification')
        if latest.read_bytes() != blob: raise ValueError('Checkpoint changed during migration')
        note.update(previous_checkpoint_sha256=source_sha,
                    previous_checkpoint=str(archive.relative_to(resolve('.'))),
                    previous_plan=old_plan, plan=updated['plan'])
        # A crash between these atomic writes fails closed at the worker's exact
        # plan check. The SHA-pinned old checkpoint and migration receipt recover it.
        receipt = 'results/' + run + '_budget_amendments.json'
        worker.atomic_json(receipt, updated['state']['solver_budget_amendments'])
        worker.atomic_pickle('models/' + run + '_latest.pkl', updated)
        worker.atomic_json(config, updated['plan'])
        worker.persist(run, updated)
        print(json.dumps(note), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--split-max-iter', type=int, required=True)
    parser.add_argument('--reason', required=True); args = parser.parse_args()
    apply(args.config, args.split_max_iter, args.reason)
