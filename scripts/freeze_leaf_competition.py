"""Freeze the completed constant winner beside an immutable linear-v3 pointer."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from tao.portable import export_model, predict_margin, deterministic_working_point, cpp_header
from fpga.leaf_competition.models import arithmetic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'fpga_handoff/leaf_competition_v1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p, v): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')


def main():
    study = read(ROOT/'paper/sources/constant_leaf_study.json')
    assert study['complete'] and study['completed'] == study['total'] == 27
    winner = study['winner']; source = ROOT/winner['immutable_checkpoint']
    assert sha(source) == winner['checkpoint_sha256'] and winner['source_converged']
    current = read(ROOT/'fpga_handoff/CURRENT_MODEL.json')
    contract = read(ROOT/'fpga_handoff/v3/input_contract.json')
    assert study['dataset_sha256'] == current['dataset_sha256']
    dataset = ROOT/'data/qg_vs_wzt_standardized.npz'; assert sha(dataset) == study['dataset_sha256']
    with source.open('rb') as stream: model = pickle.load(stream)['model']
    with np.load(dataset, allow_pickle=False) as data: x, y = data['X_val'].astype(float), data['y_val']
    if OUT.exists(): raise FileExistsError('Immutable competition bundle exists; choose a new version')
    OUT.mkdir(parents=True)
    specs = {space: export_model(model, contract['features'], contract['classes'], contract['mean'], contract['scale'], raw=space=='raw') for space in ['standardized','raw']}
    assert specs['standardized'] == read(ROOT/'paper/sources/constant_leaf_winner.json')
    margin, routes, _ = predict_margin(specs['standardized'], x)
    assert np.max(np.abs(margin+model.decision_function(x)[:,0])) < 1e-9
    working = deterministic_working_point(margin,y)
    indices = set(np.random.default_rng(42).choice(len(x),128,replace=False).tolist())
    indices.update(np.argsort(np.abs(margin-working['signal_margin_threshold']))[:32].tolist())
    for tree in model.trees:
        reaching=tree.routes(x,'dense')
        for node,rows in enumerate(reaching[:tree.internal]):
            if len(rows):
                distances=np.abs(x[rows]@tree.split_w[node]+tree.split_b[node])
                indices.update(rows[np.argsort(distances)[:4]].tolist())
    indices=np.asarray(sorted(indices),dtype=int)
    base=OUT/'constant_5t_d4';base.mkdir()
    np.savetxt(base/'validation_row_indices.tsv',indices,fmt='%d')
    np.savetxt(base/'labels.tsv',y[indices],fmt='%d')
    for space,spec in specs.items():
        inputs=x if space=='standardized' else x*np.asarray(contract['scale'])+np.asarray(contract['mean'])
        m,r,t=predict_margin(spec,inputs)
        assert np.array_equal(r,routes) and np.max(np.abs(m-margin))<1e-8
        spec['working_point']=deterministic_working_point(m,y)
        folder=base/space;folder.mkdir()
        write(folder/'model.json',spec);(folder/'model_coefficients.h').write_text(cpp_header(spec))
        np.savetxt(folder/'vectors.tsv',inputs[indices],fmt='%.17g')
        np.savetxt(folder/'expected.tsv',np.column_stack([m[indices],r[indices]]),fmt=['%.17g']+['%d']*model.ntrees)
    constant=dict(id='constant_5t_d4',role='constant_leaf_winner',model=str((base/'standardized/model.json').relative_to(ROOT)),
        model_sha256=sha(base/'standardized/model.json'),raw_model=str((base/'raw/model.json').relative_to(ROOT)),
        checkpoint_sha256=winner['checkpoint_sha256'],checkpoint=winner['immutable_checkpoint'],run_name=winner['run_name'],
        configuration=winner['configuration'],selected_epoch=winner['selected_epoch'],source_terminal_epoch=winner['source_terminal_epoch'],
        source_converged=True,selected_converged_at_save=winner['selected_converged_at_save'],validation=winner['selected_metrics'],
        working_point=working,arithmetic=arithmetic(specs['standardized']),conifer='Stock conifer 1.11 weighted-cut/scalar-leaf backend; adapter required, no backend changes')
    linear=dict(id='compact_refit_3t_d4',role='linear_leaf_winner',model=current['standardized_model'],
        model_sha256=current['standardized_model_sha256'],raw_model=current['raw_model'],checkpoint_sha256=current['checkpoint_sha256'],
        run_name=current['run_name'],configuration=current['training'],selected_epoch=current['selected_continuation_epoch'],
        source_converged=current['training_run_converged'],selected_converged_at_save=current['selected_checkpoint_converged'],
        validation=current['validation'],working_point=current['deterministic_validation_working_point'],
        arithmetic=arithmetic(read(ROOT/current['standardized_model'])),conifer='Requires a linear-leaf backend extension; v3 coefficients unchanged')
    write(base/'candidate.json',constant)
    manifest=dict(schema='tree-benchmark.leaf-competition.v1',as_of='2026-10-28',dataset_sha256=study['dataset_sha256'],
        partition='validation',test_evaluated=False,constant_scan=dict(completed=27,total=27),candidates=[constant,linear],
        selection='Best completed constant configuration and Robert-selected sparse linear Compact refit; not an identical-support leaf ablation',
        hardware=dict(synthesized=False,device=None,clock=None),
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    write(OUT/'manifest.json',manifest)
    write(ROOT/'fpga_handoff/CURRENT_CANDIDATES.json',dict(schema='tree-benchmark.current-candidates.v1',
        manifest=str((OUT/'manifest.json').relative_to(ROOT)),manifest_sha256=sha(OUT/'manifest.json'),
        linear_candidate='compact_refit_3t_d4',constant_candidate='constant_5t_d4',
        note='Two competing algorithms. CURRENT_MODEL.json retains the historical linear implementation default. Neither is a synthesized resource winner.'))
    print(json.dumps({'bundle':str(OUT),'golden_vectors':len(indices)}))

if __name__ == '__main__': main()
