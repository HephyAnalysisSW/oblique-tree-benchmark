"""Validation-only ten-model table with a common sparse C++ timing kernel.

One XGBoost reference, then three size/accuracy representatives per family.
This prepares a comparison; it does not change any handover model or training.
"""
import hashlib,json,os,pickle,shutil,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import xgboost as xgb
from tao.portable import export_model,predict_margin
from tao.axis import axis_parameter_counts,assert_axis
from scripts.validate_xgboost_conifer import metrics,write_inputs,error

OUT=ROOT/'artifacts/ten_model_comparison_v1'

def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def load_checkpoint(path,digest=None):
    path=ROOT/path
    if digest:assert sha(path)==digest
    blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
    archive=ROOT/'models/ten_model_comparison_v1_selected'/f'{digest}.pkl';archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():archive.write_bytes(blob)
    assert sha(archive)==digest
    return pickle.loads(blob),dict(checkpoint=str(archive.relative_to(ROOT)),checkpoint_sha256=digest)


def operations(spec):
    """Products after exact-zero/unit elimination, plus comparator bounds."""
    all_products=all_comparisons=0
    def products(values):return sum(v!=0 and abs(v)!=1 for v in values)
    path_products=path_comparisons=0
    if 'n_trees' in spec:
        return dict(path_multiplications=0,all_node_multiplications=0,path_comparisons=sum(depth(group[0],0) for group in spec['trees']),
            all_node_comparisons=sum(sum(f>=0 for f in group[0]['feature']) for group in spec['trees']))
    internal=spec['internal_count'];linear=spec['leaf_mode']=='linear'
    for tree in spec['trees']:
        def visit(node):
            nonlocal all_products,all_comparisons
            if node>=internal:
                count=products(tree['leaf_parameters'][node-internal][:-1]) if linear else 0
                all_products+=count;return count,0
            weights=tree['split_weights'][node]
            if not any(weights):return visit(2*node+1+int(tree['split_bias'][node]>0))
            count=products(weights);all_products+=count;all_comparisons+=1
            left,right=visit(2*node+1),visit(2*node+2)
            return count+max(left[0],right[0]),1+max(left[1],right[1])
        mul,cmp=visit(0);path_products+=mul;path_comparisons+=cmp
    return dict(path_multiplications=path_products,all_node_multiplications=all_products,path_comparisons=path_comparisons,all_node_comparisons=all_comparisons)


def depth(tree,node):
    return 0 if tree['feature'][node]<0 else 1+max(depth(tree,tree['children_left'][node]),depth(tree,tree['children_right'][node]))


def nnz(spec):
    return sum(sum(v!=0 for row in t['split_weights'] for v in row)+sum(v!=0 for v in t['split_bias'])+
        sum(v!=0 for row in t['leaf_parameters'] for v in row) for t in spec['trees'])


def fixture_checks(binary):
    # Validate unit-cut direction/equality and arbitrary weighted affine leaves.
    checks=[]
    x=np.array([[0.,2.],[1.,2.],[np.nextafter(1.,2.),2.],[2.,2.]])
    write_inputs(OUT/'fixture.bin',x)
    for sign in [1.,-1.]:
        for linear in [False,True]:
            spec=dict(schema='tree-benchmark.binary-hard-oblique.v1',feature_count=2,tree_count=1,depth=1,internal_count=1,leaf_count=2,leaf_mode='linear' if linear else 'constant',
                trees=[dict(split_weights=[[sign,0.]],split_bias=[-sign],leaf_parameters=[[.5,0.,1.],[0.,-1.,3.]] if linear else [[1.],[3.]])])
            write(OUT/'fixture.json',spec)
            subprocess.run([str(binary),str(OUT/'fixture.json'),str(OUT/'fixture.bin'),str(OUT/'fixture.out')],check=True,capture_output=True)
            expected,routes,_=predict_margin(spec,x)
            np.testing.assert_array_equal(np.fromfile(OUT/'fixture.out',dtype='<f8'),expected)
            np.testing.assert_array_equal(np.fromfile(str(OUT/'fixture.out')+'.routes',dtype='<i4'),routes[:,0]+1)
            checks.append(dict(sign=sign,linear=linear,passed=True))
    return checks


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    contract=read('fpga_handoff/v3/input_contract.json');audit=read('results/tao_split_audit.json')
    dataset=ROOT/'data/qg_vs_wzt_standardized.npz';assert sha(dataset)==audit['dataset_sha256']
    with np.load(dataset,allow_pickle=False) as data:x,y=data['X_val'].astype(float),data['y_val']
    assert len(x)==132800
    write_inputs(OUT/'validation.bin',x)
    binary=OUT/'benchmark'
    command=['g++','-std=c++17','-O3','-ffp-contract=off','-I'+os.environ['JSON_ROOT'],str(ROOT/'fpga/comparison/benchmark.cc'),'-o',str(binary)]
    subprocess.run(command,check=True)
    fixtures=fixture_checks(binary)
    candidates=[]
    axis=read('docs/axis_linear_v1/status.json');assert axis['complete'] and axis['converged_sources']==4
    for index in [3,1,0]:
        row=next(r for r in axis['models'] if r['case']['index']==index);selected=row['selected']
        assert selected['source_converged']
        payload,provenance=load_checkpoint(selected['checkpoint'],selected['sha256']);model=payload['model'];assert_axis(model)
        spec=export_model(model,contract['features'],contract['classes'],contract['mean'],contract['scale'])
        params=axis_parameter_counts(model)
        candidates.append(dict(id=f'axis_linear_d{model.depth}_n{model.ntrees}',family='Axis / linear',spec=spec,
            learned_nonzero_scalars=params['learned_nonzero_scalars'],stored_nonzero_scalars=nnz(spec),extra_cut_indices=params['feature_index_slots'],
            source_status='Regularized; refit held' if row['refit'].get('phase')=='separated' else ('Leaf refit converged' if selected['stage']=='leaf_refit' else 'Regularized; source converged'),
            expected_efficiency=selected['metrics']['signal_efficiency_at_fixed_fpr'],**provenance))
    constant=read('paper/sources/constant_leaf_study.json');assert constant['complete'] and constant['completed']==27
    for trees in [5,3,1]:
        row=next(r for r in constant['best_by_layout'] if r['case']['trees']==trees and r['case']['depth']==4)
        assert row['source_converged']
        payload,provenance=load_checkpoint(row['immutable_checkpoint'],row['checkpoint_sha256']);model=payload['model']
        spec=export_model(model,contract['features'],contract['classes'],contract['mean'],contract['scale'])
        candidates.append(dict(id=f'oblique_constant_d4_n{trees}',family='Oblique / constant',spec=spec,learned_nonzero_scalars=nnz(spec),stored_nonzero_scalars=nnz(spec),extra_cut_indices=0,
            source_status='Regularized; source converged',expected_efficiency=row['selected_metrics']['signal_efficiency_at_fixed_fpr'],**provenance))
    run='qg_vs_wzt_hardtao_linear_d4_n5_sa1e-05_la1e-05_seed42_hyper_scan_v1_c50'
    status=read('results/'+run+'_status.json');marker=read('results/tao_hyper_scan_v1_c50_member_0.json')
    assert status['converged'] and marker['complete'] and marker['converged']
    payload,provenance=load_checkpoint('models/'+run+'_best_efficiency.pkl');model=payload['model']
    spec=export_model(model,contract['features'],contract['classes'],contract['mean'],contract['scale'])
    candidates.append(dict(id='oblique_linear_performance_winner',family='Oblique / linear',spec=spec,learned_nonzero_scalars=nnz(spec),stored_nonzero_scalars=nnz(spec),extra_cut_indices=0,
        source_status='Regularized; source converged',expected_efficiency=payload['validation']['signal_efficiency_at_fixed_fpr'],**provenance))
    frozen=read('fpga_handoff/v3/manifest.json')
    for tag in ['highest_efficiency_3t_d4','compact_refit_3t_d4']:
        path=f'fpga_handoff/v3/{tag}/standardized/model.json';spec=read(path)
        assert sha(ROOT/path)==frozen['files'][f'{tag}/standardized/model.json']
        record=next(c for c in frozen['candidates'] if c['id']==tag)
        candidates.append(dict(id=tag,family='Oblique / linear',spec=spec,learned_nonzero_scalars=nnz(spec),stored_nonzero_scalars=nnz(spec),extra_cut_indices=0,
            source_status='Leaf refit converged' if tag.startswith('highest') else 'Full refit converged',
            expected_efficiency=record['metrics']['signal_efficiency_at_fixed_fpr'],source_model=path,model_sha256=sha(ROOT/path)))
    assert len(candidates)==9
    bdt=xgb.Booster();bdt.load_model(ROOT/'fpga_handoff/v3/xgboost_reference/model.json');bdt.set_param({'nthread':1})
    matrix=xgb.DMatrix(np.asarray(x,dtype=np.float32));native_xgb=bdt.predict(matrix,output_margin=True);native_leaves=bdt.predict(matrix,pred_leaf=True).astype(int)
    reference=read('fpga/conifer/reference/xgboost_v1/model.json')
    total=sum(len(g[0]['feature']) for g in reference['trees'])+len(reference['init_predict'])
    candidates.insert(0,dict(id='xgboost',family='XGBoost reference',spec=reference,learned_nonzero_scalars=total,stored_nonzero_scalars=total,extra_cut_indices=1499,source_status='Frozen reference'))
    report=dict(complete=False,created_utc=datetime.now(timezone.utc).isoformat(),dataset_sha256=audit['dataset_sha256'],events=len(x),test_evaluated=False,
        host=socket.gethostname(),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],cpp_fixture_checks=fixtures,rows=[],
        source_hashes={p:sha(ROOT/p) for p in ['scripts/compare_ten_models.py','fpga/comparison/benchmark.cc']},
        timing='Same binary64 C++ sparse traversal kernel; one thread; full batch; nine repeats after warmup; parsing, I/O and input conversion excluded. XGBoost inputs rounded to binary32 before timing.',
        products='Exact-zero and +/-1 factors removed. Other constant multiplier optimizations not assumed. Path values are worst-case bounds; all-node values include every structurally reachable node and leaf. No preprocessing or sigmoid.',
        selection='Winner plus two smaller representatives per family. Axis 3D3 is dominated in scalar-count/working-point space by axis 1D4. Oblique linear winner is the newer converged case50, not a change to the frozen handover.',
        synthesis_run=False)
    write(OUT/'report.json',report)
    for c in candidates:
        print('Benchmarking',c['id'],flush=True);spec=c.pop('spec');path=OUT/(c['id']+'.json');write(path,spec)
        run=subprocess.run([str(binary),str(path),str(OUT/'validation.bin'),str(OUT/(c['id']+'.bin'))],capture_output=True,text=True,check=True)
        timing=json.loads(run.stdout);scores=np.fromfile(OUT/(c['id']+'.bin'),dtype='<f8')
        routes=np.fromfile(OUT/(c['id']+'.bin.routes'),dtype='<i4').reshape(len(x),len(spec['trees']))
        if c['id']=='xgboost':
            canonical=native_xgb;assert np.array_equal(routes,native_leaves)
            assert np.max(np.abs(scores-canonical))<1e-5
            layout=dict(trees=reference['n_trees'],depth=reference['max_depth'])
        else:
            canonical,expected_leaves,_=predict_margin(spec,x)
            assert np.array_equal(routes,expected_leaves+spec['internal_count'])
            assert np.max(np.abs(scores-canonical))<1e-9
            layout=dict(trees=spec['tree_count'],depth=spec['depth'])
            # Cross-check against the saved shared probability-based metric too.
            from scipy.special import expit
            from evaluate import compute_metrics
            probability=expit(canonical);shared,_=compute_metrics(y,np.column_stack([1-probability,probability]).astype(np.float32),contract['classes'])
            assert abs(shared['signal_efficiency_at_fixed_fpr']-c['expected_efficiency'])<1e-7
        row=dict(c,layout=layout,metrics=metrics(y,scores),cpu=timing,arithmetic=operations(spec),model_sha256=sha(path),canonical_margin_error=error(scores,canonical),all_leaf_assignments_exact=True)
        report['rows'].append(row);write(OUT/'report.json',report)
    # Also provide the actual optimized XGBoost library throughput for context.
    xb=np.ascontiguousarray(x,dtype=np.float32);bdt.inplace_predict(xb);elapsed=[]
    for _ in range(9):
        start=time.perf_counter();bdt.inplace_predict(xb);elapsed.append(time.perf_counter()-start)
    report['native_xgboost_microseconds_per_event']=float(np.median(elapsed)*1e6/len(x))
    report['complete']=True;write(OUT/'report.json',report)
    lines=['| Model | Layout | Learned scalar nnz | Extra feature indices | Efficiency [%] | Realizable [%] | AUC | C++ µs/event | MUL path / all | Status |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in report['rows']:
        m=r['metrics'];a=r['arithmetic'];layout=r['layout']
        lines.append(f"| {r['family']} | {layout['trees']} × D{layout['depth']} | {r['learned_nonzero_scalars']} | {r['extra_cut_indices']} | {100*m['signal_efficiency_at_1pct_background']:.2f} | {100*m['strict_signal_efficiency']:.2f} | {m['auc']:.5f} | {r['cpu']['median_microseconds_per_event']:.3f} | {a['path_multiplications']} / {a['all_node_multiplications']} | {r['source_status']} |")
    (OUT/'table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
