"""Freeze the ten-model scientific comparison and standardized FPGA references."""
import hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np
from tao.portable import cpp_header,predict_margin,deterministic_working_point
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'fpga_handoff/comparison_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    source=ROOT/'artifacts/ten_model_comparison_v1';report=read(source/'report.json')
    assert report['complete'] and len(report['rows'])==10 and not report['test_evaluated']
    for path,digest in report['source_hashes'].items():assert sha(ROOT/path)==digest
    dataset=ROOT/'data/qg_vs_wzt_standardized.npz';assert sha(dataset)==report['dataset_sha256']
    if OUT.exists():raise FileExistsError('Frozen bundle exists; use a new version')
    with np.load(dataset,allow_pickle=False) as data:x,y=data['X_val'].astype(float),data['y_val']
    contract=read(ROOT/'fpga_handoff/v3/input_contract.json');OUT.mkdir(parents=True)
    write(OUT/'input_contract.json',contract)
    records=[];axis=read(ROOT/'docs/axis_linear_v1/status.json');constant=read(ROOT/'paper/sources/constant_leaf_study.json')
    rng=np.random.default_rng(42)
    for row in report['rows']:
        ident=row['id'];spec=read(source/(ident+'.json'));assert sha(source/(ident+'.json'))==row['model_sha256']
        base=OUT/ident;base.mkdir();is_xgb=ident=='xgboost'
        score=np.fromfile(source/(ident+'.bin'),dtype='<f8')
        nodes=np.fromfile(source/(ident+'.bin.routes'),dtype='<i4').reshape(len(x),row['layout']['trees'])
        indices=set(rng.choice(len(x),128,replace=False).tolist())
        indices.update(np.argsort(np.abs(score-row['metrics']['strict_threshold']))[:32].tolist())
        # Include representatives of every reached leaf in every tree.
        for tree in range(nodes.shape[1]):
            _,first=np.unique(nodes[:,tree],return_index=True);indices.update(first.tolist())
        indices=np.asarray(sorted(indices),dtype=int)
        np.savetxt(base/'validation_row_indices.tsv',indices,fmt='%d');np.savetxt(base/'labels.tsv',y[indices],fmt='%d')
        record=dict(row,input_space='standardized',role='representative',training_configuration=None,working_point_calibration='common_cpp_binary64_feature_order_reference')
        if is_xgb:
            write(base/'conifer_model.json',spec)
            shutil.copyfile(ROOT/'fpga_handoff/v3/xgboost_reference/model.json',base/'native_model.json')
            folder=base
            record.update(kind='xgboost',model=str((base/'conifer_model.json').relative_to(ROOT)),
                native_model=str((base/'native_model.json').relative_to(ROOT)),role='reference',training_config='configs/bdt_qg_vs_wzt_seed42.yaml')
        else:
            canonical,routes,_=predict_margin(spec,x)
            assert np.array_equal(routes,nodes-spec['internal_count']) and np.max(np.abs(canonical-score))<1e-9
            # Preserve the benchmark coefficients byte-for-byte in value; add the usable margin threshold.
            spec['working_point']=dict(signal_margin_threshold=row['metrics']['strict_threshold'],comparison='>',
                background_count=52511,maximum_false_positives=525,false_positives=row['metrics']['accepted_background'],
                achieved_background_efficiency=row['metrics']['strict_background_efficiency'],signal_efficiency=row['metrics']['strict_signal_efficiency'])
            spec['working_point']['calibration_inference']='common_cpp_binary64_feature_order_reference'
            numpy_working=deterministic_working_point(canonical,y)
            numpy_working['calibration_inference']='tao.portable.predict_margin_numpy_binary64'
            assert numpy_working['false_positives']<=525
            spec['working_points']=dict(comparison_cpp=dict(spec['working_point']),portable_numpy=numpy_working)
            folder=base/'standardized';folder.mkdir();write(folder/'model.json',spec)
            (folder/'model_coefficients.h').write_text(cpp_header(spec))
            record.update(kind='tao',model=str((folder/'model.json').relative_to(ROOT)))
            if row.get('checkpoint'):
                p=ROOT/row['checkpoint'];assert sha(p)==row['checkpoint_sha256'];payload=pickle.loads(p.read_bytes())
                record['training_configuration']=payload.get('config',payload.get('plan'))
                record['selected_epoch']=payload.get('epoch')
            if row['family']=='Axis / linear':
                a=next(r for r in axis['models'] if r['selected'] and r['selected']['sha256']==row['checkpoint_sha256'])
                record.update(training_config=a['case']['config'],refit_config=a['case']['refit_config'],
                    source_terminal_epoch=a['training']['epochs'],refit_outcome=a['refit']['phase'])
            elif row['family']=='Oblique / constant':
                c=next(r for r in constant['best_by_layout'] if r['checkpoint_sha256']==row['checkpoint_sha256'])
                record.update(training_config=c['case']['config'],source_terminal_epoch=c['source_terminal_epoch'],selected_epoch=c['selected_epoch'])
            elif ident=='oblique_linear_performance_winner':
                status=read(ROOT/'results/qg_vs_wzt_hardtao_linear_d4_n5_sa1e-05_la1e-05_seed42_hyper_scan_v1_c50_status.json')
                assert status['converged'];record.update(training_config='configs/hyper_scan_v1/case_50.json',source_terminal_epoch=status['epochs'])
            else:
                role='highest_efficiency_3t_d4' if ident.startswith('highest') else 'compact_refit_3t_d4'
                legacy=read(ROOT/f'fpga_handoff/v3/{role}/candidate.json')
                record['legacy_candidate']=legacy
                record['training_config']='configs/support_refit_scan_v1/leaf3.json' if ident.startswith('highest') else 'configs/support_refit_scan_v1/split300.json'
                record['selected_epoch']=legacy.get('selected_epoch')
            margin,native_routes,_=predict_margin(spec,x[indices])
            assert np.array_equal(native_routes,nodes[indices]-spec['internal_count']) and np.max(np.abs(margin-score[indices]))<1e-9
        np.savetxt(folder/'vectors.tsv',x[indices],fmt='%.17g')
        expected_routes=nodes[indices] if is_xgb else nodes[indices]-spec['internal_count']
        np.savetxt(folder/'expected.tsv',np.column_stack([score[indices],expected_routes]),fmt=['%.17g']+['%d']*row['layout']['trees'])
        record.update(model_sha256=sha(ROOT/record['model']),benchmark_model_sha256=row['model_sha256'],golden_vector_count=len(indices),
            source_stage_converged=None if is_xgb else True,training_complete=True,
            convergence_scope='fixed 100 boosting rounds; no TAO stationarity test' if is_xgb else 'numerical convergence of selected TAO source stage',selection_rule='validation efficiency at 1% background, AUC secondary; earlier snapshots may precede a converged source terminal')
        if ident in ['axis_linear_d4_n3','oblique_constant_d4_n5','oblique_linear_performance_winner']:record['role']='family_performance_winner'
        if ident=='compact_refit_3t_d4':record['role']='compressed_oblique_linear_reference'
        write(base/'candidate.json',record);records.append(record)
    manifest=dict(schema='tree-benchmark.final-comparison.v1',as_of='2026-10-29',dataset_sha256=report['dataset_sha256'],partition='validation',
        test_evaluated=False,benchmark_report_sha256=sha(source/'report.json'),candidates=records,
        input_contract='standardized only in this bundle; use supplied training scaler; raw variants remain available in historical v3 for its original candidates',
        numeric='binary64 TAO coefficients and inference; XGBoost thresholds/leaves reconcile native binary32 values and inputs',
        hardware=dict(quantized=False,synthesized=False,device=None,clock=None),
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    write(OUT/'manifest.json',manifest)
    docs=ROOT/'docs/ten_model_comparison_v1';docs.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source/'report.json',docs/'validation.json');shutil.copyfile(source/'table.md',docs/'table.md')
    shutil.copyfile(source/'report.json',ROOT/'paper/sources/ten_model_comparison.json')
    print('Frozen',len(records),'candidates in',OUT)

if __name__=='__main__':main()
