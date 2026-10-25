"""Freeze verified research exports as an immutable FPGA bundle; promote after C++ checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,hashlib,json,shutil,subprocess
from datetime import datetime,timezone
import numpy as np
from tao.portable import cpp_header,predict_margin,operation_counts
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def freeze(root,config,output):
    cfg=read(root/config);out=root/output
    if out.exists():raise FileExistsError('Choose a new bundle version; existing directories are never overwritten')
    selection_path=root/cfg['research_selection'];selection=read(selection_path)
    assert sha(selection_path)==cfg['research_selection_sha256']
    old=root/'fpga_handoff/v2';old_manifest=read(old/'manifest.json')
    assert selection['dataset_sha256']==old_manifest['dataset_sha256']
    contract=read(old/'input_contract.json')
    assert sha(old/'input_contract.json')==old_manifest['files']['input_contract.json']
    out.mkdir(parents=True);write(out/'input_contract.json',contract)
    frozen=[];comparison=[]
    for entry in cfg['candidates']:
        item=selection['selections'][entry['research_role']];r=item['selection']['record']
        assert r['sha256']==entry['checkpoint_sha256']
        src=selection_path.parent/entry['research_role'];dest=out/entry['id'];dest.mkdir()
        shutil.copyfile(src/'validation_row_indices.tsv',dest/'validation_row_indices.tsv')
        for space in ['standardized','raw']:
            name=space+'_model.json';assert sha(src/name)==item['model_json_sha256'][name]
            spec=read(src/name);assert spec['features']==contract['features'] and spec['classes']==contract['classes']
            assert spec['preprocessing']['mean']==contract['mean'] and spec['preprocessing']['scale']==contract['scale']
            folder=dest/space;folder.mkdir();shutil.copyfile(src/name,folder/'model.json')
            (folder/'model_coefficients.h').write_text(cpp_header(spec))
            raw=space=='raw';vector='raw_validation_vectors.tsv' if raw else 'validation_vectors.tsv'
            expected='raw_expected_margin_and_leaves.tsv' if raw else 'expected_margin_and_leaves.tsv'
            x=np.loadtxt(src/vector,ndmin=2);e=np.loadtxt(src/expected,ndmin=2)
            margin,leaves,outputs=predict_margin(spec,x)
            assert np.array_equal(leaves,e[:,1:]) and np.allclose(margin,e[:,0],atol=1e-9,rtol=1e-10)
            shutil.copyfile(src/vector,folder/'vectors.tsv');shutil.copyfile(src/expected,folder/'expected.tsv')
            np.savetxt(folder/'tree_free_logits.tsv',outputs,fmt='%.17g')
            lo=np.asarray(contract['training_raw_min' if raw else 'training_standardized_min'])
            hi=np.asarray(contract['training_raw_max' if raw else 'training_standardized_max']);bounds=[]
            for tree in spec['trees']:
                w=np.asarray(tree['split_weights']);b=np.asarray(tree['split_bias']);p=np.asarray(tree['leaf_parameters'])
                bounds.append(dict(split_min=(b+np.minimum(w*lo,w*hi).sum(1)).tolist(),split_max=(b+np.maximum(w*lo,w*hi).sum(1)).tolist(),
                    leaf_min=(p[:,-1]+np.minimum(p[:,:-1]*lo,p[:,:-1]*hi).sum(1)).tolist(),leaf_max=(p[:,-1]+np.maximum(p[:,:-1]*lo,p[:,:-1]*hi).sum(1)).tolist()))
            write(folder/'ranges.json',dict(input_min=lo.tolist(),input_max=hi.tolist(),trees=bounds,
                signal_margin_min=-sum(max(t['leaf_max']) for t in bounds),signal_margin_max=-sum(min(t['leaf_min']) for t in bounds),
                note='Conservative bounds within observed training box, not physical input guarantees; no clipping.'))
        spec=read(dest/'standardized/model.json')
        record=dict(entry,run_name=item['selection']['case'].get('run_name',Path(r['source']).name.removesuffix('_best_efficiency.pkl')),
            checkpoint_path=r['immutable_snapshot'],original_checkpoint_path=r['source'],stage=r['stage'],metrics=r['metrics'],
            arithmetic=operation_counts(spec),sparsity=r['sparsity'],deterministic_working_point=spec['working_point'],
            raw_deterministic_working_point=read(dest/'raw/model.json')['working_point'],
            golden_vector_count=len(x),training_policy=entry['support_selection'],timing=r['timing'])
        write(dest/'candidate.json',record);frozen.append(record)
        comparison.append(dict(id=entry['id'],label=entry['label'],parameters=r['sparsity']['dense_parameter_slots'],nonzero_coefficients=r['sparsity']['nonzero_coefficients'],metrics=r['metrics'],timings=[r['timing']]))
    for name in ['model.json','vectors.tsv','expected_signal_probability.tsv']:
        rel='xgboost_reference/'+name;assert sha(old/rel)==old_manifest['files'][rel]
        (out/'xgboost_reference').mkdir(exist_ok=True);shutil.copyfile(old/rel,out/rel)
    primary=cfg['primary_candidate'];assert primary in [c['id'] for c in frozen]
    write(out/'comparison.json',comparison)
    lines=['# FPGA v3: Compact refit is the implementation default','',
        '| Role | Candidate | Trees x depth | Dense / nonzero | Efficiency at 1% background | AUC |', '|---|---|---|---:|---:|---:|']
    for c in frozen:
        n=c['sparsity'];lines.append(f"| {'DEFAULT' if c['id']==primary else 'Alternative'} | {c['id']} | {c['architecture']['trees']} x D{c['architecture']['depth']} linear | {n['dense_parameter_slots']} / {n['nonzero_coefficients']} | {c['metrics']['signal_efficiency_at_fixed_fpr']:.2%} | {c['metrics']['signal_auc']:.5f} |")
    lines+=['','XGBoost: 100 depth-4 constant-leaf trees; 3,099 scalar slots; efficiency 55.48%, AUC 0.92056.',
        '', 'Validation only; test untouched. Default is chosen for compression, not highest efficiency. Counts refer to standardized inputs. Selected snapshots can precede their converged source stage terminal. Floating point only; no quantization or synthesis claim.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    manifest=dict(schema='tree-benchmark.fpga-handoff.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dataset_sha256=selection['dataset_sha256'],partition='validation',test_evaluated=False,primary_metric='signal_efficiency_at_fixed_fpr',fixed_fpr=.01,
        research_selection=cfg['research_selection'],research_selection_sha256=cfg['research_selection_sha256'],primary_candidate=primary,
        selection_reason='Robert requests Compact refit for Dorian; alternatives retained for comparison.',candidates=frozen,xgboost=old_manifest['xgboost'],
        hardware=dict(quantized=False,synthesized=False,target_device=None,clock_target=None),files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    write(out/'manifest.json',manifest)

def promote(root,output):
    out=root/output;m=read(out/'manifest.json');report=read(out/'cpp_verification.json')
    for rel,digest in m['files'].items():assert sha(out/rel)==digest,'Bundle checksum mismatch: '+rel
    expected={(c['id'],space) for c in m['candidates'] for space in ['standardized','raw']}
    assert report['all_passed'] and {(c['candidate'],c['input_space']) for c in report['checks']}==expected
    c=next(c for c in m['candidates'] if c['id']==m['primary_candidate']);base=output+'/'+c['id']
    current=dict(schema='tree-benchmark.current-model.v1',as_of=datetime.now(timezone.utc).date().isoformat(),handoff_version=Path(output).name,
        candidate_id=c['id'],compact_candidate_id=c['id'],alternative_candidate_ids=[a['id'] for a in m['candidates'] if a['id']!=c['id']],
        manifest=output+'/manifest.json',manifest_sha256=sha(out/'manifest.json'),standardized_model=base+'/standardized/model.json',raw_model=base+'/raw/model.json',
        standardized_model_sha256=sha(root/(base+'/standardized/model.json')),raw_model_sha256=sha(root/(base+'/raw/model.json')),
        cpp_verification_sha256=sha(out/'cpp_verification.json'),checkpoint_sha256=c['checkpoint_sha256'],dataset_sha256=m['dataset_sha256'],
        run_name=c['run_name'],selected_continuation_epoch=c['selected_epoch'],training_run_converged=c['training_run_converged'],selected_checkpoint_converged=c['selected_checkpoint_converged'],
        architecture=dict(c['architecture'],stored_scalar_parameters=c['sparsity']['dense_parameter_slots'],nonzero_coefficients=c['sparsity']['nonzero_coefficients']),
        training=c['training_policy'],validation=dict(primary_metric=m['primary_metric'],fixed_fpr=.01,signal_efficiency=c['metrics']['signal_efficiency_at_fixed_fpr'],signal_auc=c['metrics']['signal_auc']),
        deterministic_validation_working_point=c['deterministic_working_point'],test_evaluated=False,update_policy='Compact refit is Dorian\'s default; preserve v1/v2/v3 and freeze a new version for any coefficient change.')
    write(root/'fpga_handoff/CURRENT_MODEL.json',current);write(root/('fpga_handoff/selections/'+Path(output).name+'_compact_refit.json'),current)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',default='configs/fpga_handoff_v3.json');p.add_argument('--output',default='fpga_handoff/v3');p.add_argument('--promote',action='store_true');a=p.parse_args()
    promote(ROOT,a.output) if a.promote else freeze(ROOT,a.selection,a.output)
