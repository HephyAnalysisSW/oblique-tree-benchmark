"""Verify TAO uses the reference BDT's immutable exported partitions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import hashlib,json,pickle,zipfile
import numpy as np
from data_utils import resolve,load_config,write_json

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    reference=load_config('configs/bdt_qg_vs_wzt_seed42.yaml')
    with resolve(reference['model']['output_model_path']).open('rb') as f:bdt=pickle.load(f)
    original_split=bdt['config']['split']
    assert original_split==dict(train=.64,validation=.16,test=.2,seed=42)
    dataset=resolve(reference['data']['dataset_path']).resolve()
    original=Path('/users/dorian.sloot/tree_benchmark')/reference['data']['dataset_path']
    file_hash=digest(dataset)
    assert original.is_file() and digest(original)==file_hash, 'Dataset differs from Dorian export'
    partitions={}
    # Hash the uncompressed NPY members, including dtype/shape and row order.
    # Test members are read only for integrity, never for fitting or metrics.
    with zipfile.ZipFile(dataset) as z:
        for split in ['train','val','test']:
            members={}
            for key in ['X_'+split,'y_'+split]:
                with z.open(key+'.npy') as f:
                    version=np.lib.format.read_magic(f)
                    shape,fortran,dtype=np.lib.format._read_array_header(f,version)
                with z.open(key+'.npy') as f:
                    h=hashlib.sha256()
                    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
                members[key]=dict(shape=list(shape),dtype=str(dtype),fortran_order=fortran,sha256_npy=h.hexdigest())
            assert members['X_'+split]['shape'][0]==members['y_'+split]['shape'][0]
            partitions[split]=members
    checks=[]
    paths=[resolve('configs/bdt_qg_vs_wzt_seed42_retrained.yaml')]
    paths+=sorted(resolve('configs').glob('*layout_scan_v1*.yaml'))
    assert len(paths)==11
    for path in paths:
        cfg=load_config(path)
        with resolve(cfg['model']['output_model_path']).open('rb') as f:payload=pickle.load(f)
        assert cfg['data']==reference['data'] and payload['config']['data']==reference['data']
        assert resolve(payload['config']['data']['dataset_path']).resolve()==dataset
        assert cfg['seed']==reference['seed']==42
        model=payload['model']
        assert model.n_features_in_==partitions['train']['X_train']['shape'][1]
        if 'tao' in cfg['model']:
            assert np.all(model.transform.mean==0) and np.all(model.transform.scale==1)
            assert np.array_equal(model.classes_,np.arange(2))
        checks.append(dict(run=cfg['run_name'],dataset_path=str(dataset),same_exported_partitions=True,
                           model_sha256=digest(resolve(cfg['model']['output_model_path']))))
    result=dict(reference_run=reference['run_name'],reference_model_sha256=digest(resolve(reference['model']['output_model_path'])),
        dataset_path=str(dataset),dataset_sha256=file_hash,original_export_path=str(original),original_export_identical=True,
        partitions=partitions,checked_models=checks,same_partitions=True,original_bdt_split=original_split,
        training_code='Both trainers consume X_train/y_train and X_val/y_val directly from this NPZ; TAO performs no new split or scaling.',
        test_used_for_training_or_model_selection=False,
        scope='Verifies exact current exported arrays and model/config lineage. Historical BDT checkpoints did not embed array hashes.')
    write_json('results/tao_split_audit.json',result)
    print(json.dumps(dict(same_partitions=True,dataset_sha256=file_hash,
        rows={s:partitions[s]['X_'+s]['shape'][0] for s in partitions},checked_models=len(checks))),flush=True)

if __name__=='__main__':main()
