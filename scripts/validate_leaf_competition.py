"""Compare the frozen constant and linear candidates without vendor synthesis.

Stock conifer 1.11 handles the constant candidate. An independent sparse C++
reference supports both leaf types and supplies a matched CPU benchmark.
Fixed-point profiles are exploratory common formats, not tuned hardware designs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import conifer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpga.leaf_competition.models import normalized, conifer_constant, arithmetic
from tao.portable import predict_margin
from scripts.validate_xgboost_conifer import metrics, error, write_inputs
from sklearn.metrics import roc_curve

ROOT=Path(__file__).resolve().parents[1]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')



def check_cpp_fixtures(binary, out):
    """Check actual compiled routing, equality, tiny cuts and both leaf modes."""
    checks=[]
    x=np.array([[0.,2.],[1.,2.],[np.nextafter(1.,2.),2.],[2.,2.]])
    write_inputs(out/'fixture.bin',x)
    for mode in ['constant','linear']:
        for constant_cut in [False,True]:
            spec=dict(feature_count=2,features=['a','b'],tree_count=1,depth=1,internal_count=1,leaf_count=2,leaf_mode=mode,
                trees=[dict(split_weights=[[0.,0.]] if constant_cut else [[2.**-40,0.]],
                    split_bias=[np.nextafter(0.,1.)] if constant_cut else [-2.**-40],
                    leaf_parameters=[[1.],[3.]] if mode=='constant' else [[.5,0.,1.],[0.,2.,3.]])])
            write(out/'fixture.json',normalized(spec))
            for profile in ['double','f18','f12','f8']:
                subprocess.run([str(binary),profile,str(out/'fixture.json'),'-',str(out/'fixture.bin'),str(out/'fixture.out')],check=True,capture_output=True)
                score=np.fromfile(out/'fixture.out',dtype='<f8')
                if constant_cut:expected=[-3.]*4 if mode=='constant' else [-7.]*4
                elif profile=='double':expected=[-1.,-1.,-3.,-3.] if mode=='constant' else [-1.,-1.5,-7.,-7.]
                else:expected=[-1.,-1.,-1.,-3.] if mode=='constant' else [-1.,-1.5,-1.5,-7.]
                np.testing.assert_array_equal(score,expected)
                checks.append(dict(leaf_mode=mode,constant_cut=constant_cut,profile=profile,passed=True))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='artifacts/conifer/leaf_competition_v1')
    parser.add_argument('--max-events',type=int)
    args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    pointer=read(ROOT/'fpga_handoff/CURRENT_CANDIDATES.json');manifest=read(ROOT/pointer['manifest'])
    assert sha(ROOT/pointer['manifest'])==pointer['manifest_sha256']
    for rel,digest in manifest['files'].items():assert sha((ROOT/pointer['manifest']).parent/rel)==digest
    dataset=ROOT/'data/qg_vs_wzt_standardized.npz';assert sha(dataset)==manifest['dataset_sha256']
    contract=read(ROOT/'fpga_handoff/v3/input_contract.json')
    # Guard-bit bounds depend on training only; validation/test do not select ranges.
    train_abs=np.maximum(np.abs(contract['training_standardized_min']),np.abs(contract['training_standardized_max']))
    with np.load(dataset,allow_pickle=False) as data:x,y=data['X_val'].astype(float),data['y_val']
    if args.max_events:x,y=x[:args.max_events],y[:args.max_events]
    write_inputs(out/'validation.bin',x)
    base=Path(conifer.__file__).parent
    binary=out/'emulate'
    cmd=['g++','-O3','-std=c++17','-ffp-contract=off','-I'+os.environ['JSON_ROOT'],
         '-I'+os.environ['XILINX_AP_INCLUDE'],'-I'+str(base/'backends/cpp/include'),str(ROOT/'fpga/leaf_competition/emulate.cc'),'-o',str(binary)]
    print('Compiling shared sparse emulator and unmodified conifer reference',flush=True)
    build=subprocess.run(cmd,capture_output=True,text=True);(out/'compile.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,'See compile.log'
    fixture_checks=check_cpp_fixtures(binary,out)
    report=dict(cpp_fixture_checks=fixture_checks,complete=False,scope='smoke' if args.max_events else 'full_validation',events=len(x),test_evaluated=False,
        dataset_sha256=manifest['dataset_sha256'],manifest_sha256=pointer['manifest_sha256'],host=socket.gethostname(),
        conifer_version=importlib.metadata.version('conifer'),compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        source_hashes={p:sha(ROOT/p) for p in ['scripts/validate_leaf_competition.py','fpga/leaf_competition/emulate.cc','fpga/leaf_competition/models.py']},
        header_sha256=sha(base/'backends/cpp/include/conifer.h'),candidates=[],synthesis_run=False,
        limitations=['CPU timings are single-thread in-memory C++ path traversal, not FPGA latency.',
        'Stock conifer HLS evaluates all nodes; its resource counts differ from path traversal.',
        'Coefficient payload excludes topology, indices and padding; not BRAM/DSP/LUT utilization.',
        'Common fixed-point formats are exploratory and not independently minimized for either model.'])
    write(out/'report.json',report);curves=[]
    for candidate in manifest['candidates']:
        name=candidate['id'];folder=out/name;folder.mkdir(exist_ok=True)
        spec=read(ROOT/candidate['model']);assert sha(ROOT/candidate['model'])==candidate['model_sha256']
        norm=normalized(spec);write(folder/'normalized.json',norm)
        canonical,leaves,_=predict_margin(spec,x)
        scaled,scaled_leaves,_=predict_margin(norm,x)
        assert np.array_equal(leaves,scaled_leaves) and np.max(np.abs(canonical-scaled))<1e-9
        # Conservative absolute bounds cover every partial dot/score sum in the training box.
        cut_bound=0.
        score_bound=0.
        for tree in norm['trees']:
            w=np.asarray(tree['split_weights']);b=np.asarray(tree['split_bias']);p=np.asarray(tree['leaf_parameters'])
            cut_bound=max(cut_bound,float(np.max(np.abs(w)@train_abs)),float(np.max(np.abs(b))))
            score_bound+=float(np.max(np.abs(p[:,:-1])@train_abs+np.abs(p[:,-1]))) if spec['leaf_mode']=='linear' else float(np.max(np.abs(p)))
        assert cut_bound+16 < 512 and np.max(train_abs)<512 and score_bound+16<32768
        assert all(np.max(np.abs(t['leaf_parameters']))<32 for t in spec['trees'])
        assert all(np.max(np.abs(t['split_weights']))<32 for t in norm['trees'])
        stock='-'
        if spec['leaf_mode']=='constant':write(folder/'conifer.json',conifer_constant(spec));stock=str(folder/'conifer.json')
        entry=dict(id=name,leaf_mode=spec['leaf_mode'],model_sha256=candidate['model_sha256'],arithmetic=arithmetic(spec),
            source_validation=candidate['validation'],native_margin_metrics=metrics(y,canonical),profiles=[],
            training_box_bounds=dict(split_partial_abs_bound=cut_bound,forest_partial_abs_bound=score_bound),
            normalization='Positive power-of-two cut scaling, constrained to preserve every nonzero binary64 coefficient; canonical handover coefficients unchanged')
        for profile in ['double','f18','f12','f8']:
            print(name,profile,flush=True)
            destination=folder/(profile+'.bin')
            run=subprocess.run([str(binary),profile,str(folder/'normalized.json'),stock,str(out/'validation.bin'),str(destination)],capture_output=True,text=True)
            (folder/(profile+'.log')).write_text(run.stdout+run.stderr);assert run.returncode==0,'See '+str(folder/(profile+'.log'))
            record=json.loads(run.stdout);score=np.fromfile(destination,dtype='<f8')
            routes=np.fromfile(str(destination)+'.routes',dtype='<i4').reshape(len(x),spec['tree_count'])
            record.update(name=profile,metrics=metrics(y,score),native_margin_error=error(score,canonical),
                leaf_assignment_mismatches=int(np.count_nonzero(routes!=leaves)))
            if profile=='double':
                assert record['native_margin_error']['max_abs']<1e-9 and record['leaf_assignment_mismatches']==0
                if stock!='-':assert record['stock_conifer_max_error']<1e-9
                curves.append((name,score,record['median_microseconds_per_event']))
            else:
                fraction=int(profile[1:]);record['types']=dict(input=f'ap_fixed<{10+fraction},10,AP_RND_CONV,AP_SAT>',
                    weight=f'ap_fixed<{6+fraction},6,AP_RND_CONV,AP_SAT>',score=f'ap_fixed<{16+fraction},16,AP_RND_CONV,AP_SAT>')
                if stock!='-':assert record['stock_conifer_exact_mismatches']==0
            entry['profiles'].append(record);write(out/'report.json',report)
        if stock!='-':
            entry['hls_cpp_checks']=[]
            # Standard conifer HLS C++ array interface; no backend patches or vendor tools.
            for profile in ['double','f12']:
                hcfg=conifer.backends.xilinxhls.auto_config()
                types={k:'double' for k in ['input','weight','score']} if profile=='double' else next(r['types'] for r in entry['profiles'] if r['name']==profile)
                hcfg.update(ProjectName='constant_'+profile,OutputDir=str(folder/'hls'/profile),InputPrecision=types['input'],ThresholdPrecision=types['input'],WeightPrecision=types['weight'],ScorePrecision=types['score'])
                hls=conifer.model.make_model(read(folder/'conifer.json'),hcfg);hls.compile()
                indices=np.linspace(0,len(x)-1,min(512,len(x)),dtype=int)
                hscore=np.asarray(hls.decision_function(x[indices])).reshape(-1)
                reference=np.fromfile(folder/(profile+'.bin'),dtype='<f8')[indices]
                comparison=error(hscore,reference)
                assert comparison['max_abs']<1e-9 if profile=='double' else comparison['exact_mismatches']==0
                entry['hls_cpp_checks'].append(dict(profile=profile,events=len(indices),error=comparison,backend_modified=False))
        report['candidates'].append(entry);write(out/'report.json',report)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,score,timing in curves:
        fpr,tpr,_=roc_curve(y,score)
        for ax in axes:ax.plot(fpr,tpr,label=f'{name} ({timing:.3f} μs/event C++)',lw=1.3)
    # Frozen XGBoost ROC is a reference; CPU timings above compare the same sparse kernel only.
    import xgboost as xgb
    bdt=xgb.Booster();bdt.load_model(ROOT/'fpga_handoff/v3/xgboost_reference/model.json');bdt.set_param({'nthread':1})
    bm=bdt.inplace_predict(np.asarray(x,dtype=np.float32),predict_type='margin');report['xgboost_metrics']=metrics(y,bm)
    fpr,tpr,_=roc_curve(y,bm)
    for ax in axes:ax.plot(fpr,tpr,label='XGBoost reference',ls='--',color='black',lw=1.)
    axes[0].set(xlim=(0,1),ylim=(0,1),title='Validation ROC');axes[1].set_xscale('log');axes[1].set(xlim=(1e-4,.1),ylim=(0,1),title='Low background acceptance')
    for ax in axes:ax.set_xlabel('Background efficiency');ax.set_ylabel('Signal efficiency');ax.grid(alpha=.2);ax.axvline(.01,color='gray',ls=':')
    axes[1].legend(fontsize=7,loc='lower right');fig.tight_layout();fig.savefig(out/'roc.pdf');fig.savefig(out/'roc.png',dpi=170);plt.close(fig)
    report['complete']=True;write(out/'report.json',report);print('Competition emulation complete',flush=True)

if __name__=='__main__':main()
