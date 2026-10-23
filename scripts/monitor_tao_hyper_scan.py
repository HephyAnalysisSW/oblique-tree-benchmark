"""Rolling, concurrency-limited full-training scan with provisional diagnostics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,fcntl,json,subprocess,time
from datetime import datetime,timezone
import numpy as np
from data_utils import resolve,write_json
import common.syncer as syncer
import matplotlib.pyplot as plt


def read(path,default=None):
    path=resolve(path)
    return json.loads(path.read_text()) if path.exists() else default


def command(args):
    return subprocess.run(args,cwd=resolve('.'),text=True,capture_output=True,check=True).stdout.strip()


def save_state(path,state):
    temporary=path.with_suffix('.tmp');write_json(temporary,state);temporary.replace(path)


def collect(case):
    cfg=read(case['config']);layout=cfg['layouts'][0]
    run=(f"{cfg['task']}_hardtao_{cfg['leaf_mode']}_d{layout['depth']}_n{layout['trees']}"
         f"_sa{cfg['split_alpha']:g}_la{cfg['leaf_alpha']:g}_seed{cfg['seed']}_{cfg['run_suffix']}")
    status=read('results/'+run+'_status.json',dict(phase='not started',epochs=0,converged=False,training_seconds=0))
    history=read('results/'+run+'_history.json',[])
    marker=read(f"results/{cfg['scan_name']}_member_0.json",{})
    return dict(case,run_name=run,status=status,converged=bool(marker.get('complete') and marker.get('converged')),
        validation=history[-1]['validation'] if history else None,
        best_observed_efficiency=max((r['validation']['signal_efficiency_at_fixed_fpr'] for r in history),default=None),
        parameters=layout['trees']*((2**layout['depth']-1)*17+2**layout['depth']*(17 if cfg['leaf_mode']=='linear' else 1)))


def schedule(scan,state,rows,active):
    """Only resume explicit exit 75; failed/time-limited jobs require inspection."""
    active_ids={job['job_id'] for job in active};eligible=[];attention=[]
    for row in rows:
        key=str(row['index']);attempts=state['attempts'].get(key,[])
        if row['converged']:continue
        if not attempts:eligible.append(row);continue
        last=attempts[-1]
        if last['job_id'] in active_ids:continue
        ended=read(f"results/hyper_scan_v1/case_{row['index']:02d}_exit.json",{})
        if ended.get('job_id')==last['job_id'] and ended.get('exit_code')==75:
            eligible.append(row)
        else:attention.append(dict(index=row['index'],job_id=last['job_id'],reason='No active job or clean continuation exit; inspect Slurm/logs'))
    free=max(0,scan['concurrency']-len(active))
    eligible.sort(key=lambda row:(len(state['attempts'].get(str(row['index']),[])),row.get('priority',row['index']),row['index']))
    launched=[]
    for row in eligible[:free]:
        job=command(['sbatch','--parsable','scripts/train_tao_hyper_scan.slurm',str(row['index'])]).split(';')[0]
        state['attempts'].setdefault(str(row['index']),[]).append(dict(job_id=job,submitted_utc=datetime.now(timezone.utc).isoformat()))
        save_state(resolve('results/tao_hyper_scan_v1_scheduler.json'),state)
        launched.append(dict(index=row['index'],job_id=job))
        print('Submitted scan case',row['index'],'job',job,flush=True)
    return launched,attention


def publish(scan,rows,active,launched,attention,state):
    reference=read('plots/comparisons/full_training_summary.json')
    bdt=reference['models'][0]['metrics']['signal_efficiency_at_fixed_fpr']
    ranking=sorted([r for r in rows if r['validation']],key=lambda r:r['validation']['signal_efficiency_at_fixed_fpr'],reverse=True)
    terminal=[r for r in ranking if r['converged']]
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),scan=scan,models=rows,active=active,launched=launched,
        attention=attention,complete=sum(r['converged'] for r in rows),total=len(rows),primary_metric='signal_efficiency_at_fixed_fpr',
        fixed_fpr=.01,xgboost_efficiency=bdt,matched_xgboost=any(r['validation']['signal_efficiency_at_fixed_fpr']>=bdt for r in terminal),
        ranked_case_indices=[r['index'] for r in ranking],test_evaluated=False)
    root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    write_json(root/'hyper_scan_v1_summary.json',result)
    lines=['# Performance-first TAO scan','', 'Updated UTC: '+result['updated_utc'],'',
        f"Converged: {result['complete']}/{len(rows)}. XGBoost validation efficiency at 1% background: {100*bdt:.2f}%.",
        '','54 factorial cases plus four split/leaf penalty isolation cases. Same CART initialization seed 42, train/validation arrays and numerical stopping rule. Incomplete cases are provisional. Test data remain unused.',
        '', '| Case | Trees | Depth | Leaves | Split L1 | Leaf L1 | Parameters | Latest efficiency at 1% | Best observed | AUC | Passes | Training min | Status |',
        '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in ranking+[r for r in rows if not r['validation']]:
        m=row['validation'];s=row['status'];eff=f"{100*m['signal_efficiency_at_fixed_fpr']:.2f}%" if m else '—';auc=f"{m['signal_auc']:.5f}" if m else '—'
        best=f"{100*row['best_observed_efficiency']:.2f}%" if row['best_observed_efficiency'] is not None else '—'
        status='converged' if row['converged'] else s['phase']+' (provisional)'
        lines.append(f"| {row['index']} | {row['trees']} | {row['depth']} | {row['leaf_mode']} | {row['split_alpha']:g} | {row['leaf_alpha']:g} | {row['parameters']} | {eff} | {best} | {auc} | {s['epochs']} | {s['training_seconds']/60:.1f} | {status} |")
    if attention:lines+=['','Attention: '+json.dumps(attention)]
    lines+=['','Latest and best observed efficiencies are shown separately because cross-entropy improvements need not improve the 1% working point. Best-efficiency checkpoints are archived by the workers. Terminal-case ranking remains explicit in the final comparison.', '', 'Regularization multiplies norms in the **mean** cross-entropy objective. Zero L1 uses an unpenalized L-BFGS split surrogate because LIBLINEAR does not support this case. Constant leaves have no penalized leaf feature weights.',
        '', 'Efficiencies use the shared interpolated ROC convention. Constant-leaf scores can have coarse threshold steps.',
        '', 'Training time is measured compute wall time; this page does not estimate FPGA latency. Top-model CPU timings and ROC comparisons follow after the full scan converges.']
    (root/'hyper_scan_v1_report.md').write_text('\n'.join(lines)+'\n')
    for name in ['hyper_scan_v1_report.md','hyper_scan_v1_summary.json']:syncer.register_file(root/name)
    if ranking:
        fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        upper=max(.56,max(r['validation']['signal_efficiency_at_fixed_fpr'] for r in ranking))
        for mode,axs in zip(['linear','constant'],axes):
            for alpha,ax in zip([0.,1e-7,1e-5],axs):
                values=np.full((3,3),np.nan);labels={}
                for row in rows:
                    if row['kind']!='factorial' or row['leaf_mode']!=mode or row['split_alpha']!=alpha or not row['validation']:continue
                    i=[1,3,5].index(row['trees']);j=[2,3,4].index(row['depth']);v=row['validation']['signal_efficiency_at_fixed_fpr']
                    values[i,j]=100*v;labels[i,j]=f"{100*v:.2f}%{' *' if row['converged'] else ''}\n{row['status']['training_seconds']/60:.0f} min"
                im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=100*upper,cmap='viridis',aspect='auto')
                for (i,j),label in labels.items():ax.text(j,i,label,ha='center',va='center',color='white',fontsize=9)
                ax.set_title(f'{mode} leaves | split/leaf L1={alpha:g}');ax.set_xticks(range(3),[2,3,4]);ax.set_yticks(range(3),[1,3,5]);ax.set_xlabel('Depth');ax.set_ylabel('Trees')
        fig.suptitle(f'Signal efficiency at 1% background | XGBoost {100*bdt:.2f}%\n* converged; other cells provisional; time: training compute wall time')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Signal efficiency [%]',shrink=.8)
        fig.subplots_adjust(top=.86,bottom=.08,left=.06,right=.86,hspace=.35,wspace=.2)
        fig.savefig(root/'hyper_scan_v1_heatmap.png',dpi=170);plt.close(fig)
    syncer.sync()
    if result['complete']==len(rows) and not state.get('summary_job'):
        args=['--config',scan['cases'][0]['config'],'--output-prefix','hyper_scan_top','--top-n','8']
        for case in scan['cases'][1:]:args.extend(['--extra-config',case['config']])
        state['summary_job']=command(['sbatch','--parsable','scripts/summarize_tao_study.slurm']+args)
        save_state(resolve('results/tao_hyper_scan_v1_scheduler.json'),state)
    print(json.dumps(dict(updated_utc=result['updated_utc'],complete=result['complete'],total=len(rows),active=len(active),launched=launched,attention=attention)),flush=True)
    return result['complete']==len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    with resolve('results/tao_hyper_scan_v1_monitor.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Hyperparameter scan monitor already running')
        while True:
            try:
                scan=read('configs/tao_hyper_scan_v1.json');state=read('results/tao_hyper_scan_v1_scheduler.json',dict(attempts={}))
                queue=command(['squeue','-h','-u','robert.schoefbeck','-o','%i|%T|%j'])
                active=[dict(zip(['job_id','state','name'],line.split('|',2))) for line in queue.splitlines() if line and line.endswith('|tao-hyper-scan')]
                rows=[collect(case) for case in scan['cases']]
                launched,attention=schedule(scan,state,rows,active)
                finished=publish(scan,rows,active,launched,attention,state)
                if finished or args.once:return
            except Exception as exc:
                print(datetime.now(timezone.utc).isoformat(),type(exc).__name__,str(exc),flush=True)
                if args.once:raise
            time.sleep(120)


if __name__=='__main__':main()
