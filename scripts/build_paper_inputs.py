"""Generate manuscript tables, actual tree equations, provenance, and optional ROC.

Text assets need the checked-in model/result snapshots; --with-roc additionally
needs the CLIP validation NPZ and immutable saved model pickles. Never reads test arrays.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from tao.printing import TreePrinter

PAPER=ROOT/'paper'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def pct(v):return f'{100*v:.2f}'
def metric(r):return r['metrics']['signal_efficiency_at_fixed_fpr']

def snapshot():
    source=ROOT/'plots/comparisons/support_refit_scan_v1_final_summary.json'
    target=PAPER/'sources/results_snapshot.json'
    if not source.exists():return read(target)
    s=read(source);manifest=read(ROOT/'fpga_handoff/v3/manifest.json')
    assert s['dataset_sha256']==manifest['dataset_sha256']
    for entry in manifest['candidates']:
        assert s['selections'][entry['research_role']]['record']['sha256']==entry['checkpoint_sha256'], 'Research selection differs from frozen v3; do not silently relabel new models as v3'
    keep={k:s[k] for k in ['updated_utc','dataset_sha256','test_evaluated','xgboost_metrics','xgboost_parameters','xgboost_timing','uncertainty','selections','models']}
    keep['source_result_sha256']=sha(source)
    keep['research_status']=read(ROOT/'docs/handover_status_2026-10-26.json')
    keep['fpga_manifest']=manifest
    # Preserve enough frozen data to rebuild manuscript tables without /models or /data.
    write(target,json.dumps(keep,indent=2)+'\n')
    return keep

def latex_expression(printer,r,precision=8):
    terms=[]
    for w,name in zip(r['weights'],printer.features):
        if w!=0:terms.append((w,r'\mathtt{'+esc(name)+'}'))
    if r['bias']!=0:terms.append((r['bias'],''))
    out=[]
    for i,(v,name) in enumerate(terms):
        n=format(abs(v),f'.{precision}g')
        if 'e' in n:
            a,b=n.split('e');n=a+r'\!\times\!10^{'+str(int(b))+'}'
        term=n+(r'\,'+name if name else '')
        sign=('-' if v<0 else '') if i==0 else (' - ' if v<0 else ' + ')
        out.append(sign+term)
    return out or ['0']

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--with-roc',action='store_true');args=parser.parse_args()
    for folder in ['generated','figures','sources']:(PAPER/folder).mkdir(parents=True,exist_ok=True)
    s=snapshot();source=ROOT/'fpga_handoff/v3/compact_refit_3t_d4/standardized/model.json'
    manifest=s['fpga_manifest'];expected=manifest['files']['compact_refit_3t_d4/standardized/model.json']
    assert sha(source)==expected
    spec=read(source);write(PAPER/'sources/compact_standardized_model.json',json.dumps(spec,indent=2)+'\n')
    printer=TreePrinter(spec,17);write(PAPER/'generated/compact_forest_full_precision.txt',printer.format_forest())
    display=TreePrinter(spec,9)
    for t in range(3):write(PAPER/f'generated/compact_tree_{t}.txt',display.format_tree(t))
    snippets=[]
    for t,n in [(0,0),(0,1),(0,2),(1,1),(2,0)]:
        r=next(r for r in printer.records(t) if r['kind']=='split' and r['index']==n)
        terms=latex_expression(printer,r,8)
        chunks=[''.join(terms[i:i+2]) for i in range(0,len(terms),2)]
        label=f'h_{{{t},{n}}}(x)'
        snippets += [r'\begin{equation}\begin{aligned}',label+r' &\simeq '+chunks[0]+(r'\\' if len(chunks)>1 else '')]
        for i,c in enumerate(chunks[1:]):snippets.append(r'&\quad '+c+(r'\\' if i<len(chunks)-2 else ''))
        snippets.append(r'\end{aligned}\end{equation}')
    write(PAPER/'generated/example_cuts.tex','\n'.join(snippets)+'\n')
    rows=[]
    # Explicit priority: compact, higher efficiency, one-tree. XGBoost is a reference.
    roles=[('Compact refit','compact_match_xgb'),('Higher efficiency','winner'),('One-tree reference','compact_path_match_xgb')]
    for label,tag in roles:
        c=s['selections'][tag]['case'];r=s['selections'][tag]['record'];n=r['sparsity']
        rows.append(f"{label} & {c['trees']} $\\times$ D{c['depth']} & {n['dense_parameter_slots']} & {n['nonzero_coefficients']} & {pct(metric(r))} & {r['metrics']['signal_auc']:.5f} & {r['timing']['median_microseconds_per_event']:.3f} \\\\")
    rows.append(f"XGBoost & 100 $\\times$ D4 & {s['xgboost_parameters']} & --- & {pct(s['xgboost_metrics']['signal_efficiency_at_fixed_fpr'])} & {s['xgboost_metrics']['signal_auc']:.5f} & {s['xgboost_timing']['median_microseconds_per_event']:.3f} \\\\")
    write(PAPER/'generated/main_results.tex','\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    rows=[]
    for label,tag in roles:
        r=s['selections'][tag]['record'];n=r['sparsity']
        rows.append(f"{label} & {n['reachable_nonzero_coefficients']} & {n['selected_path_multiplications_upper_bound']} & {n['selected_path_comparisons_upper_bound']} \\\\")
    write(PAPER/'generated/arithmetic.tex','\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    rows=[]
    for row in s['models']:
        stages=row['stages'];p=row['selection_penalties'];case=row['case']
        def eff(key):return pct(metric(stages[key])) if key in stages else '---'
        best='full_best' if 'full_best' in stages else 'full';value=eff(best)
        if row['phase']!='complete' and value!='---':value+=r'$^{\dagger}$'
        penalty=f"{p['split_alpha']:g}/{p['leaf_alpha']:g}".replace('e-0','e-')
        status={'complete':'complete','separated':'separated','inconclusive':'LP timeout'}.get(row['phase'],row['phase'])
        name=esc(case['id'])+(r'$^{*}$' if case['trees']==1 else '')
        rows.append(f"{name} & \\texttt{{{penalty}}} & {case['parent_nonzeros']} & {eff('parent')} & {eff('leaf_only')} & {eff('full')} & {value} & {status} \\\\")
    write(PAPER/'generated/sfr_results.tex','\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    rows=[]
    for label,tag in roles:
        u=s['uncertainty']['models'][tag];a,b=u['difference_interval95']
        rows.append(f"{label} & {100*u['difference_vs_reference']:+.2f} & [{100*a:+.2f}, {100*b:+.2f}] \\\\")
    write(PAPER/'generated/uncertainty.tex','\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    featurelines=[f"{i} & \\texttt{{{esc(name)}}} \\\\" for i,name in enumerate(printer.features)]
    write(PAPER/'generated/features.tex','\n'.join(featurelines)+'\n'+r'\bottomrule'+'\n')
    if args.with_roc:make_figures(s)
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=['src/tao/hard.py','src/tao/support.py','src/tao/split_search.py','src/tao/epoch.py','src/tao/convergence.py','src/tao/initialization.py','src/tao/printing.py','scripts/train_tao_support_refit.py','scripts/build_paper_inputs.py','configs/support_refit_scan_v1/split300.json','configs/bdt_qg_vs_wzt_seed42.yaml','requirements.txt']
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),repository_base_commit=base,dataset_sha256=s['dataset_sha256'],model_json_sha256=expected,
        selected_checkpoint_sha256=s['selections']['compact_match_xgb']['record']['sha256'],fpga_manifest_sha256=sha(ROOT/'fpga_handoff/v3/manifest.json'),
        source_hashes={f:sha(ROOT/f) for f in files},generated_assets={str(p.relative_to(PAPER)):sha(p) for p in sorted((PAPER/'generated').glob('*')) if p.is_file() and p.name!='provenance.tex'},figure_assets={str(p.relative_to(PAPER)):sha(p) for p in sorted((PAPER/'figures').glob('*.pdf'))},test_evaluated=False)
    write(PAPER/'sources/provenance.json',json.dumps(provenance,indent=2)+'\n')
    write(PAPER/'generated/provenance.tex',r'\newcommand{\SourceCommit}{'+base+'}\n'+r'\newcommand{\DatasetHash}{'+s['dataset_sha256']+'}\n'+r'\newcommand{\CompactCheckpointHash}{'+provenance['selected_checkpoint_sha256']+'}\n'+r'\newcommand{\CompactModelHash}{'+expected+'}\n')
    print('Manuscript inputs generated; all 93 nodes/leaves printed; model JSON SHA',expected)

def make_figures(s):
    import pickle
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from evaluate import compute_metrics
    dataset=ROOT/'data/qg_vs_wzt_standardized.npz';assert sha(dataset)==s['dataset_sha256']
    with np.load(dataset) as d:x,y=d['X_val'].astype(float),d['y_val']
    cases=[]
    for label,tag,color in [('Compact refit (3 x D4)','compact_match_xgb','#d76a15'),('Higher efficiency (3 x D4)','winner','#2166ac'),('One-tree reference (1 x D4)','compact_path_match_xgb','#31834b')]:
        r=s['selections'][tag]['record'];p=ROOT/r['immutable_snapshot'];assert sha(p)==r['sha256']
        model=pickle.loads(p.read_bytes())['model'];prob=model.predict_proba(x)
        m=compute_metrics(y,prob,['quark/gluon','W/Z/top'])[0];assert abs(m['signal_efficiency_at_fixed_fpr']-metric(r))<1e-7
        cases.append((label,color,prob[:,1],m))
    reference=ROOT/'models/qg_vs_wzt_xgboost_bdt_seed42.pkl'
    assert sha(reference)==s['fpga_manifest']['xgboost']['source_checkpoint_sha256']
    model=pickle.loads(reference.read_bytes())['model'];model.get_booster().set_param({'nthread':1})
    prob=model.predict_proba(x);m=compute_metrics(y,prob,['quark/gluon','W/Z/top'])[0]
    assert abs(m['signal_efficiency_at_fixed_fpr']-s['xgboost_metrics']['signal_efficiency_at_fixed_fpr'])<1e-7
    cases.append(('XGBoost (100 x D4)','#222222',prob[:,1],m))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.7,3.1))
    for label,color,score,m in cases:
        fpr,tpr,_=roc_curve(y,score)
        for ax in axes:ax.plot(fpr,tpr,label=label,color=color,lw=1.25);ax.plot(.01,m['signal_efficiency_at_fixed_fpr'],marker='o',ms=3,color=color)
    axes[0].set(xlim=(0,1),ylim=(0,1),title='Full ROC')
    axes[1].set(xscale='log',xlim=(1e-4,.1),ylim=(.05,.9),title='Low background acceptance')
    for ax in axes:ax.set_xlabel('Background efficiency');ax.set_ylabel('Signal efficiency');ax.grid(alpha=.2);ax.axvline(.01,color='#888888',ls=':',lw=.8)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.16,1,1));fig.savefig(PAPER/'figures/roc.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.7,2.9))
    for row in s['models']:
        n=row['case']['parent_nonzeros'];p=metric(row['stages']['parent'])
        ax.scatter(n,100*p,marker='o',facecolors='none',edgecolors='#888888',s=28)
        eligible=[r for tag,r in row['stages'].items() if tag.startswith('leaf_only') or (row['phase']=='complete' and tag.startswith('full'))]
        best=max(metric(r) for r in eligible)
        ax.plot([n,n],[100*p,100*best],color='#aebfce',lw=.9)
        ax.scatter(n,100*best,color='#2166ac',s=26)
        if row['case']['id'] in ['split300','both10','leaf3','both100']:ax.annotate(row['case']['id'],(n,100*best),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.scatter([],[],marker='o',facecolors='none',edgecolors='#888888',label='L1-selected parent')
    ax.scatter([],[],color='#2166ac',label='Best eligible refit checkpoint')
    ax.axhline(100*s['xgboost_metrics']['signal_efficiency_at_fixed_fpr'],color='#222222',ls='--',lw=1,label='XGBoost')
    ax.set(xlabel='Frozen support budget (including biases)',ylabel='Signal efficiency at 1% background [%]',ylim=(50.5,58),xlim=(280,950));ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(PAPER/'figures/sfr_tradeoff.pdf');plt.close(fig)

if __name__=='__main__':main()
