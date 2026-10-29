"""Render the final ten-row scientific table from its frozen measured report.

Default needs only Python standard library. --plot additionally needs the
validation score files in the ignored CLIP artifact directory and matplotlib.
"""
import argparse,hashlib,json
from pathlib import Path
PAPER=Path(__file__).resolve().parent;ROOT=PAPER.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plot',action='store_true');args=parser.parse_args()
    source=PAPER/'sources/ten_model_comparison.json';report=json.loads(source.read_text())
    assert report['complete'] and len(report['rows'])==10 and report['events']==132800 and not report['test_evaluated']
    rows=[];families={'Axis / linear':'Axis / linear','Oblique / constant':'Oblique / constant','Oblique / linear':'Oblique / linear','XGBoost reference':'XGBoost'}
    for index,r in enumerate(report['rows']):
        if index in [1,4,7]:rows.append(r'\midrule')
        m=r['metrics'];a=r['arithmetic'];layout=r['layout'];name=families[r['family']]
        if r['id']=='compact_refit_3t_d4':name+=r'$^{c}$'
        rows.append(f"{name} & {layout['trees']}$\\times$D{layout['depth']} & {r['learned_nonzero_scalars']} & {100*m['signal_efficiency_at_1pct_background']:.2f} & {100*m['strict_signal_efficiency']:.2f} & {m['auc']:.5f} & {r['cpu']['median_microseconds_per_event']:.3f} & {a['path_multiplications']} & {a['all_node_multiplications']} "+r'\\')
    (PAPER/'generated/final_comparison.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    axis=json.loads((PAPER/'sources/axis_linear_study.json').read_text());rows=[]
    for r in axis['models']:
        s=r['selected'];c=r['case'];p=s['parameters'];outcome='separated' if r['refit']['phase']=='separated' else 'complete'
        rows.append(f"{c['trees']}$\\times$D{c['depth']} & {p['continuous_slots']} & {p['learned_nonzero_scalars']} & {p['feature_index_slots']} & {100*s['metrics']['signal_efficiency_at_fixed_fpr']:.2f} & {s['metrics']['signal_auc']:.5f} & {outcome} "+r'\\')
    (PAPER/'generated/axis_cohort.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
    if args.plot:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        out=ROOT/'artifacts/ten_model_comparison_v1'
        dataset=ROOT/'data/qg_vs_wzt_standardized.npz';assert sha(dataset)==report['dataset_sha256']
        with np.load(dataset,allow_pickle=False) as data:y=data['y_val']
        colors={'Axis / linear':'#2374ab','Oblique / constant':'#d47d20','Oblique / linear':'#ab3556','XGBoost reference':'black'}
        counts={};styles=['-','--',':'];fig,axes=plt.subplots(1,2,figsize=(8.2,5.7))
        for r in report['rows']:
            family=r['family'];n=counts.get(family,0);counts[family]=n+1
            values=np.fromfile(out/(r['id']+'.bin'),dtype='<f8');assert len(values)==len(y)
            fpr,tpr,_=roc_curve(y,values);layout=r['layout']
            label='XGBoost (100 × D4)' if r['id']=='xgboost' else f"{family}: {layout['trees']} × D{layout['depth']}, {r['learned_nonzero_scalars']} scalars"
            for ax in axes:ax.plot(fpr,tpr,label=label,color=colors[family],ls=styles[n%3],lw=1.3)
        axes[0].set(xlim=(0,1),ylim=(0,1),title='Validation ROC');axes[1].set_xscale('log');axes[1].set(xlim=(1e-4,.1),ylim=(0,1),title='Low background acceptance')
        for ax in axes:ax.set_xlabel('Background efficiency');ax.set_ylabel('Signal efficiency');ax.grid(alpha=.2);ax.axvline(.01,color='gray',ls=':',lw=.8)
        handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='lower center',fontsize=9,bbox_to_anchor=(.5,.01))
        fig.subplots_adjust(left=.07,right=.96,top=.92,bottom=.34,wspace=.23)
        fig.savefig(PAPER/'figures/family_roc.pdf');fig.savefig(PAPER/'figures/family_roc.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(8.2,4.5))
        for family,color in colors.items():
            selected=[r for r in report['rows'] if r['family']==family]
            for ax,key in zip(axes,['learned_nonzero_scalars','all_node_multiplications']):
                xx=[r[key] if key in r else r['arithmetic'][key] for r in selected]
                yy=[100*r['metrics']['signal_efficiency_at_1pct_background'] for r in selected]
                ax.plot(xx,yy,'o-',color=color,label=family,lw=1.,markersize=4)
                for xv,yv,r in zip(xx,yy,selected):
                    name='Compact' if r['id']=='compact_refit_3t_d4' else str(r['learned_nonzero_scalars'])
                    offset=((-37,14) if key=='learned_nonzero_scalars' else (-50,13)) if r['id']=='compact_refit_3t_d4' else ((-25,5) if r['id']=='xgboost' and key=='learned_nonzero_scalars' else ((-29,6) if key=='all_node_multiplications' and xv>1000 else (3,5)))
                    arrow=dict(arrowstyle='-',color=color,lw=.4) if r['id']=='compact_refit_3t_d4' else None
                    ax.annotate(name,(xv,yv),xytext=offset,textcoords='offset points',fontsize=9,color=color,arrowprops=arrow)
        axes[0].set_xscale('log');axes[0].set_xlim(100,4000);axes[0].set_xticks([100,300,1000,3000],['100','300','1000','3000']);axes[0].set_xlabel('Learned nonzero scalars');axes[1].set_xlabel('All-reachable multiplications')
        for ax in axes:ax.set_ylabel('Signal efficiency at 1% background [%]');ax.set_ylim(43,60);ax.grid(alpha=.2)
        axes[1].legend(loc='lower right',fontsize=9);fig.tight_layout();fig.savefig(PAPER/'figures/family_tradeoff.pdf');fig.savefig(PAPER/'figures/family_tradeoff.png',dpi=170);plt.close(fig)
    outputs=['generated/final_comparison.tex','generated/axis_cohort.tex','figures/family_roc.pdf','figures/family_tradeoff.pdf']
    provenance=dict(source_sha256=sha(source),axis_source_sha256=sha(PAPER/'sources/axis_linear_study.json'),generator_sha256=sha(Path(__file__)),
        outputs={p:sha(PAPER/p) for p in outputs if (PAPER/p).exists()})
    (PAPER/'sources/final_comparison_assets.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':main()
