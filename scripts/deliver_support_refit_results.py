"""Record completed scan results and portable research candidates, then push explicit paths."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import hashlib,json,pickle,shutil,subprocess
import numpy as np
from data_utils import resolve,load_config,write_json
from tao.portable import predict_margin
from tao.support import freeze_support

BEGIN='<!-- SUPPORT_REFIT_SCAN_CURRENT_BEGIN -->'
END='<!-- SUPPORT_REFIT_SCAN_CURRENT_END -->'
BASE='https://schoef.web.cern.ch/schoef/tree_benchmark/plots/comparisons/'


def summary_text(summary):
    completed=sum(row['phase']=='complete' for row in summary['models']);held=len(summary['models'])-completed
    lines=['### Support-refit scan results','',f"Completed **{completed}/{len(summary['models'])}** full refits; **{held}** cases have explicitly held outcomes. All results use validation; test data are untouched.",'']
    roles={'winner':'Highest efficiency','compact_match_xgb':'Fewest coefficients matching XGBoost','compact_path_match_xgb':'Fewest sparse path multiplications matching XGBoost'}
    for tag in roles:
        selected=summary['selections'].get(tag)
        if selected:
            c,r=selected['case'],selected['record'];metric=r['metrics'];counts=r['sparsity']
            source=next((row for row in summary['models'] if row.get('case',{}).get('id')==c['id']),None)
            penalty=''
            if source and source.get('selection_penalties'):
                p=source['selection_penalties'];penalty=f" Support selected with split/leaf L1 **{p['split_alpha']:g}/{p['leaf_alpha']:g}**."
            strategy=('Fixed sparse splits; surviving linear-leaf coefficients refitted without L1.' if r['stage'].startswith('leaf_only') else
                'Surviving split and linear-leaf coefficients refitted without L1 inside immutable masks.' if r['stage'].startswith('full') else
                'Original trained reference checkpoint.')
            lines.append(f"- {roles[tag]}: **{c['trees']} x D{c['depth']} linear**, {counts['nonzero_coefficients']} nonzero coefficients, **{100*metric['signal_efficiency_at_fixed_fpr']:.2f}%** signal efficiency at 1% background, AUC {metric['signal_auc']:.5f}. Selector `{c['id']}`; saved `{r['stage']}` checkpoint from a converged source stage.{penalty} {strategy}")
    lines+=['',f"XGBoost reference: **{100*summary['xgboost_metrics']['signal_efficiency_at_fixed_fpr']:.2f}%** at the same working point.",'',
        f"[Final table]({BASE}support_refit_scan_v1_final_report.md) · [ROC]({BASE}support_refit_scan_v1_roc.png) · [Screenshot table]({BASE}support_refit_scan_v1_recommendations.png).",'',
        'See [full results](docs/support_refit_scan_results.md) and the coefficient-pinned [research candidates](research_handoff/support_refit_scan_v1/selection.json). The FPGA v2 implementation baseline remains unchanged; these candidates are floating-point references, not quantized or synthesized implementations.']
    u=summary.get('uncertainty',{}).get('models',{}).get('winner')
    if u:
        lo,hi=u['difference_interval95']
        lines+=['',f"Winner minus XGBoost: {100*u['difference_vs_reference']:+.2f} percentage points; paired validation bootstrap 95% interval [{100*lo:+.2f}, {100*hi:+.2f}]. This conditions on selected trained models and does not correct for validation selection or training-seed variation."]
    return '\n'.join(lines)+'\n'


def main():
    summary=json.loads(resolve('plots/comparisons/support_refit_scan_v1_final_summary.json').read_text())
    plan=json.loads(resolve('configs/tao_support_refit_scan_v1.json').read_text())
    assert len(summary['models'])==len(plan['cases'])
    assert all(row['phase'] in ['complete','separated','inconclusive','stalled'] for row in summary['models'])
    root=resolve('research_handoff/support_refit_scan_v1');root.mkdir(parents=True,exist_ok=True)
    cfg=load_config('configs/bdt_qg_vs_wzt_seed42.yaml')
    with np.load(resolve(cfg['data']['dataset_path'])) as data:
        validation=data['X_val'].astype(float);mean,scale=data['scaler_mean'],data['scaler_scale']
    records={}
    for tag,selection in summary['selections'].items():
        r=selection['record'];blob=resolve(r['immutable_snapshot']).read_bytes();assert hashlib.sha256(blob).hexdigest()==r['sha256']
        payload=pickle.loads(blob);model=payload['model']
        if not hasattr(model,'assert_support'):model=freeze_support(model)  # Freeze coefficients for inference export, without training.
        model.assert_support()
        folder=root/tag;folder.mkdir(exist_ok=True);cache=resolve('models/support_refit_scan_v1_selected')/tag
        indices=np.atleast_1d(np.loadtxt(cache/'validation_row_indices.tsv',dtype=int));x=validation[indices]
        spec=json.loads((cache/'standardized_model.json').read_text());margin,routes,_=predict_margin(spec,x)
        assert np.max(np.abs(margin+model.decision_function(x)[:,0]))<1e-9
        raw_spec=json.loads((cache/'raw_model.json').read_text())
        raw_margin,raw_routes,_=predict_margin(raw_spec,x*scale+mean)
        assert np.array_equal(routes,raw_routes) and np.max(np.abs(raw_margin-margin))<1e-8
        np.savetxt(folder/'validation_vectors.tsv',x,fmt='%.17g')
        np.savetxt(folder/'validation_row_indices.tsv',indices,fmt='%d')
        np.savetxt(folder/'expected_margin_and_leaves.tsv',np.column_stack((margin,routes)),fmt=['%.17g']+['%d']*model.ntrees)
        np.savetxt(folder/'raw_validation_vectors.tsv',x*scale+mean,fmt='%.17g')
        np.savetxt(folder/'raw_expected_margin_and_leaves.tsv',np.column_stack((raw_margin,raw_routes)),fmt=['%.17g']+['%d']*model.ntrees)
        hashes={}
        for name in ['standardized_model.json','raw_model.json']:
            shutil.copyfile(cache/name,folder/name);hashes[name]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
        records[tag]=dict(selection=selection,model_json_sha256=hashes,validation_vectors=len(x),
            native_max_margin_difference=float(np.max(np.abs(margin+model.decision_function(x)[:,0]))),
            raw_fold_max_margin_difference=float(np.max(np.abs(raw_margin-margin))),
            support_sha256=model.support_sha256)
    write_json(root/'selection.json',dict(dataset_sha256=summary['dataset_sha256'],selections=records,test_evaluated=False,
        note='Floating-point research candidates. Saved checkpoint selection is distinct from terminal source-run convergence. Each input-space JSON has its own strict margin cut accepting at most floor(1% of validation background), separately from interpolated ROC metrics. v2 CURRENT_MODEL is unchanged.'))
    text=summary_text(summary);readme=resolve('README.md').read_text()
    assert readme.count(BEGIN)==1 and readme.count(END)==1
    start=readme.index(BEGIN)+len(BEGIN);end=readme.index(END)
    resolve('README.md').write_text(readme[:start]+'\n'+text+readme[end:])
    detailed=resolve('plots/comparisons/support_refit_scan_v1_final_report.md').read_text()
    for name in ['support_refit_scan_v1_final_table.png','support_refit_scan_v1_recommendations.png','support_refit_scan_v1_roc.png']:detailed=detailed.replace('('+name+')','('+BASE+name+')')
    resolve('docs/support_refit_scan_results.md').write_text(text.replace('(docs/','(').replace('(research_handoff/','(../research_handoff/')+'\n'+detailed)
    def command(args):return subprocess.check_output(args,cwd=resolve('.'),text=True).strip()
    assert command(['git','symbolic-ref','--short','HEAD'])=='nn-bdt-comparison'
    assert command(['git','config','--get','remote.origin.url'])=='git@github.com:HephyAnalysisSW/oblique-tree-benchmark.git'
    assert not command(['git','diff','--cached','--name-only']),'Refusing to include unrelated staged changes'
    command(['git','add','README.md','docs/support_refit_scan_results.md','research_handoff/support_refit_scan_v1'])
    command(['git','diff','--cached','--check'])
    if command(['git','diff','--cached','--name-only']):command(['git','commit','-m','Record completed support-refit scan and portable research winners'])
    command(['git','push','origin','nn-bdt-comparison','nn-bdt-comparison:main'])
    print('Delivered support-refit results:',command(['git','rev-parse','--short','HEAD']),flush=True)


if __name__=='__main__':main()
