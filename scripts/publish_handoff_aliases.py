"""Publish current-winner aliases from the pinned, verified handoff bundle."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import hashlib,json,shutil
from data_utils import resolve,write_json
import common.syncer as syncer
import matplotlib.pyplot as plt


def main():
    current=json.loads(resolve('fpga_handoff/CURRENT_MODEL.json').read_text())
    manifest_path=resolve(current['manifest']);manifest=json.loads(manifest_path.read_text())
    selected=next(c for c in manifest['candidates'] if c['id']==current['candidate_id'])
    assert selected['checkpoint_sha256']==current['checkpoint_sha256']
    assert abs(selected['metrics']['signal_efficiency_at_fixed_fpr']-current['validation']['signal_efficiency'])<1e-12
    for key in ['standardized_model','raw_model']:
        assert hashlib.sha256(resolve(current[key]).read_bytes()).hexdigest()==current[key+'_sha256']
    version=current['handoff_version'];root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    architecture=current['architecture'];folder=resolve(f"plots/tao_{architecture['trees']}tree_depth{architecture['depth']}_{architecture['leaf_mode']}");folder.mkdir(parents=True,exist_ok=True)
    for source,target in [(root/f'fpga_handoff_{version}_roc.png',root/'current_winner_roc.png'),
        (root/f'fpga_handoff_{version}_roc.png',folder/'winner_roc.png'),
        (root/f'fpga_handoff_{version}_comparison.png',root/'current_winner_comparison.png')]:
        shutil.copyfile(source,target);syncer.register_file(target)
    comparison=json.loads((manifest_path.parent/'comparison.json').read_text())
    labels={}
    for candidate in manifest['candidates']:
        policy=candidate['training_policy'];suffix=''
        if policy['split_alpha']==0:suffix=' [L1=0]' if policy['leaf_alpha']==0 else ' [split L1=0]'
        if candidate['id']==current.get('compact_candidate_id'):suffix+=' [compact]'
        if candidate['id'].startswith('previous_'):suffix+=' [v1]'
        labels[candidate['id']]=suffix
    cells=[]
    for row in comparison:
        metric=row['metrics']
        cells.append([row['label'].replace('1 trees,','1 tree,')+labels.get(row['id'],''),str(row['parameters']),
            f"{metric['signal_efficiency_at_fixed_fpr']:.2%}",f"{metric['signal_auc']:.5f}",
            f"{row['timings'][-1]['median_microseconds_per_event']:.3f}"])
    fig,ax=plt.subplots(figsize=(14,5));ax.axis('off')
    table=ax.table(cellText=cells,colLabels=['Model','Parameters','Signal eff. at 1% bkg','AUC','CPU us/event'],
        colWidths=[.44,.10,.20,.12,.14],cellLoc='center',bbox=[0,.20,1,.68])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d9dee7')
        if row==0:cell.set_facecolor('#23334d');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        else:cell.set_facecolor('#edf5ff' if '(winner)' in cells[row-1][0] else ('#f5f6f8' if row%2 else 'white'))
    fig.suptitle('Finalized floating-point FPGA baseline | '+version,fontsize=18,y=.98)
    ax.text(.5,.94,'q/g vs W/Z/top | validation | frozen best-efficiency checkpoints',ha='center',transform=ax.transAxes,fontsize=12)
    ax.text(.01,.10,'Parameters exclude shared preprocessing and topology. Some source training runs continue; snapshots remain fixed.',transform=ax.transAxes,fontsize=9)
    ax.text(.01,.04,'CPU: batch 132,800, one thread; NumPy TAO versus compiled XGBoost. FPGA precision and synthesis are subsequent steps.',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(root/'current_winner_comparison.png',dpi=180);plt.close(fig);syncer.register_file(root/'current_winner_comparison.png')
    table_lines=['| Model | Parameters | Efficiency at 1% background | AUC | Bulk CPU us/event |','|---|---:|---:|---:|---:|']+['| '+' | '.join(row)+' |' for row in cells]
    lines=['# Current verified FPGA handoff baseline','',
        f"Version {version}; {architecture['trees']} depth-{architecture['depth']} trees; {architecture['leaf_mode']} leaves; {architecture['stored_scalar_parameters']} scalar parameters.",
        '', f"Selected pass {selected['selected_epoch']}; source training run converged at freeze: {selected['training_run_converged']}. The frozen snapshot is implementation-ready as a floating reference; numerical training convergence and FPGA synthesis are separate statuses.",
        '', '\n'.join(table_lines),
        '', '![Frozen comparison](current_winner_comparison.png)', '', '![ROC](current_winner_roc.png)']
    (root/'current_winner_report.md').write_text('\n'.join(lines)+'\n');syncer.register_file(root/'current_winner_report.md')
    result=dict(current_selection=current,selected_model=selected,models=json.loads((manifest_path.parent/'comparison.json').read_text()),test_evaluated=False)
    for path in [root/'current_winner_summary.json',folder/'winner_metrics.json']:write_json(path,result);syncer.register_file(path)
    print('Published current-winner aliases from frozen',version,current['candidate_id'],flush=True)


if __name__=='__main__':main()
