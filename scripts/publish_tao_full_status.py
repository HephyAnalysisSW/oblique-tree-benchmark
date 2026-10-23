"""Publish the split audit and last saved full-training state, clearly provisional."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
from datetime import datetime,timezone
from data_utils import resolve,write_json
import common.syncer as syncer

scan=json.loads(resolve('configs/tao_full_training_v1.json').read_text())
audit=json.loads(resolve('results/tao_split_audit.json').read_text())
rows=[]
for i,layout in enumerate(scan['layouts']):
    run=(f"{scan['task']}_hardtao_linear_d{layout['depth']}_n{layout['trees']}"
         f"_sa{scan['split_alpha']:g}_la{scan['leaf_alpha']:g}_seed{scan['seed']}_full_training_v1")
    status=resolve('results/'+run+'_status.json')
    saved=json.loads(status.read_text()) if status.exists() else dict(phase='no saved optimizer step yet',epochs=0,converged=False,stable_epochs=0)
    rows.append(dict(saved,layout=layout,run_name=run))
root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
updated=datetime.now(timezone.utc).isoformat()
complete=all(r['converged'] for r in rows)
lines=['# Full TAO training: last saved state','',
    'All layouts converged; comparison/timing publication follows automatically.' if complete else
    'Training is in progress. These are checkpoint states, not a completed comparison.',
    '', 'Updated UTC: '+updated,'',
    '| Trees | Depth | Saved phase | Additional epochs | Stable passes | Converged |',
    '|---:|---:|---|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['layout']['trees']} | {r['layout']['depth']} | {r['phase']} | {r['epochs']} | {r['stable_epochs']} | {r['converged']} |")
lines+=['','## Verified partitions','',
    'The NPZ is byte-identical to Dorian\'s original export. TAO and retrained XGBoost consume the same arrays directly, with no new split or scaler fit.',
    '- Training: 531,200 events; validation: 132,800; held-out test: 166,000.',
    '- Original XGBoost split: 64/16/20%, seed 42.',
    '- Test data are not used for training or selection. Test members were hashed only for integrity.',
    '- Dataset SHA-256: `'+audit['dataset_sha256']+'`.','',
    'The complete member hashes, model lineage and audit scope are in split_audit.json.','',
    '## Numerical convergence','',
    'At least five additional passes and three consecutive stable passes: relative objective gain <=1e-5, changed training routes <=1e-4, converged local leaf/split fits, and joint-leaf normalized L1 KKT violation <=1e-6.',
    'Scheduler and inner-iteration limits cause continuation, never a converged label. This is coordinate convergence, not a proof of a global optimum.',
    '', 'Seven layouts: single trees at depths 2–5; two depth-3 trees; three depth-2 and three depth-3 trees. L1 penalties remain 1e-5; all leaves are linear.',
    '', 'Final results will be in full_training_report.md, full_training_summary.json and full_training_*.png. Historical layout_scan_* figures are the four-epoch screening study.']
(root/'full_training_status.md').write_text('\n'.join(lines)+'\n')
write_json(root/'full_training_status.json',dict(updated_utc=updated,all_converged=complete,models=rows))
write_json(root/'split_audit.json',audit)
for name in ['full_training_status.md','full_training_status.json','split_audit.json']:syncer.register_file(root/name)
print('Published full-training status:',sum(r['converged'] for r in rows),'/',len(rows),'converged',flush=True)
