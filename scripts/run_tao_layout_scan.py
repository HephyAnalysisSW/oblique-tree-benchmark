"""One isolated, fixed-budget member of the depth/forest scan."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
import json
import subprocess
import time
from data_utils import resolve, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--index', type=int, required=True)
a = p.parse_args()
scan = json.loads(resolve('configs/tao_layout_scan_v1.json').read_text())
layout = scan['layouts'][a.index]
settings = dict(scan['shared'], task=scan['task'], **layout)
run = (f"{settings['task']}_hardtao_{settings['leaf_mode']}_d{settings['depth']}_n{settings['trees']}"
       f"_sa{settings['split_alpha']:g}_la{settings['leaf_alpha']:g}_seed{settings['seed']}_{settings['suffix']}")
command = [sys.executable, '-u', '-B', str(resolve('src/train_tao.py'))]
for key, value in settings.items():
    command += ['--' + key.replace('_', '-'), str(value)]
started = time.perf_counter()
subprocess.run(command, check=True, cwd=resolve('.'))
alternating_seconds = time.perf_counter() - started
subprocess.run([sys.executable, '-u', '-B', str(resolve('scripts/refine_tao_leaves.py')),
                '--config', f'configs/{run}.yaml', '--budgets', str(scan['final_leaf_iteration_budget'])],
               check=True, cwd=resolve('.'))
write_json(f"results/{scan['scan_name']}_member_{a.index}.json", dict(
    index=a.index, layout=layout, base_run=run, selected_run=run+'_fixedrouting_refinement',
    alternating_process_seconds=alternating_seconds, total_process_seconds=time.perf_counter()-started,
    complete=True, test_evaluated=False))
