import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import data_utils
from scripts import monitor_tao_improvements as monitor


class StudyControllerTests(unittest.TestCase):
    def test_no_duplicate_submissions_and_measurement_gates_next_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for folder in ['configs','results','plots/comparisons']:(root/folder).mkdir(parents=True)
            phases=[]
            for name in ['initialization','sequential_2']:
                config='configs/'+name+'.json'
                (root/config).write_text(json.dumps(dict(scan_name=name,layouts=[dict(trees=1,depth=2)])))
                phases.append(dict(name=name,configs=[config],report=name))
            (root/'configs/tao_training_improvements_v1.json').write_text(json.dumps(dict(phases=phases)))
            calls=[]
            def command(args):
                calls.append(args)
                return '123' if args[0]=='sbatch' else ''
            with patch.object(data_utils,'ROOT',root),patch.object(monitor,'command',command),patch.object(monitor.syncer,'register_file'),patch.object(monitor.syncer,'sync'):
                self.assertFalse(monitor.advance());self.assertEqual(len(calls),2)  # one submit, one queue read
                monitor.advance();self.assertEqual(sum(c[0]=='sbatch' for c in calls),1)
                (root/'results/initialization_member_0.json').write_text(json.dumps(dict(complete=True,converged=True)))
                monitor.advance();self.assertEqual(sum(c[0]=='sbatch' for c in calls),2)
                self.assertEqual(calls[-2][2],'scripts/summarize_tao_study.slurm')
                self.assertNotIn('configs/sequential_2.json',json.loads((root/'results/tao_improvements_controller_state.json').read_text())['submitted'])
                (root/'plots/comparisons/initialization_summary.json').write_text(json.dumps(dict(all_training_converged=True)))
                monitor.advance();self.assertEqual(sum(c[0]=='sbatch' for c in calls),3)
                state=json.loads((root/'results/tao_improvements_controller_state.json').read_text())
                self.assertEqual(state['submitted']['configs/sequential_2.json'],[0])
                monitor.advance();self.assertEqual(sum(c[0]=='sbatch' for c in calls),3)
