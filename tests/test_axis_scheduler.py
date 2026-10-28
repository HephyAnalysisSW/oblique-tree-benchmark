"""Continuation safety for the isolated axis study."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts import monitor_axis_study as monitor

class AxisSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.old=monitor.ROOT;monitor.ROOT=self.root
        self.case=dict(index=0,config='case.json',run_name='train',refit_run_name='refit')
        self.put('configs/axis_linear_v1.json',dict(concurrency=2,cases=[self.case]));self.put('case.json',dict(scan_name='scan'))
    def tearDown(self):monitor.ROOT=self.old;self.tmp.cleanup()
    def put(self,path,value):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def tick(self,active=''):
        with patch.object(monitor.subprocess,'check_output',side_effect=[active,'234\n']):return monitor.tick()
    def test_active_case_cannot_be_submitted_twice(self):
        self.put('results/axis_linear_v1/scheduler.json',dict(attempts={'0:train':[dict(job_id='123')]}))
        self.assertEqual(self.tick('123\n')['launched'],[])
    def test_failure_is_not_automatically_resubmitted(self):
        self.put('results/axis_linear_v1/scheduler.json',dict(attempts={'0:train':[dict(job_id='123')]}))
        self.put('results/axis_linear_v1/case_0_train_exit.json',dict(job_id='123',exit_code=1))
        result=self.tick();self.assertEqual(result['launched'],[]);self.assertEqual(len(result['attention']),1)
    def test_clean_continuation_is_resubmitted(self):
        self.put('results/axis_linear_v1/scheduler.json',dict(attempts={'0:train':[dict(job_id='123')]}))
        self.put('results/axis_linear_v1/case_0_train_exit.json',dict(job_id='123',exit_code=75))
        self.assertEqual(self.tick()['launched'],[dict(index=0,stage='train',job_id='234')])
    def test_refit_requires_converged_source_marker(self):
        self.put('results/scan_member_0.json',dict(complete=True,converged=False))
        self.assertEqual(self.tick()['launched'][0]['stage'],'train')
        self.put('results/scan_member_0.json',dict(complete=True,converged=True))
        self.assertEqual(self.tick()['launched'][0]['stage'],'refit')

if __name__=='__main__':unittest.main()
