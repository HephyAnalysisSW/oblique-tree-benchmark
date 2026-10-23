import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import data_utils
from scripts import monitor_tao_hyper_scan as monitor


class HyperScanSchedulerTests(unittest.TestCase):
    def test_concurrency_cap_is_global_across_continuations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'results').mkdir()
            state=dict(attempts={'0':[dict(job_id='1')]})
            rows=[dict(index=i,converged=False,priority=-1 if i==3 else i) for i in range(5)]
            active=[dict(job_id='1'),dict(job_id='unrelated-scan-job')]
            calls=[]
            def command(args):calls.append(args);return str(10+len(calls))
            with patch.object(data_utils,'ROOT',root),patch.object(monitor,'command',command):
                launched,attention=monitor.schedule(dict(concurrency=3),state,rows,active)
                self.assertEqual(len(launched),1);self.assertEqual(len(calls),1);self.assertEqual(attention,[])
                self.assertEqual(launched[0]['index'],3)
                active.append(dict(job_id=launched[0]['job_id']))
                launched,_=monitor.schedule(dict(concurrency=3),state,rows,active)
                self.assertEqual(launched,[])
                self.assertEqual(len(calls),1)

    def test_only_clean_continuation_is_automatically_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'results').mkdir()
            state=dict(attempts={'0':[dict(job_id='1')],'1':[dict(job_id='2')]})
            rows=[dict(index=i,converged=False) for i in range(2)]
            def read(path,default=None):return dict(job_id='1',exit_code=75) if 'case_00' in path else dict(job_id='2',exit_code=1)
            with patch.object(data_utils,'ROOT',root),patch.object(monitor,'read',read),patch.object(monitor,'command',return_value='3'):
                launched,attention=monitor.schedule(dict(concurrency=4),state,rows,[])
            self.assertEqual([r['index'] for r in launched],[0])
            self.assertEqual([r['index'] for r in attention],[1])

    def test_explicit_inspected_timeout_recovery_can_resume_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'results').mkdir()
            state=dict(attempts={'0':[dict(job_id='timed_out')]})
            rows=[dict(index=0,converged=False),dict(index=1,converged=False)]
            def read(path,default=None):
                return dict(job_id='timed_out',approved=True,reason='Inspected checkpoint after timeout') if 'recovery' in path else {}
            with patch.object(data_utils,'ROOT',root),patch.object(monitor,'read',read),patch.object(monitor,'command',return_value='new_job'):
                launched,attention=monitor.schedule(dict(concurrency=1),state,rows,[])
                self.assertEqual([r['index'] for r in launched],[0]);self.assertEqual(attention,[])
                self.assertEqual(state['attempts']['0'][-1]['recovery_of']['job_id'],'timed_out')
                launched,attention=monitor.schedule(dict(concurrency=1),state,[rows[0]],[])
                self.assertEqual(launched,[]);self.assertEqual(len(attention),1)
