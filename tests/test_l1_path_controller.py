import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import data_utils
from scripts import monitor_tao_l1_path as monitor


class L1PathControllerTests(unittest.TestCase):
    def test_control_then_one_terminal_warm_start_per_branch_with_global_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'results').mkdir()
            def row(name,branch,complete=False):return dict(config=name+'.json',scan_name=name,branch=branch,complete=complete)
            control=row('control','control');rows=[control]
            for branch in ['split','leaf','both']:
                rows.extend([row(branch+'1',branch),row(branch+'2',branch)])
            plan=dict(concurrency=3,paths={b:[] for b in ['split','leaf','both']});state=dict(submissions={});calls=[]
            def command(args):calls.append(args);return str(len(calls))
            with patch.object(data_utils,'ROOT',root),patch.object(monitor,'command',command):
                launched,_=monitor.schedule(plan,state,rows,[])
                self.assertEqual([r['config'] for r in launched],['control.json'])
                control['complete']=True
                launched,_=monitor.schedule(plan,state,rows,[dict(name='control')]);self.assertEqual(launched,[])
                launched,_=monitor.schedule(plan,state,rows,[])
                self.assertEqual({r['config'] for r in launched},{'split1.json','leaf1.json','both1.json'})
                launched,_=monitor.schedule(plan,state,rows,[dict(name=b+'1') for b in ['split','leaf','both']]);self.assertEqual(launched,[])
                rows[1]['complete']=True
                launched,_=monitor.schedule(plan,state,rows,[dict(name='leaf1'),dict(name='both1')])
                self.assertEqual([r['config'] for r in launched],['split2.json'])
                launched,attention=monitor.schedule(plan,state,rows,[])
                self.assertEqual(launched,[]);self.assertEqual({r['config'] for r in attention},{'split2.json','leaf1.json','both1.json'})
