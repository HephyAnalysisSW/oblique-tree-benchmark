"""Scheduling priority may change; numerical training policies must not."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import prioritize_constant_scan as focus


class ConstantPriorityTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.old=focus.ROOT;focus.ROOT=self.root
  self.scan={'concurrency':4,'cases':[{'index':52,'leaf_mode':'constant','scan_name':'constant52','config':'case52.json'}]}
  self.put('configs/tao_hyper_scan_v1.json',self.scan)
  self.put('results/tao_hyper_scan_v1_scheduler.json',{'attempts':{'52':[{'job_id':'123'}]}})
 def tearDown(self):focus.ROOT=self.old;self.tmp.cleanup()
 def put(self,path,data):
  p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
 def get(self,path):return json.loads((self.root/path).read_text())
 def test_active_focus_restores_limit_without_changing_cases(self):
  self.scan['concurrency']=5;self.put('configs/tao_hyper_scan_v1.json',self.scan)
  with patch.object(focus.subprocess,'check_output',return_value='123\n'):
   result=focus.tick()
  self.assertEqual(result['phase'],'running')
  self.assertEqual(self.get('configs/tao_hyper_scan_v1.json'),dict(self.scan,concurrency=4))
 def test_only_clean_continuation_receives_priority(self):
  self.put('results/hyper_scan_v1/case_52_exit.json',{'job_id':'123','exit_code':75})
  before=(self.root/'results/tao_hyper_scan_v1_scheduler.json').read_bytes()
  with patch.object(focus.subprocess,'check_output',return_value='999\n'):
   result=focus.tick()
  self.assertEqual(result['phase'],'prioritized_clean_continuation')
  self.assertTrue(self.get('results/hyper_scan_v1/case_52_recovery.json')['approved'])
  self.assertEqual(self.get('configs/tao_hyper_scan_v1.json'),dict(self.scan,concurrency=5))
  self.assertEqual(before,(self.root/'results/tao_hyper_scan_v1_scheduler.json').read_bytes())
 def test_failure_is_held_without_relaunch(self):
  self.put('results/hyper_scan_v1/case_52_exit.json',{'job_id':'123','exit_code':1})
  with patch.object(focus.subprocess,'check_output',return_value=''):
   result=focus.tick()
  self.assertEqual(result['phase'],'attention_required')
  self.assertFalse((self.root/'results/hyper_scan_v1/case_52_recovery.json').exists())
  self.assertEqual(self.get('configs/tao_hyper_scan_v1.json'),self.scan)
 def test_final_assets_require_complete_source_markers(self):
  self.put('results/constant52_member_0.json',{'complete':True,'converged':True})
  with patch.object(focus.subprocess,'check_output',return_value=''),patch.object(focus.subprocess,'run') as run:
   run.return_value.returncode=0;result=focus.tick()
  self.assertTrue(result['final_assets_ready'])
  self.assertIn('--require-complete',run.call_args.args[0])

if __name__=='__main__':unittest.main()
