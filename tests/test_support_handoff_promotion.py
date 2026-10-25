import hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from scripts import freeze_support_handoff as export

class SupportHandoffPromotionTests(unittest.TestCase):
    def fixture(self,root):
        shutil.copytree(export.ROOT/'fpga_handoff/v3',root/'fpga_handoff/v3')
        export.write(root/'fpga_handoff/CURRENT_MODEL.json',dict(candidate_id='unchanged_old_default'))
        return root/'fpga_handoff/v3'

    def test_compact_is_explicit_default_even_with_lower_efficiency(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=self.fixture(root);export.promote(root,'fpga_handoff/v3')
            p=export.read(root/'fpga_handoff/CURRENT_MODEL.json')
            self.assertEqual(p['candidate_id'],'compact_refit_3t_d4')
            self.assertEqual(p['architecture']['nonzero_coefficients'],418)
            self.assertTrue(p['training_run_converged']);self.assertFalse(p['selected_checkpoint_converged'])
            self.assertEqual(len(p['alternative_candidate_ids']),2)
            self.assertEqual(p['checkpoint_sha256'],'e89b8923fba66f5df02046e04a1828a8ea4c56f60c1d7b201a2a2f92a0ab00fa')
            with self.assertRaises(FileExistsError):export.freeze(root,export.ROOT/'configs/fpga_handoff_v3.json','fpga_handoff/v3')

    def test_corruption_prevents_pointer_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=self.fixture(root)
            p=out/'compact_refit_3t_d4/standardized/model.json';p.write_text(p.read_text()+' ')
            with self.assertRaises(AssertionError):export.promote(root,'fpga_handoff/v3')
            self.assertEqual(export.read(root/'fpga_handoff/CURRENT_MODEL.json')['candidate_id'],'unchanged_old_default')

    def test_incomplete_cpp_coverage_prevents_pointer_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=self.fixture(root);p=out/'cpp_verification.json';r=export.read(p);r['checks'].pop();export.write(p,r)
            with self.assertRaises(AssertionError):export.promote(root,'fpga_handoff/v3')
            self.assertEqual(export.read(root/'fpga_handoff/CURRENT_MODEL.json')['candidate_id'],'unchanged_old_default')
