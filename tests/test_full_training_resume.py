import hashlib,json,pickle,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import data_utils
from scripts import train_tao_fully as worker


class FullTrainingResume(unittest.TestCase):
    def test_wall_limit_resumes_and_only_terminal_convergence_is_complete(self):
        self.exercise_resume('linear')

    def test_constant_leaf_training_resumes_to_convergence(self):
        self.exercise_resume('constant')

    def exercise_resume(self,leaf_mode):
        real_root=data_utils.ROOT
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for folder in ['configs','data','results','src/tao']:(root/folder).mkdir(parents=True)
            (root/'src/tao/hard.py').write_bytes((real_root/'src/tao/hard.py').read_bytes())
            rng=np.random.default_rng(42)
            x=rng.normal(size=(200,2));xv=rng.normal(size=(80,2))
            y=(rng.random(200)<1/(1+np.exp(-x[:,0]))).astype(int)
            yv=(rng.random(80)<1/(1+np.exp(-xv[:,0]))).astype(int)
            # No test arrays exist: the training worker must not request them.
            np.savez(root/'data/fixture.npz',X_train=x,y_train=y,X_val=xv,y_val=yv)
            (root/'data/meta.json').write_text(json.dumps(dict(class_names=['background','signal'])))
            scan=json.loads((real_root/'configs/tao_full_training_v1.json').read_text())
            scan['leaf_mode']=leaf_mode
            scan['layouts']=[dict(trees=1,depth=0)];scan['source_scan']='no_source_fixture'
            (root/'configs/tao_full_training_v1.json').write_text(json.dumps(scan))
            reference=dict(data=dict(dataset_path='data/fixture.npz',metadata_path='data/meta.json'))
            (root/'configs/bdt_qg_vs_wzt_seed42.yaml').write_text(json.dumps(reference))
            digest=hashlib.sha256((root/'data/fixture.npz').read_bytes()).hexdigest()
            (root/'results/tao_split_audit.json').write_text(json.dumps(dict(dataset_sha256=digest)))
            with patch.object(data_utils,'ROOT',root),patch.object(worker,'plots',lambda *args:None):
                with patch.object(sys,'argv',['worker','--index','0','--wall-seconds','0']):
                    self.assertEqual(worker.main(),75)
                self.assertEqual(len(list((root/'models').glob('*_latest.pkl'))),1)
                self.assertFalse((root/'results/tao_full_training_v1_member_0.json').exists())
                with patch.object(sys,'argv',['worker','--index','0','--wall-seconds','1200']):
                    self.assertEqual(worker.main(),0)
            marker=json.loads((root/'results/tao_full_training_v1_member_0.json').read_text())
            self.assertTrue(marker['complete'] and marker['converged'])
            with (root/'models'/str(marker['run_name']+'.pkl')).open('rb') as f:payload=pickle.load(f)
            self.assertTrue(payload['training_converged'])
            self.assertEqual(payload['model'].leaf_mode,leaf_mode)
            self.assertEqual(payload['model'].trees[0].leaf_width,3 if leaf_mode=='linear' else 1)
            self.assertGreaterEqual(len(payload['history']),5)
            self.assertGreaterEqual(payload['convergence']['stable_epochs'],3)
            self.assertLessEqual(payload['convergence']['leaf_kkt_residual'],1e-6)


if __name__=='__main__':unittest.main()
