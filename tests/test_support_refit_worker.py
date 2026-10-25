"""End-to-end refit provenance, stop/resume and failure tests without test data."""
import hashlib,json,pickle,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import data_utils
from tao.hard import HardTAOClassifier
from scripts import train_tao_support_refit as worker


class SupportWorkerTests(unittest.TestCase):
    def fixture(self,root,separated=False):
        for folder in ['configs','data','models','results']:(root/folder).mkdir(parents=True)
        rng=np.random.default_rng(71)
        x=np.repeat([[-1.,0.],[0.,0.],[1.,0.]],60,axis=0)
        y=np.tile([0,1,2],60)
        if separated:y=np.repeat([0,1,2],60)
        xv=rng.normal(size=(60,2));yv=rng.integers(0,3,60)
        np.savez(root/'data/fixture.npz',X_train=x,y_train=y,X_val=xv,y_val=yv)
        metadata=dict(class_names=['qg','w','top'],feature_names=['a','b'])
        (root/'data/meta.json').write_text(json.dumps(metadata))
        reference=dict(data=dict(dataset_path='data/fixture.npz',metadata_path='data/meta.json'))
        (root/'configs/reference.json').write_text(json.dumps(reference))
        digest=hashlib.sha256((root/'data/fixture.npz').read_bytes()).hexdigest()
        (root/'results/tao_split_audit.json').write_text(json.dumps(dict(dataset_sha256=digest)))
        model=HardTAOClassifier(depth=0,trees=2).initialize(2,3)
        for t in model.trees:t.leaf_parameters[:]=.1;t.leaf_parameters[:,1,:]=0
        parent=dict(model=model,metadata=metadata,config=dict(dataset_sha256=digest,data=reference['data']),training_converged=True)
        (root/'models/parent.pkl').write_bytes(pickle.dumps(parent))
        plan=dict(run_name='fixture_support',reference_config='configs/reference.json',source_checkpoint='models/parent.pkl',
            source_sha256=hashlib.sha256((root/'models/parent.pkl').read_bytes()).hexdigest(),global_leaf_chunk_iter=100,
            local_leaf_max_iter=100,split_max_iter=100,max_epochs=30)
        return plan,parent,x,y

    def hashes(self):return {path:hashlib.sha256(Path(data_utils.ROOT/path).read_bytes()).hexdigest() for path in worker.IMPLEMENTATION}

    def test_worker_both_stages_complete_and_resume_checks_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan,parent,x,y=self.fixture(root);code=self.hashes();real_sha=worker.sha
            def fixture_sha(path):return code[path] if path in code else real_sha(path)
            with patch.object(data_utils,'ROOT',root),patch.object(worker,'sha',fixture_sha):
                self.assertEqual(worker.train(plan,0),75)
                status=json.loads((root/'results/fixture_support_status.json').read_text());self.assertFalse(status['converged'])
                self.assertEqual(status['history'],[])
                # Force a scheduler return after the first real action; next invocation resumes phase state.
                real_persist=worker.persist
                class StopAfterSave(Exception):pass
                calls=[]
                def stop(prefix,payload):
                    real_persist(prefix,payload);calls.append(payload['state']['phase']);raise StopAfterSave
                with patch.object(worker,'persist',side_effect=stop):
                    with self.assertRaises(StopAfterSave):worker.train(plan,2000)
                self.assertEqual(calls,['global'])
                self.assertEqual(worker.train(plan,2000),0)
                self.assertEqual(worker.train(plan,0),0)
                status=json.loads((root/'results/fixture_support_status.json').read_text())
                self.assertTrue(status['converged']);self.assertFalse(status['test_evaluated'])
                self.assertEqual(set(status['stages']),{'leaf_only','full'})
                masks=[]
                for stage in ['leaf_only','full']:
                    payload=pickle.loads((root/f'models/fixture_support_{stage}.pkl').read_bytes());model=payload['model'];model.assert_support()
                    self.assertTrue(payload['training_converged']);masks.append(model.support_sha256)
                    self.assertLessEqual(payload['metrics']['training_cross_entropy'],status['parent']['metrics']['training_cross_entropy'])
                    self.assertGreaterEqual(status['stages'][stage]['epoch'],5)
                    self.assertGreaterEqual(payload['convergence']['stable_epochs'],3)
                    self.assertLessEqual(payload['convergence']['leaf_kkt_residual'],1e-6)
                self.assertEqual(masks[0],masks[1])
                with self.assertRaises(ValueError):worker.train(dict(plan,leaf_ftol=1e-10),2000)
                with patch.object(worker,'sha',side_effect=lambda path: 'changed' if path=='src/tao/support.py' else fixture_sha(path)):
                    with self.assertRaisesRegex(ValueError,'implementation changed'):worker.train(plan,2000)

    def test_separation_is_not_success_even_with_small_gradient(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan,_,_,_=self.fixture(root,separated=True);code=self.hashes();real_sha=worker.sha
            def fixture_sha(path):return code[path] if path in code else real_sha(path)
            with patch.object(data_utils,'ROOT',root),patch.object(worker,'sha',fixture_sha):
                self.assertEqual(worker.train(plan,2000),78)
                self.assertEqual(worker.train(plan,2000),78)
            state=json.loads((root/'results/fixture_support_status.json').read_text())
            self.assertEqual(state['phase'],'separated');self.assertFalse(state['converged'])
            self.assertFalse((root/'models/fixture_support_leaf_only.pkl').exists())

    def test_inconclusive_lp_and_iteration_budget_never_mark_complete(self):
        for failure in ['lp','budget']:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);plan,_,_,_=self.fixture(root);code=self.hashes();real_sha=worker.sha
                def fixture_sha(path):return code[path] if path in code else real_sha(path)
                with patch.object(data_utils,'ROOT',root),patch.object(worker,'sha',fixture_sha):
                    if failure=='lp':
                        with patch.object(worker,'separation_check',return_value=dict(status='inconclusive',score=None)):
                            self.assertEqual(worker.train(plan,2000),79)
                        self.assertEqual(worker.train(plan,2000),79)
                        self.assertEqual(worker.train(plan,2000,retry_separation=True),0)
                    else:
                        plan.update(max_global_chunks=1,kkt_tol=1e-30,global_leaf_chunk_iter=1)
                        self.assertEqual(worker.train(plan,2000),80)
                        state=json.loads((root/'results/fixture_support_status.json').read_text())
                        self.assertEqual(state['phase'],'stalled');self.assertFalse(state['converged'])

    def test_source_hash_terminal_provenance_and_audit_rejections(self):
        for corruption in ['hash','terminal','data','metadata','audit']:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);plan,parent,_,_=self.fixture(root);code=self.hashes();real_sha=worker.sha
                if corruption=='hash':plan['source_sha256']='0'*64
                elif corruption=='terminal':parent['training_converged']=False
                elif corruption=='data':parent['config']['dataset_sha256']='wrong'
                elif corruption=='metadata':parent['metadata']['class_names']=['wrong']
                elif corruption=='audit':(root/'results/tao_split_audit.json').write_text(json.dumps(dict(dataset_sha256='wrong')))
                if corruption in ['terminal','data','metadata']:
                    (root/'models/parent.pkl').write_bytes(pickle.dumps(parent));plan['source_sha256']=hashlib.sha256((root/'models/parent.pkl').read_bytes()).hexdigest()
                def fixture_sha(path):return code[path] if path in code else real_sha(path)
                with patch.object(data_utils,'ROOT',root),patch.object(worker,'sha',fixture_sha):
                    with self.assertRaises(ValueError):worker.train(plan,2000)
                self.assertFalse((root/'models/fixture_support_latest.pkl').exists())
