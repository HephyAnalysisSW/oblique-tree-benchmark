import copy,hashlib,json,pickle,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import data_utils
from tao.hard import HardTAOClassifier
from tao.support import freeze_support
from scripts.monitor_support_refit_scan import schedule
from scripts.support_refit_artifacts import install_archiver
from scripts.train_tao_support_refit import atomic_pickle,atomic_json


class SupportScanTests(unittest.TestCase):
    def cases(self):
        return [dict(id=str(i),config=str(i)+'.json',run_name='run'+str(i),job_name='job'+str(i),adopted_pilot=i==0) for i in range(4)]

    def test_slots_include_adopted_pilot_and_pending_jobs(self):
        cases=self.cases();plan=dict(concurrency=3,cases=cases);state=dict(submissions={});status={c['id']:dict(phase='not_started') for c in cases};sent=[]
        jobs=[dict(name='job0'),dict(name='job1')]
        launched,attention=schedule(plan,state,status,jobs,lambda *a:None,lambda c:sent.append(c['id']) or '123')
        self.assertEqual(sent,['2']);self.assertEqual(len(launched),1);self.assertEqual(attention,[])

    def test_only_budget_exit_resumes_and_terminal_errors_stay_held(self):
        cases=self.cases();plan=dict(concurrency=3,cases=cases[1:]);state=dict(submissions={'1':dict(job_id='12'),'2':dict(job_id='13')})
        status={c['id']:dict(phase='epoch') for c in cases};status['3']['phase']='separated';sent=[]
        _,attention=schedule(plan,state,status,[],lambda c,j:dict(exit_code=75 if j=='12' else 1),lambda c:sent.append(c['id']) or '20')
        self.assertEqual(sent,['1']);self.assertEqual(attention[0]['case'],'2')

    def test_no_receipt_does_not_duplicate_submission(self):
        cases=self.cases();state=dict(submissions={'1':dict(job_id='12')});status={c['id']:dict(phase='epoch') for c in cases}
        _,attention=schedule(dict(concurrency=3,cases=cases[:2]),state,status,[],lambda *a:None,lambda c:self.fail('Must not submit'))
        self.assertEqual(len(attention),2)

    def test_round_robin_gives_unstarted_case_a_slot_before_long_continuation(self):
        cases=self.cases()[1:3];state=dict(submissions={'1':dict(job_id='12',allocations=2)})
        status={c['id']:dict(phase='epoch') for c in cases};sent=[]
        schedule(dict(concurrency=1,cases=cases),state,status,[],lambda *a:dict(exit_code=75),lambda c:sent.append(c['id']) or '20')
        self.assertEqual(sent,['2']);self.assertEqual(state['submissions']['2']['allocations'],1)

    def test_archive_matches_evaluated_weights_and_never_pairs_partial_epoch_history(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(data_utils,'ROOT',Path(tmp)):
            root=Path(tmp);(root/'results').mkdir();(root/'models').mkdir()
            worker=types.SimpleNamespace(persist=lambda *a:None,atomic_pickle=atomic_pickle,atomic_json=atomic_json)
            plan=dict(run_name='fixture');parent=HardTAOClassifier(depth=0).initialize(1,2)
            parent.trees[0].leaf_parameters[:]=.1;model=freeze_support(parent)
            payload=dict(model=model,metadata={},dataset_sha256='data',state=dict(history=[],parent={}))
            install_archiver(worker,plan)
            def row(stage,epoch,eff):return dict(stage=stage,epoch=epoch,metrics=dict(validation=dict(signal_efficiency_at_fixed_fpr=eff,signal_auc=.9)),convergence={})
            payload['state']['history'].append(row('leaf_only',1,.5));worker.persist('fixture',payload)
            path=root/'models/fixture_leaf_only_best_efficiency.pkl';frozen=pickle.loads(path.read_bytes())
            model.trees[0].leaf_parameters[:]=.2;payload['state']['history'].append(row('leaf_only',2,.49));worker.persist('fixture',payload)
            np.testing.assert_array_equal(pickle.loads(path.read_bytes())['model'].trees[0].leaf_parameters,frozen['model'].trees[0].leaf_parameters)
            # A terminal leaf row can be persisted just after the worker switches the live model to full mode.
            model.support_mode='full';payload['state']['history'].append(row('leaf_only',3,.51));worker.persist('fixture',payload)
            archived=pickle.loads(path.read_bytes());self.assertEqual(archived['model'].support_mode,'leaf_only')
            np.testing.assert_array_equal(archived['model'].trees[0].leaf_parameters,model.trees[0].leaf_parameters)
            # A resume starts with older history, whose coefficients no longer match the partially updated model.
            atomic_pickle(root/'models/fixture_latest.pkl',payload)
            second=types.SimpleNamespace(persist=lambda *a:None,atomic_pickle=atomic_pickle,atomic_json=atomic_json)
            install_archiver(second,plan);payload['state']['work']={'stage':'leaf'};model.trees[0].leaf_parameters[:]=.3
            second.persist('fixture',payload)
            self.assertEqual(pickle.loads(path.read_bytes())['epoch'],3)
            book=json.loads((root/'results/fixture_archived_best.json').read_text());self.assertEqual(book['leaf_only']['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
