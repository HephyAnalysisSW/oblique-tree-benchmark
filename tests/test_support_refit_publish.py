import hashlib,json,pickle,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import data_utils
from tao.hard import HardTAOClassifier
from tao.support import freeze_support,continue_full_refit
from scripts import publish_support_refit_results as publisher
from scripts import deliver_support_refit_results as delivery


class FakeBooster:
    def set_param(self,*args):pass


class FakeReference(HardTAOClassifier):
    def get_booster(self):return FakeBooster()


class SupportPublishTests(unittest.TestCase):
    def test_final_report_recomputes_snapshot_metrics_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ['data','configs','results','models']:(root/folder).mkdir()
            rng=np.random.default_rng(2);x=rng.normal(size=(80,2));y=rng.integers(0,2,80)
            np.savez(root/'data/fixture.npz',X_val=x,y_val=y,scaler_mean=[0,0],scaler_scale=[1,1])
            (root/'data/meta.json').write_text(json.dumps(dict(feature_names=['a','b'],class_names=['bkg','signal'])))
            (root/'configs/bdt_qg_vs_wzt_seed42.yaml').write_text(json.dumps(dict(data=dict(dataset_path='data/fixture.npz',metadata_path='data/meta.json'),model=dict(output_model_path='models/xgb.pkl'))))
            (root/'results/tao_split_audit.json').write_text(json.dumps(dict(dataset_sha256=hashlib.sha256((root/'data/fixture.npz').read_bytes()).hexdigest())))
            parent=HardTAOClassifier(depth=0).initialize(2,2);parent.trees[0].leaf_parameters[:]=.1
            reference=FakeReference(depth=0).initialize(2,2);reference.trees[0].leaf_parameters[:]=.1
            (root/'models/xgb.pkl').write_bytes(pickle.dumps(dict(model=reference)))
            (root/'models/parent.pkl').write_bytes(pickle.dumps(dict(model=parent,training_converged=True)))
            digest=hashlib.sha256((root/'models/parent.pkl').read_bytes()).hexdigest()
            (root/'configs/case.json').write_text(json.dumps(dict(source_checkpoint='models/parent.pkl',source_sha256=digest)))
            case=dict(id='fixture',config='configs/case.json',run_name='fixture',parent_nonzeros=3,trees=1,depth=0)
            (root/'configs/tao_support_refit_scan_v1.json').write_text(json.dumps(dict(cases=[case])))
            metrics=publisher.compute_metrics(y,parent.predict_proba(x),['bkg','signal'])[0]
            previous=HardTAOClassifier(depth=0).initialize(2,2);previous.trees[0].leaf_parameters[:]=.1;previous.trees[0].leaf_parameters[:,1]=0
            previous_metrics=publisher.compute_metrics(y,previous.predict_proba(x),['bkg','signal'])[0]
            dataset_sha=hashlib.sha256((root/'data/fixture.npz').read_bytes()).hexdigest()
            (root/'models/reference_hyper_scan_v1_c12_best_efficiency.pkl').write_bytes(pickle.dumps(dict(model=previous,
                config=dict(dataset_sha256=dataset_sha,run_name='previous'),validation=previous_metrics)))
            previous_status=root/'results/previous_status.json';previous_status.write_text(json.dumps(dict(converged=True)))
            status=dict(phase='complete',parent=dict(metrics=dict(validation=metrics)),stages=dict(leaf_only={},full={}),training_seconds=2.)
            status_path=root/'results/fixture_status.json';status_path.write_text(json.dumps(status))
            for stage in ['leaf_only','full']:
                model=freeze_support(parent) if stage=='leaf_only' else continue_full_refit(freeze_support(parent))
                (root/('models/fixture_'+stage+'.pkl')).write_bytes(pickle.dumps(dict(model=model,training_converged=True,metrics=dict(validation=metrics))))
            with patch.object(data_utils,'ROOT',root),patch.object(publisher,'bdt_counts',return_value=dict(internal_nodes=1499,leaves=1599)),\
                 patch.object(publisher,'benchmark',return_value=dict(median_microseconds_per_event=1.)),\
                 patch.object(publisher,'roc_figure'),patch('matplotlib.figure.Figure.savefig'),\
                 patch.object(publisher.syncer,'register_file'),patch.object(publisher.syncer,'sync'):
                publisher.main()
                report=json.loads((root/'plots/comparisons/support_refit_scan_v1_final_summary.json').read_text())
                self.assertFalse(report['test_evaluated']);self.assertIn('winner',report['selections'])
                self.assertEqual(report['reference_models'][0]['case']['id'],'previous_1t_no_l1')
                self.assertEqual(report['models'][0]['stages']['full']['metrics'],metrics)
                record=report['selections']['winner']['record'];snapshot=root/record['immutable_snapshot']
                self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(),record['sha256'])
                (root/'README.md').write_text('keep before\n'+delivery.BEGIN+'\nold status\n'+delivery.END+'\nkeep after\n')
                (root/'docs').mkdir()
                def git_command(args,**kwargs):
                    if args[:3]==['git','symbolic-ref','--short']:return 'nn-bdt-comparison\n'
                    if args[:3]==['git','config','--get']:return 'git@github.com:HephyAnalysisSW/oblique-tree-benchmark.git\n'
                    if args[:3]==['git','rev-parse','--short']:return 'fixture\n'
                    return ''
                with patch.object(delivery.subprocess,'check_output',side_effect=git_command):delivery.main()
                delivered=json.loads((root/'research_handoff/support_refit_scan_v1/selection.json').read_text())
                self.assertEqual(delivered['selections']['winner']['validation_vectors'],80)
                self.assertLess(delivered['selections']['winner']['native_max_margin_difference'],1e-9)
                readme=(root/'README.md').read_text();self.assertIn('keep before',readme);self.assertIn('keep after',readme)
                self.assertIn('Completed **1/1**',readme)
                previous_status.write_text(json.dumps(dict(converged=False)))
                with self.assertRaises(AssertionError):publisher.main()
                previous_status.write_text(json.dumps(dict(converged=True)))
                status['parent']['metrics']['validation']['signal_auc']+=.01;status_path.write_text(json.dumps(status))
                with self.assertRaises(AssertionError):publisher.main()
