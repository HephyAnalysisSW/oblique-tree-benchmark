import fcntl, hashlib, json, pickle, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import data_utils
from tao.hard import HardTAOClassifier
from tao.support import freeze_support, continue_full_refit
from scripts import amend_support_solver_budget as amendment
from scripts import train_tao_support_refit as worker
import test_support_refit_worker as worker_tests


class BudgetAmendmentTests(unittest.TestCase):
    def fixture(self):
        model = HardTAOClassifier(depth=1, trees=1, split_max_iter=800).initialize(2, 2)
        model.trees[0].leaf_parameters[:] = .1
        model.trees[0].leaf_parameters[:, 0, :] = 0
        model = continue_full_refit(freeze_support(model))
        plan = dict(run_name='fixture', split_max_iter=800, kkt_tol=1e-6, loss_rtol=1e-5)
        return dict(model=model, plan=plan, dataset_sha256='data', implementation_sha256={'code': 'sha'},
                    state=dict(phase='epoch', history=[dict(epoch=1)], stable=0))

    def test_budget_increase_preserves_coefficients_predictions_policy_and_history(self):
        payload = self.fixture(); original = pickle.dumps(payload)
        x = np.random.default_rng(40).normal(size=(100, 2))
        updated, note = amendment.prepare(payload, payload['plan'], 4000, 'Repeated measured 800-iteration split exits')
        self.assertEqual(pickle.dumps(payload), original)
        np.testing.assert_array_equal(payload['model'].decision_function(x), updated['model'].decision_function(x))
        self.assertEqual(amendment.coefficient_digest(payload['model']), amendment.coefficient_digest(updated['model']))
        self.assertEqual(payload['model'].support_sha256, updated['model'].support_sha256)
        self.assertEqual(updated['model'].split_max_iter, 4000)
        self.assertEqual(updated['plan'], dict(payload['plan'], split_max_iter=4000))
        self.assertEqual(updated['state']['history'], payload['state']['history'])
        self.assertEqual(updated['implementation_sha256'], payload['implementation_sha256'])
        self.assertEqual(note['previous_split_max_iter'], 800)

    def test_rejects_decrease_noninteger_blank_reason_and_policy_mismatch(self):
        p = self.fixture()
        for budget in (400, 800, 800.1, True):
            with self.assertRaises(ValueError): amendment.prepare(p, p['plan'], budget, 'measured')
        with self.assertRaises(ValueError): amendment.prepare(p, p['plan'], 4000, ' ')
        with self.assertRaises(ValueError): amendment.prepare(p, dict(p['plan'], kkt_tol=1e-4), 4000, 'measured')
        p['model'].split_max_iter = 900
        with self.assertRaises(ValueError): amendment.prepare(p, p['plan'], 4000, 'measured')

    def test_rejects_held_or_complete_sources(self):
        for phase in ('complete', 'separated', 'inconclusive', 'stalled'):
            p = self.fixture(); p['state']['phase'] = phase
            with self.assertRaises(ValueError): amendment.prepare(p, p['plan'], 4000, 'measured')

    def test_quiescent_archive_and_actual_worker_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan, _, _, _ = worker_tests.SupportWorkerTests().fixture(root)
            codes = {path: hashlib.sha256(Path(data_utils.ROOT / path).read_bytes()).hexdigest() for path in worker.IMPLEMENTATION}
            real_sha = worker.sha
            def sha(path): return codes[path] if path in codes else real_sha(path)
            with patch.object(data_utils, 'ROOT', root), patch.object(worker, 'sha', sha):
                self.assertEqual(worker.train(plan, 0), 75)
                config = 'configs/refit.json'; worker.atomic_json(config, plan)
                worker.atomic_json('configs/tao_support_refit_scan_v1.json', dict(concurrency=5))
                original = (root / 'models/fixture_support_latest.pkl').read_bytes()
                with self.assertRaisesRegex(ValueError, 'Pause'):
                    amendment.apply(config, 500, 'measured solver cap')
                worker.atomic_json('configs/tao_support_refit_scan_v1.json', dict(concurrency=0))
                with (root / 'results/fixture_support_support.lock').open('a+') as active:
                    fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaisesRegex(ValueError, 'Worker is active'):
                        amendment.apply(config, 500, 'measured solver cap')
                self.assertEqual((root / 'models/fixture_support_latest.pkl').read_bytes(), original)
                amendment.apply(config, 500, 'measured solver cap')
                amended = json.loads((root / config).read_text())
                archive = root / 'models/support_refit_policy_archive' / (hashlib.sha256(original).hexdigest() + '.pkl')
                self.assertEqual(archive.read_bytes(), original)
                self.assertEqual(worker.train(amended, 2000), 0)
                with self.assertRaisesRegex(ValueError, 'Only an unfinished'):
                    amendment.apply(config, 600, 'completed model must stay immutable')
