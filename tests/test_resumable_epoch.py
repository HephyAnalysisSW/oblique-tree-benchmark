import copy,pickle,unittest
import numpy as np
from tao.hard import HardTAOClassifier
from tao.epoch import begin_epoch,step_epoch


class ResumableEpochTests(unittest.TestCase):
    def test_pickle_resume_between_every_node_matches_uninterrupted_epoch(self):
        rng=np.random.default_rng(13);x=rng.normal(size=(160,4));weights=rng.uniform(.2,2,size=len(x))
        for mode in ['linear','constant']:
            for classes in [2,3]:
                for depth in [0,2]:
                    y=rng.integers(classes,size=len(x))
                    a=HardTAOClassifier(depth=depth,trees=2,leaf_mode=mode,leaf_max_iter=150,split_max_iter=200,indexed='dense').initialize(4,classes)
                    a.split_search='hard_threshold';a.global_refit(x,y,weights);b=copy.deepcopy(a)
                    reference=a.fit_epoch(x,y,weights,global_refit=False)
                    work=begin_epoch(b,x,y,weights)
                    while not step_epoch(b,work,x,y,weights):
                        b,work=pickle.loads(pickle.dumps((b,work)))
                    np.testing.assert_allclose(a.decision_function(x),b.decision_function(x),rtol=0,atol=1e-12)
                    self.assertAlmostEqual(reference['objective_after'],work['profile']['objective_after'],places=12)
                    self.assertEqual(len(reference['split_fits']),len(work['profile']['split_fits']))
                    self.assertEqual([r['accepted'] for r in reference['split_fits']],[r['accepted'] for r in work['profile']['split_fits']])
