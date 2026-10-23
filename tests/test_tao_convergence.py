import unittest
import numpy as np
from tao.hard import HardTAOClassifier
from tao.convergence import convergence_check,leaf_kkt_residual


class ConvergenceChecks(unittest.TestCase):
    def profile(self):
        return dict(objective_before=.35,objective_after=.35-1e-7,
                    leaf_fits=[dict(converged=True)],split_fits=[dict(converged=True)],
                    global_refit=dict(converged=True))

    def test_requires_consecutive_small_changes_and_inner_convergence(self):
        p=self.profile();count=0
        for i in range(3):
            r=convergence_check(p,1e-8,0,count);count=r['stable_epochs']
            self.assertEqual(r['converged'],i==2)
        p['leaf_fits'][0]['converged']=False
        self.assertEqual(convergence_check(p,1e-8,0,count)['stable_epochs'],0)
        p=self.profile()
        for kkt,route in [(1e-3,0),(0,.01)]:
            self.assertFalse(convergence_check(p,kkt,route,2)['converged'])
        p['objective_after']=.34
        self.assertFalse(convergence_check(p,0,0,2)['converged'])

    def test_kkt_detects_wrong_leaf_and_recovers_after_fit(self):
        x=np.zeros((100,2));y=np.r_[np.zeros(70,dtype=int),np.ones(30,dtype=int)]
        m=HardTAOClassifier(depth=0,trees=1,leaf_mode='linear',leaf_max_iter=1000).initialize(2,2)
        self.assertGreater(leaf_kkt_residual(m,x,y),.1)
        m.global_refit(x,y)
        self.assertLess(leaf_kkt_residual(m,x,y),1e-6)
        m.trees[0].leaf_parameters[0,0,0]=1
        self.assertGreaterEqual(leaf_kkt_residual(m,x,y),m.leaf_alpha*.99)

    def test_objective_increase_is_not_convergence(self):
        p=self.profile();p['objective_after']=.36
        with self.assertRaises(ValueError):convergence_check(p,0,0,10)


if __name__=='__main__':unittest.main()
