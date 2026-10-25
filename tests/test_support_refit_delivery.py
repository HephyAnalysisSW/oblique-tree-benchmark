import unittest
from scripts.deliver_support_refit_results import summary_text


class SupportDeliveryTests(unittest.TestCase):
    def test_summary_distinguishes_completed_and_held_and_preserves_architecture(self):
        selection=dict(case=dict(trees=3,depth=4,id='both30'),record=dict(stage='full_best',
            metrics=dict(signal_efficiency_at_fixed_fpr=.567,signal_auc=.93),sparsity=dict(nonzero_coefficients=394)))
        text=summary_text(dict(models=[dict(phase='complete'),dict(phase='separated')],selections=dict(winner=selection),
                               xgboost_metrics=dict(signal_efficiency_at_fixed_fpr=.5548)))
        self.assertIn('1/2',text);self.assertIn('3 x D4 linear',text);self.assertIn('394 nonzero',text)
        self.assertIn('56.70%',text);self.assertIn('converged source run',text);self.assertIn('55.48%',text)
        self.assertIn('v2 implementation baseline remains unchanged',text)
