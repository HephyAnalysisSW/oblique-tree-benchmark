import unittest
from scripts.compression_frontier import pareto_frontier,smallest_at_efficiency


class CompressionFrontierTests(unittest.TestCase):
    def test_dominance_keeps_only_improving_efficiency_sparsity_tradeoffs(self):
        points=[dict(id='a',nonzero_coefficients=100,efficiency=.50,mac_bound=60),
                dict(id='b',nonzero_coefficients=90,efficiency=.50,mac_bound=55),
                dict(id='c',nonzero_coefficients=200,efficiency=.56,mac_bound=90),
                dict(id='d',nonzero_coefficients=210,efficiency=.55,mac_bound=95)]
        self.assertEqual([p['id'] for p in pareto_frontier(points)],['b','c'])
        self.assertEqual(smallest_at_efficiency(points,.55)['id'],'c')
        self.assertIsNone(smallest_at_efficiency(points,.57))

    def test_efficiency_constraint_is_enforced_before_sparsity_ranking(self):
        points=[dict(id='too_low',nonzero_coefficients=10,efficiency=.5499,mac_bound=5),
                dict(id='meets',nonzero_coefficients=30,efficiency=.55,mac_bound=20),
                dict(id='faster',nonzero_coefficients=30,efficiency=.55,mac_bound=15)]
        self.assertEqual(smallest_at_efficiency(points,.55)['id'],'faster')
