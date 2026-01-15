import unittest
import numpy as np
from effdim import metrics

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Spectrum: [10, 1, 0...]
        self.spec = np.array([10.0, 1.0, 0.0, 0.0])
        # Uniform: [1, 1, 1, 1]
        self.uniform = np.array([1.0, 1.0, 1.0, 1.0])

    def test_stable_rank(self):
        # sum / max = 11 / 10 = 1.1
        val = metrics.stable_rank(self.spec)
        self.assertAlmostEqual(val, 1.1)

        # Uniform: 4 / 1 = 4
        val = metrics.stable_rank(self.uniform)
        self.assertAlmostEqual(val, 4.0)

    def test_regularized_trace_ratio(self):
        # z=1.0. 10/11 + 1/2 = 0.909 + 0.5 = 1.409
        val = metrics.regularized_trace_ratio(self.spec, z=1.0)
        self.assertAlmostEqual(val, 1.4090909)

        # z=0 (limit -> rank)
        # 10/10 + 1/1 = 2.
        # But z=0 might div zero if spec has 0?
        # spec/(spec+0) -> 0/0 -> NaN.
        # Function doesn't handle z=0 explicitly for 0 values?
        # Let's test non-zero z.
        val = metrics.regularized_trace_ratio(self.uniform, z=1e-9)
        # 1/(1+e) ~ 1. sum = 4.
        self.assertAlmostEqual(val, 4.0, places=5)

    def test_metrics_kwargs(self):
        # Ensure kwargs don't crash
        val = metrics.pca_explained_variance(self.spec, threshold=0.9, useless_arg=1)
        self.assertEqual(val, 1.0)

if __name__ == '__main__':
    unittest.main()
