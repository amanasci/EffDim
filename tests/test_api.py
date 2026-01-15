import unittest
import numpy as np
from effdim import api

class TestAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 5)
        self.s = np.array([10.0, 5.0, 1.0])

    def test_analyze_integration(self):
        # Mixed methods
        methods = ['pca', 'lid', 'stable_rank', 'mle']
        res = api.analyze(self.X, methods=methods, k=10, z=1.0)

        self.assertIn('pca', res)
        self.assertIn('lid', res)
        self.assertIn('stable_rank', res)
        self.assertIn('mle', res)

        # Check values roughly
        self.assertTrue(res['lid'] > 0)
        self.assertTrue(res['stable_rank'] > 0)

    def test_compute_spectrum(self):
        # Pass spectrum directly
        # Participation Ratio of [10, 5, 1].
        # Var: [100, 25, 1]. Sum=126. SumSq=10000+625+1=10626.
        # PR = 126^2 / 10626 = 15876 / 10626 ~ 1.49
        dim = api.compute(self.s, method='participation_ratio', is_spectrum=True)
        self.assertAlmostEqual(dim, 1.494, places=2)

    def test_analyze_spectrum(self):
        # Spectrum input -> Geometric methods should be NaN
        res = api.analyze(self.s, methods=['pr', 'lid'], is_spectrum=True, k=5)
        self.assertAlmostEqual(res['pr'], 1.494, places=2)
        self.assertTrue(np.isnan(res['lid']))

    def test_aliases(self):
        res = api.analyze(self.X, methods=['regularized', 'erank'])
        self.assertIn('regularized', res)
        self.assertIn('erank', res)

if __name__ == '__main__':
    unittest.main()
