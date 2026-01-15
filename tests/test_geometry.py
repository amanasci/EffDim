import unittest
import numpy as np
from effdim import geometry

class TestGeometry(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate 1D line in 5D space
        # X = t * v + noise
        N = 200
        t = np.random.rand(N) * 10
        v = np.array([1, 1, 1, 1, 1]) / np.sqrt(5)
        self.X_line = np.outer(t, v) + np.random.randn(N, 5) * 0.0001

        # 3D gaussian blob
        self.X_blob = np.random.randn(200, 3)

    def test_lid_intrinsic_dimension(self):
        # Line -> ID ~ 1
        # Use small k for 1D structure
        dim = geometry.lid_intrinsic_dimension(self.X_line, k=10)
        self.assertAlmostEqual(dim, 1.0, delta=0.2)

        # Blob -> ID ~ 3
        dim = geometry.lid_intrinsic_dimension(self.X_blob, k=20)
        self.assertAlmostEqual(dim, 3.0, delta=0.5)

    def test_correlation_dimension(self):
        # Line -> ID ~ 1
        dim = geometry.correlation_dimension(self.X_line, n_steps=10)
        self.assertAlmostEqual(dim, 1.0, delta=0.2)

        # Blob -> ID ~ 3
        dim = geometry.correlation_dimension(self.X_blob, n_steps=10)
        self.assertAlmostEqual(dim, 3.0, delta=0.5)

    def test_knn_alias(self):
        # Just check it runs
        dim = geometry.knn_intrinsic_dimension(self.X_blob, k=5)
        self.assertTrue(dim > 0)

if __name__ == '__main__':
    unittest.main()
