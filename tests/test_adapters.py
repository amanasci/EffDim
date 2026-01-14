import numpy as np
import pytest
from effdim.adapters import get_singular_values

def test_get_singular_values_data_matrix():
    # Random (100, 10) matrix
    np.random.seed(42)
    data = np.random.randn(100, 10)
    s = get_singular_values(data)
    
    # Check shape
    assert s.shape == (10,)
    # Check sorted
    assert np.all(s[:-1] >= s[1:])
    # Check values against numpy
    _, expected_s, _ = np.linalg.svd(data, full_matrices=False)
    assert np.allclose(s, expected_s)

def test_get_singular_values_covariance():
    # Create covariance (10, 10)
    np.random.seed(42)
    # create positive definite matrix
    A = np.random.randn(10, 10)
    cov = A @ A.T
    
    s = get_singular_values(cov)
    
    # Should return sqrt(eigenvalues)
    vals = np.linalg.eigvalsh(cov)
    expected_s = np.sqrt(np.abs(vals)) # sorted ascending usually
    expected_s = np.sort(expected_s)[::-1]
    
    assert np.allclose(s, expected_s)
    
def test_input_validation():
    with pytest.raises(ValueError):
        get_singular_values(np.array([1, 2, 3])) # 1D
