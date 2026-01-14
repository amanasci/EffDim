import numpy as np
import pytest
from effdim import geometry

def test_knn_intrinsic_dimension_hypercube():
    # Generate points in D-dim hypercube
    # If points fill the space, ID should be close to D.
    np.random.seed(42)
    N = 1000
    D = 5
    data = np.random.rand(N, D)
    
    # Levina-Bickel usually slightly underestimates on bounds, but should be close.
    # k should be small relative to N
    id_est = geometry.knn_intrinsic_dimension(data, k=10)
    assert 4.0 < id_est < 6.0

def test_two_nn_intrinsic_dimension_hypercube():
    np.random.seed(42)
    N = 1000
    D = 5
    data = np.random.rand(N, D)
    
    id_est = geometry.two_nn_intrinsic_dimension(data)
    # Two-NN robustly estimates D
    assert 4.0 < id_est < 6.0

def test_knn_low_manifold():
    # 2D plane in 10D space
    np.random.seed(42)
    N = 1000
    # 2 latent vars
    latent = np.random.randn(N, 2)
    # Project to 10D linearly
    P = np.random.randn(2, 10)
    data = latent @ P
    
    id_est_k = geometry.knn_intrinsic_dimension(data, k=5)
    id_est_2 = geometry.two_nn_intrinsic_dimension(data)
    
    assert 1.5 < id_est_k < 2.5
    assert 1.5 < id_est_2 < 2.5

def test_input_validation():
    with pytest.raises(ValueError):
        geometry.knn_intrinsic_dimension(np.random.rand(10)) # 1D array
    with pytest.raises(ValueError):
        geometry.knn_intrinsic_dimension(np.random.rand(5, 5), k=10) # N < k+1
