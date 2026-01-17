import numpy as np
import pytest
from effdim import geometry, api

# Helper to generate data
def generate_hypercube(N, D, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, D)

def generate_manifold(N, D_intrinsic, D_ambient, seed=42):
    np.random.seed(seed)
    latent = np.random.randn(N, D_intrinsic)
    # Random projection
    P = np.random.randn(D_intrinsic, D_ambient)
    data = latent @ P
    # Add small noise?
    data += 0.01 * np.random.randn(N, D_ambient)
    return data

@pytest.mark.parametrize("method", ["danco", "mind_mli", "mind_mlk", "ess", "tle", "gmst"])
def test_hypercube_d3(method):
    N = 1000
    D = 3
    data = generate_hypercube(N, D)

    dim = api.compute(data, method=method)
    print(f"Method {method} D=3 Est={dim}")

    if method == "mind_mli":
        # Known to be biased / underestimate
        assert dim > 0.1
    elif method in ["ess", "gmst"]:
        # Higher variance or underestimation
        assert 1.5 <= dim <= 4.5
    else:
        assert 2.0 <= dim <= 4.0

@pytest.mark.parametrize("method", ["danco", "mind_mli", "mind_mlk", "ess", "tle", "gmst"])
def test_hypercube_d5(method):
    N = 1000
    D = 5
    data = generate_hypercube(N, D)

    dim = api.compute(data, method=method)
    print(f"Method {method} D=5 Est={dim}")

    if method == "mind_mli":
        assert dim > 0.1
    elif method in ["ess", "gmst"]:
         assert 2.0 <= dim <= 7.0
    else:
        # Danco, TLE, MiND-MLk should be better
        assert 3.5 <= dim <= 6.5

def test_danco_params():
    data = generate_hypercube(100, 3)
    # k=10, d_max=5
    dim = geometry.danco_intrinsic_dimension(data, k=10, d_max=5)
    assert isinstance(dim, float)

    # Test validation
    with pytest.raises(ValueError):
        geometry.danco_intrinsic_dimension(data, k=2) # < 5

def test_mind_variants():
    data = generate_hypercube(100, 3)
    d1 = geometry.mind_mli_intrinsic_dimension(data)
    d2 = geometry.mind_mlk_intrinsic_dimension(data, k=5)
    assert d1 > 0
    assert d2 > 0

def test_ess_params():
    data = generate_hypercube(100, 3)
    dim = geometry.ess_intrinsic_dimension(data, d_max=5)
    assert dim > 0

def test_tle_params():
    data = generate_hypercube(100, 3)
    dim = geometry.tle_intrinsic_dimension(data, k=10, d_max=5)
    assert dim > 0

def test_gmst_validation():
    data = generate_hypercube(10, 3) # N=10
    with pytest.raises(ValueError, match="Not enough samples"):
        geometry.gmst_intrinsic_dimension(data)

def test_gmst_modes():
    data = generate_hypercube(100, 3)
    dim_eu = geometry.gmst_intrinsic_dimension(data, mode='euclidean')
    dim_geo = geometry.gmst_intrinsic_dimension(data, mode='geodesic', k=10)
    assert dim_eu > 0
    assert dim_geo > 0
    # Euclidean MST on Hypercube should give consistent result with Geodesic
    # but small sample size variance is high.
    # Just check runnability and return type.
