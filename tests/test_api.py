import numpy as np
import pytest
from effdim import api

def test_api_compute_pca():
    # Data with clear structure
    # 2 strong dimensions
    rng = np.random.RandomState(42)
    U = rng.randn(100, 2)
    data = U @ rng.randn(2, 50) + 0.01 * rng.randn(100, 50)
    
    # PCA threshold 0.9 should give 2
    ed = api.compute(data, method='pca', threshold=0.90)
    assert ed == 2.0

def test_api_compute_different_inputs():
    # Effective Rank expects singular values.
    # PR expects variance.
    # api.compute handles this.
    
    rng = np.random.RandomState(42)
    # create uncorrelated data (high dim)
    data = rng.randn(100, 100)
    
    ed_pr = api.compute(data, method='pr')
    ed_er = api.compute(data, method='effective_rank')
    
    # For random matrix, both should be high (~N or N/2?).
    # Just check they run and return floats.
    assert isinstance(ed_pr, float)
    assert isinstance(ed_er, float)
    assert ed_pr > 10.0
    
def test_api_analyze():
    data = np.random.randn(50, 50)
    results = api.analyze(data)
    assert 'participation_ratio' in results
    assert 'shannon' in results
    assert 'effective_rank' in results
    assert len(results) >= 3

def test_unknown_method():
    with pytest.raises(ValueError):
        api.compute(np.zeros((10,10)), method='invalid_method')
