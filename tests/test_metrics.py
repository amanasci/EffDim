import numpy as np
import pytest
from effdim import metrics

def test_pca_explained_variance():
    # Variance spectrum: [0.6, 0.3, 0.1]
    # Sum = 1.0
    s = np.array([0.6, 0.3, 0.1])
    
    # Threshold 0.5 -> 1 component (0.6)
    assert metrics.pca_explained_variance(s, threshold=0.5) == 1.0
    
    # Threshold 0.8 -> 2 components (0.9)
    assert metrics.pca_explained_variance(s, threshold=0.8) == 2.0
    
    # Threshold 0.95 -> 3 components
    assert metrics.pca_explained_variance(s, threshold=0.95) == 3.0

def test_participation_ratio():
    # [1, 1] -> sum=2, sum_sq=2 -> PR = 4/2 = 2
    s = np.array([1.0, 1.0])
    assert np.isclose(metrics.participation_ratio(s), 2.0)
    
    # [1, 0] -> sum=1, sum_sq=1 -> PR = 1
    s = np.array([1.0, 0.0])
    assert np.isclose(metrics.participation_ratio(s), 1.0)
    
    # [1, 1, 1, 1] -> PR=4
    s = np.ones(4)
    assert np.isclose(metrics.participation_ratio(s), 4.0)

def test_shannon_effective_dimension():
    # [1, 1] -> p=[0.5, 0.5] -> H = -2 * 0.5 * ln(0.5) = ln(2). Exp(H) = 2.
    s = np.array([1.0, 1.0])
    assert np.isclose(metrics.shannon_effective_dimension(s), 2.0)
    
    # [1, 0] -> p=[1] -> H = 0 -> ED = 1
    s = np.array([1.0, 0.0])
    assert np.isclose(metrics.shannon_effective_dimension(s), 1.0)

def test_renyi_effective_dimension():
    # alpha=2 should match PR
    s = np.array([1.0, 2.0, 3.0])
    # PR on this spectrum (assuming it's variance)
    pr = metrics.participation_ratio(s)
    renyi2 = metrics.renyi_effective_dimension(s, alpha=2)
    assert np.isclose(pr, renyi2)

def test_effective_rank():
    # Roy & Vetterli use s (singular values).
    # If s=[1, 1], p=[0.5, 0.5], dim=2.
    s = np.array([1.0, 1.0])
    assert np.isclose(metrics.effective_rank(s), 2.0)


def test_stable_rank_identity():
    d = 10
    spectrum = np.ones(d)
    # Expected: d
    assert np.isclose(metrics.stable_rank(spectrum), float(d))

def test_stable_rank_rank1():
    d = 10
    spectrum = np.zeros(d)
    spectrum[0] = 10.0 # Any value
    # Expected: 1
    # sum = 10, max = 10 -> 1
    assert np.isclose(metrics.stable_rank(spectrum), 1.0)

def test_numerical_rank_identity():
    d = 10
    spectrum = np.ones(d)
    # Variances = 1 -> Sigmas = 1.
    # Default epsilon ~ 1e-16
    # 1 > epsilon -> True
    # Count = d
    assert metrics.numerical_rank(spectrum) == d

def test_numerical_rank_rank1():
    d = 10
    spectrum = np.zeros(d)
    spectrum[0] = 1.0
    # Sigmas = [1, 0, ...]
    # Epsilon approx 1e-16
    # 1 > eps -> True
    # 0 > eps -> False
    assert metrics.numerical_rank(spectrum) == 1

def test_numerical_rank_custom_epsilon():
    # Sigmas: [2, 0.5, 0.01] -> Variances: [4, 0.25, 0.0001]
    spectrum = np.array([4.0, 0.25, 0.0001])
    # epsilon=1.0. 2 > 1. 0.5 > 1 (False).
    # Rank should be 1.
    assert metrics.numerical_rank(spectrum, epsilon=1.0) == 1
    # epsilon=0.1. 2>0.1, 0.5>0.1. Rank 2.
    assert metrics.numerical_rank(spectrum, epsilon=0.1) == 2

def test_cumulative_eigenvalue_ratio_identity():
    # CER = 0.5 + 1/(2d)
    for d in [1, 5, 10, 100]:
        spectrum = np.ones(d)
        expected = 0.5 + 1.0 / (2 * d)
        val = metrics.cumulative_eigenvalue_ratio(spectrum)
        assert np.isclose(val, expected)

def test_cumulative_eigenvalue_ratio_rank1():
    d = 10
    spectrum = np.zeros(d)
    spectrum[0] = 1.0
    # Expected: 1.0
    assert np.isclose(metrics.cumulative_eigenvalue_ratio(spectrum), 1.0)

def test_cumulative_eigenvalue_ratio_decay():
    # Test specific case
    # Spectrum: [4, 1]
    # Sum = 5. p = [0.8, 0.2]
    # d = 2
    # w_1 = 1 - 0/2 = 1.0
    # w_2 = 1 - 1/2 = 0.5
    # Result = 1.0*0.8 + 0.5*0.2 = 0.8 + 0.1 = 0.9
    spectrum = np.array([4.0, 1.0])
    assert np.isclose(metrics.cumulative_eigenvalue_ratio(spectrum), 0.9)

def test_empty_inputs():
    empty = np.array([])
    assert metrics.stable_rank(empty) == 0.0
    assert metrics.numerical_rank(empty) == 0.0
    assert metrics.cumulative_eigenvalue_ratio(empty) == 0.0

def test_zeros():
    z = np.zeros(5)
    assert metrics.stable_rank(z) == 0.0
    assert metrics.numerical_rank(z) == 0.0
    assert metrics.cumulative_eigenvalue_ratio(z) == 0.0
