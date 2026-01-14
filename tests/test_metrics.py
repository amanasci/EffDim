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
