"""Spectral effective-dimension metrics — thin shims over ``effdim._native``."""

import numpy as np

from effdim._native import (
    geometric_mean_eff_dimensionality as _geometric_mean_eff_dimensionality,
    participation_ratio as _participation_ratio,
    pca_explained_variance as _pca_explained_variance,
    renyi_eff_dimensionality as _renyi_eff_dimensionality,
    shannon_entropy as _shannon_entropy,
)


def pca_explained_variance(spectrum: np.ndarray, threshold: float = 0.95) -> int:
    """Number of principal components required to explain a variance threshold."""
    return int(_pca_explained_variance(np.asarray(spectrum, dtype=np.float64), float(threshold)))


def participation_ratio(spectrum: np.ndarray) -> float:
    """Participation Ratio of the given eigenvalue spectrum."""
    return float(_participation_ratio(np.asarray(spectrum, dtype=np.float64)))


def shannon_entropy(probabilities: np.ndarray) -> float:
    """Shannon effective dimension of the given probability distribution."""
    return float(_shannon_entropy(np.asarray(probabilities, dtype=np.float64)))


def renyi_eff_dimensionality(probabilities: np.ndarray, alpha: float) -> float:
    """Rényi effective dimensionality; raises ValueError for invalid alpha."""
    return float(
        _renyi_eff_dimensionality(np.asarray(probabilities, dtype=np.float64), float(alpha))
    )


def geometric_mean_eff_dimensionality(spectrum: np.ndarray) -> float:
    """Geometric-mean effective dimensionality of the given spectrum."""
    return float(_geometric_mean_eff_dimensionality(np.asarray(spectrum, dtype=np.float64)))
