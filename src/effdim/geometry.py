"""Geometric intrinsic-dimension estimators — thin shims over ``effdim._native``."""

from typing import Optional

import numpy as np

from effdim._native import (
    compute_knn_distances as _compute_knn_distances,
    danco_dimensionality as _danco_dimensionality,
    ess_dimensionality as _ess_dimensionality,
    gmst_dimensionality as _gmst_dimensionality,
    mind_mlk_dimensionality as _mind_mlk_dimensionality,
    mind_mli_dimensionality as _mind_mli_dimensionality,
    mle_dimensionality as _mle_dimensionality,
    tle_dimensionality as _tle_dimensionality,
    two_nn_dimensionality as _two_nn_dimensionality,
)


def _as_f64(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float64)


def _as_precomputed(precomputed_knn_dist_sq: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if precomputed_knn_dist_sq is None:
        return None
    return np.asarray(precomputed_knn_dist_sq, dtype=np.float32)


def compute_knn_distances(data: np.ndarray, k: int) -> np.ndarray:
    """Compute k nearest neighbors squared distances for each point."""
    return _compute_knn_distances(_as_f64(data), int(k))


def mle_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None,
) -> float:
    """Estimate intrinsic dimensionality using Levina-Bickel MLE."""
    return float(
        _mle_dimensionality(_as_f64(data), int(k), _as_precomputed(precomputed_knn_dist_sq))
    )


def two_nn_dimensionality(
    data: np.ndarray,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None,
) -> float:
    """Estimate intrinsic dimensionality using Two-NN."""
    return float(_two_nn_dimensionality(_as_f64(data), _as_precomputed(precomputed_knn_dist_sq)))


def danco_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None,
) -> float:
    """Estimate intrinsic dimensionality using DANCo."""
    return float(
        _danco_dimensionality(_as_f64(data), int(k), _as_precomputed(precomputed_knn_dist_sq))
    )


def mind_mli_dimensionality(
    data: np.ndarray,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None,
) -> float:
    """Estimate intrinsic dimensionality using MiND-MLi."""
    return float(_mind_mli_dimensionality(_as_f64(data), _as_precomputed(precomputed_knn_dist_sq)))


def mind_mlk_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None,
) -> float:
    """Estimate intrinsic dimensionality using MiND-MLk."""
    return float(
        _mind_mlk_dimensionality(_as_f64(data), int(k), _as_precomputed(precomputed_knn_dist_sq))
    )


def ess_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None,
) -> float:
    """Estimate intrinsic dimensionality using ESS."""
    return float(
        _ess_dimensionality(_as_f64(data), int(k), _as_precomputed(precomputed_knn_dist_sq))
    )


def tle_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None,
) -> float:
    """Estimate intrinsic dimensionality using TLE."""
    return float(
        _tle_dimensionality(_as_f64(data), int(k), _as_precomputed(precomputed_knn_dist_sq))
    )


def gmst_dimensionality(
    data: np.ndarray,
    geodesic: bool = False,
    random_state: int = 42,
) -> float:
    """Estimate intrinsic dimensionality using GMST."""
    return float(_gmst_dimensionality(_as_f64(data), bool(geodesic), int(random_state)))
