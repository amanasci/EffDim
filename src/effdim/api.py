from typing import Any, Dict, List, Union

import numpy as np

from effdim._native import compute_spectral

from .geometry import (
    mle_dimensionality,
    two_nn_dimensionality,
    compute_knn_distances,
    danco_dimensionality,
    mind_mli_dimensionality,
    mind_mlk_dimensionality,
    ess_dimensionality,
    tle_dimensionality,
    gmst_dimensionality,
)


def compute_dim(data: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Any]:
    """
    Compute the effective dimensionality of the given data using the specified method.

    Parameters:
    -----------
    data : Union[np.ndarray, List[np.ndarray]]
        Input data. Can be a single numpy array or a list of numpy arrays.
    Returns: dict
        A dictionary containing the results of the effective dimensionality computation.
    """
    # Getting the data and then converting to numpy array if it's a list
    if isinstance(data, list):
        data = np.vstack(data)
    elif not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array or a list of numpy arrays.")

    # Spectral keys from Rust (centers internally); do not center before this call (D-04).
    results: Dict[str, Any] = dict(
        compute_spectral(np.asarray(data, dtype=np.float64))
    )

    # Center again for the Python geometry path (D-04 duplicate center OK).
    data = _ensure_centered(data)

    # Compute KNN distances once for the largest k needed (MLE uses k=10 by default)
    # We use k=10 as a safe upper bound for default usage.
    # Convert data to float32 contiguous array once for geometry functions
    data_f32 = np.ascontiguousarray(data, dtype=np.float32)

    knn_dist_sq = compute_knn_distances(data_f32, k=10)

    results["mle_dimensionality"] = mle_dimensionality(
        data_f32, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["two_nn_dimensionality"] = two_nn_dimensionality(
        data_f32, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["danco_dimensionality"] = danco_dimensionality(
        data_f32, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["mind_mli_dimensionality"] = mind_mli_dimensionality(
        data_f32, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["mind_mlk_dimensionality"] = mind_mlk_dimensionality(
        data_f32, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["ess_dimensionality"] = ess_dimensionality(
        data_f32, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["tle_dimensionality"] = tle_dimensionality(
        data_f32, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["gmst_dimensionality"] = gmst_dimensionality(data_f32)

    return results


def _ensure_centered(data: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """
    Ensure that the data is centered around zero. If not, center it.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.
    tol : float
        Tolerance level to consider the mean as zero.

    Returns:
    --------
    np.ndarray
        Centered data array.
    """
    mean = np.mean(data, axis=0)
    if not np.all(np.abs(mean) < tol):
        data = data - mean
    return data
