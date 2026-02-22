from typing import Any, Dict, List, Union

import numpy as np
from sklearn.utils.extmath import randomized_svd

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
from .metrics import (
    geometric_mean_eff_dimensionality,
    participation_ratio,
    pca_explained_variance,
    renyi_eff_dimensionality,
    shannon_entropy,
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
    results: Dict[str, Any] = {}

    # Getting the data and then converting to numpy array if it's a list
    if isinstance(data, list):
        data = np.vstack(data)
    elif not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array or a list of numpy arrays.")

    # Ensure the data is centered
    data = _ensure_centered(data)
    s = _do_svd(data)

    # gettinf the eigenvalues from the singular values for the covariance matrix
    eigenvalues = (s**2) / (data.shape[0] - 1)

    # Total variance
    total_variance = np.sum(eigenvalues)

    #  getting the probabilities
    if total_variance == 0:
        probabilities = np.zeros_like(eigenvalues)
    else:
        probabilities = eigenvalues / total_variance

    # Computing various effective dimensionalities
    results["pca_explained_variance_95"] = pca_explained_variance(
        eigenvalues, threshold=0.95
    )
    results["participation_ratio"] = participation_ratio(eigenvalues)
    results["shannon_entropy"] = shannon_entropy(probabilities)

    # Renyi effective dimensionalities for alpha = 2,3,4,5
    for i in range(2, 6):
        results[f"renyi_eff_dimensionality_alpha_{i}"] = renyi_eff_dimensionality(
            probabilities, alpha=i
        )

    # Geometric Dimensions
    results["geometric_mean_eff_dimensionality"] = geometric_mean_eff_dimensionality(
        probabilities
    )

    # Compute KNN distances once for the largest k needed (MLE uses k=10 by default)
    # We use k=10 as a safe upper bound for default usage.
    knn_dist_sq = compute_knn_distances(data, k=10)

    results["mle_dimensionality"] = mle_dimensionality(
        data, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["two_nn_dimensionality"] = two_nn_dimensionality(
        data, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["danco_dimensionality"] = danco_dimensionality(
        data, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["mind_mli_dimensionality"] = mind_mli_dimensionality(
        data, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["mind_mlk_dimensionality"] = mind_mlk_dimensionality(
        data, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["ess_dimensionality"] = ess_dimensionality(
        data, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["tle_dimensionality"] = tle_dimensionality(
        data, precomputed_knn_dist_sq=knn_dist_sq
    )
    results["gmst_dimensionality"] = gmst_dimensionality(data)

    return results


def _do_svd(data: np.ndarray) -> np.ndarray:
    """
    Perform Singular Value Decomposition (SVD) on the input data.
    Based on dimensions, use standard SVD or randomized SVD for efficiency.

    Parameters:
    -----------
    data : np.ndarray
        Input data array.

    Returns:
    --------
    np.ndarray
        Singular values of the input data.
    """
    n_samples, n_features = data.shape
    if min(n_samples, n_features) < 1000:
        s = np.linalg.svd(data, full_matrices=False, compute_uv=False)
    else:
        _, s, _ = randomized_svd(data, n_components=min(n_samples, n_features) - 1)

    return s


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
