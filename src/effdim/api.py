import warnings
from typing import Any, Dict, Iterable, List, Optional, Union

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

_VALID_SPECTRAL_BACKENDS = ("streaming_covariance", "svd")


def compute_dim(
    data: Union[np.ndarray, List[np.ndarray]],
    *,
    spectral_backend: str = "streaming_covariance",
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute the effective dimensionality of the given data.

    Parameters
    ----------
    data : numpy.ndarray or list of numpy.ndarray
        Input data as a single ``(n_samples, n_features)`` array, or a list of
        arrays that will be stacked along the sample axis.
    spectral_backend : {"streaming_covariance", "svd"}, optional
        Backend used to obtain the sample-covariance eigenvalue spectrum for
        spectral ED metrics. The return keys are identical for both backends.

        - ``"streaming_covariance"`` (default): accumulate centered covariance
          statistics (optionally in row batches) and take ``eigvalsh`` of the
          ``d×d`` covariance. Falls back to SVD when samples are fewer than
          features.
        - ``"svd"``: thin/randomized SVD of centered data, then
          ``λ = s² / (n - 1)``.
    batch_size : int, optional
        Row batch size for ``spectral_backend="streaming_covariance"``.
        Ignored by the SVD backend. ``None`` processes all rows in one GEMM.

    Returns
    -------
    dict
        Spectral and geometric effective / intrinsic dimensionality estimates.
    """
    results: Dict[str, Any] = {}

    if isinstance(data, list):
        data = np.vstack(data)
    elif not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array or a list of numpy arrays.")

    if spectral_backend not in _VALID_SPECTRAL_BACKENDS:
        raise ValueError(
            f"Unknown spectral_backend={spectral_backend!r}. "
            f"Expected one of {_VALID_SPECTRAL_BACKENDS}."
        )

    if spectral_backend == "streaming_covariance":
        data_centered = _ensure_centered(data)
        if data.shape[0] < data.shape[1]:
            warnings.warn(
                "n_samples < n_features; falling back to the SVD spectral backend "
                "to avoid forming a larger feature covariance matrix.",
                RuntimeWarning,
                stacklevel=2,
            )
            eigenvalues = _do_svd(data_centered)
        else:
            eigenvalues = _do_streaming_covariance(data, batch_size=batch_size)
    else:
        data_centered = _ensure_centered(data)
        eigenvalues = _do_svd(data_centered)

    total_variance = np.sum(eigenvalues)
    if total_variance == 0:
        probabilities = np.zeros_like(eigenvalues)
    else:
        probabilities = eigenvalues / total_variance

    results["pca_explained_variance_95"] = pca_explained_variance(
        eigenvalues, threshold=0.95
    )
    results["participation_ratio"] = participation_ratio(eigenvalues)
    results["shannon_entropy"] = shannon_entropy(probabilities)

    for i in range(2, 6):
        results[f"renyi_eff_dimensionality_alpha_{i}"] = renyi_eff_dimensionality(
            probabilities, alpha=i
        )

    results["geometric_mean_eff_dimensionality"] = geometric_mean_eff_dimensionality(
        probabilities
    )

    # Geometric path uses centered float32 data + FAISS kNN.
    data_f32 = np.ascontiguousarray(data_centered, dtype=np.float32)

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


def _do_svd(data: np.ndarray) -> np.ndarray:
    """
    Perform Singular Value Decomposition (SVD) on the input data.
    Based on dimensions, use standard SVD or randomized SVD for efficiency.

    Parameters:
    -----------
    data : np.ndarray
        Input data array (centered).

    Returns:
    --------
    np.ndarray
        Sample-covariance eigenvalues ``λ_i = s_i² / (n - 1)``.
    """
    n_samples, n_features = data.shape
    if min(n_samples, n_features) < 1000:
        s = np.linalg.svd(data, full_matrices=False, compute_uv=False)
    else:
        _, s, _ = randomized_svd(data, n_components=min(n_samples, n_features) - 1)

    return (s**2) / (n_samples - 1)


def _do_streaming_covariance(
    data: np.ndarray,
    batch_size: Optional[int] = None,
    ddof: int = 1,
) -> np.ndarray:
    """
    Sample-covariance eigenvalues via batched accumulation of XᵀX.

    Does not require a pre-centered full matrix copy. Each row batch is centered
    around its own mean and its centered sum of products is merged into the
    running covariance using the parallel-variance update. Accumulation is in
    float64.

    Parameters
    ----------
    data : np.ndarray
        Data of shape (n_samples, n_features). Need not be centered.
    batch_size : int, optional
        Row batch size. ``None`` processes all rows in one GEMM.
    ddof : int
        Divisor adjustment for the sample covariance (default 1 → n - 1).

    Returns
    -------
    np.ndarray
        Covariance eigenvalues in descending order.
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2D array of shape (n_samples, n_features).")

    n_samples, n_features = data.shape
    if n_samples == 0 or n_features == 0:
        return np.zeros(0, dtype=np.float64)
    if n_samples < 2:
        return np.zeros(n_features, dtype=np.float64)

    mean = np.zeros(n_features, dtype=np.float64)
    M2 = np.zeros((n_features, n_features), dtype=np.float64)
    n = 0

    for Xb in _iter_row_batches(data, batch_size):
        Xb64 = np.asarray(Xb, dtype=np.float64, order="C")
        batch_n = Xb64.shape[0]
        batch_mean = Xb64.mean(axis=0)
        Xb_centered = Xb64 - batch_mean
        batch_M2 = Xb_centered.T @ Xb_centered

        new_n = n + batch_n
        delta = batch_mean - mean
        M2 += batch_M2
        if n:
            M2 += np.outer(delta, delta) * (n * batch_n / new_n)
        mean += delta * (batch_n / new_n)
        n = new_n

    denom = n - ddof
    if denom <= 0:
        return np.zeros(n_features, dtype=np.float64)

    C = M2 / denom
    C = 0.5 * (C + C.T)

    evals = np.linalg.eigvalsh(C)  # ascending
    evals = evals[::-1]
    np.clip(evals, 0.0, None, out=evals)
    return evals


def _iter_row_batches(
    data: np.ndarray, batch_size: Optional[int]
) -> Iterable[np.ndarray]:
    n_samples = data.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size >= n_samples:
        yield data
        return
    for start in range(0, n_samples, batch_size):
        yield data[start : start + batch_size]


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
