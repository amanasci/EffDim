"""Local curvature-related estimators for point clouds.

Currently provides bootstrap PCA tangent-plane stability analysis following
Tyagi et al.: local PCA is performed with respect to a reference point ``p``
(displacements ``y_j = x_j - p``), not the empirical neighbourhood mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class BootstrapPCAResult:
    point_index: int
    radius: float
    n_neighbors: int
    n_bootstrap: int
    d: int

    eigenvalues_mean: np.ndarray  # shape [d]
    eigenvalues_var: np.ndarray  # shape [d]

    mean_projector: np.ndarray  # shape [D, D]
    mean_basis: np.ndarray  # shape [D, d]

    projector_variance: float
    normalized_projector_variance: float

    bootstrap_eigenvalues: np.ndarray  # shape [B, d]
    bootstrap_projectors: np.ndarray  # shape [B, D, D]


def radius_neighborhood(
    X: np.ndarray,
    point_index: int,
    radius: float,
    metric: str = "euclidean",
    include_self: bool = False,
) -> np.ndarray:
    """Return indices of points within ``radius`` of ``X[point_index]``.

    Parameters
    ----------
    X : ndarray, shape (n, D)
        Point cloud.
    point_index : int
        Index of the query / reference point.
    radius : float
        Neighbourhood radius.
    metric : str, optional
        Distance metric passed to ``sklearn.neighbors.NearestNeighbors``.
    include_self : bool, optional
        If False (default), exclude ``point_index`` from the result.

    Returns
    -------
    ndarray
        Integer neighbour indices.

    Raises
    ------
    ValueError
        If the neighbourhood is empty after optional self-exclusion.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    n_samples = X.shape[0]
    if not 0 <= point_index < n_samples:
        raise ValueError(
            f"point_index={point_index} is out of range for X with {n_samples} samples."
        )
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")

    nn = NearestNeighbors(radius=radius, metric=metric)
    nn.fit(X)
    # radius_neighbors returns a list of arrays (one per query).
    indices = nn.radius_neighbors(X[point_index : point_index + 1], return_distance=False)[
        0
    ]
    indices = np.asarray(indices, dtype=int)

    if not include_self:
        indices = indices[indices != point_index]

    if indices.size == 0:
        raise ValueError(
            f"Empty neighbourhood for point_index={point_index} with radius={radius}."
        )
    return indices


def pca_about_reference(
    X_local: np.ndarray,
    reference_point: np.ndarray,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA of local samples about a fixed reference point (Tyagi-style).

    Displacements are ``Y = X_local - reference_point``. The empirical mean of
    ``X_local`` is **not** subtracted. Eigenvalues correspond to
    ``(1 / m) Y.T @ Y`` via SVD: ``λ = S² / m``.

    Parameters
    ----------
    X_local : ndarray, shape (n_local, D)
        Local samples (typically neighbours of the reference).
    reference_point : ndarray, shape (D,)
        Point through which the estimated tangent space must pass.
    max_components : int
        Number of leading components to return.

    Returns
    -------
    eigenvalues : ndarray, shape (max_components,)
        Leading eigenvalues in descending order.
    eigenvectors : ndarray, shape (D, max_components)
        Corresponding orthonormal columns.
    """
    X_local = np.asarray(X_local, dtype=np.float64)
    reference_point = np.asarray(reference_point, dtype=np.float64)
    if X_local.ndim != 2:
        raise ValueError("X_local must be a 2D array of shape (n_local, D).")
    if reference_point.shape != (X_local.shape[1],):
        raise ValueError(
            f"reference_point shape {reference_point.shape} does not match "
            f"feature dimension {X_local.shape[1]}."
        )
    n_local, n_features = X_local.shape
    if n_local == 0:
        raise ValueError("X_local must contain at least one sample.")
    if max_components < 1:
        raise ValueError(f"max_components must be >= 1, got {max_components}.")
    if max_components > n_features:
        raise ValueError(
            f"max_components={max_components} exceeds feature dimension {n_features}."
        )

    Y = X_local - reference_point
    # SVD of Y: Y = U S V^T with V^T shape (rank, D); eigenvectors are columns of V.
    _, S, Vt = np.linalg.svd(Y, full_matrices=False)
    eigenvalues = (S**2) / n_local
    # Clamp tiny negative / numerical junk from floating point.
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    rank = eigenvalues.shape[0]
    if rank >= max_components:
        return eigenvalues[:max_components], Vt.T[:, :max_components]

    # Pad if the local sample is rank-deficient.
    evals = np.zeros(max_components, dtype=np.float64)
    evecs = np.zeros((n_features, max_components), dtype=np.float64)
    evals[:rank] = eigenvalues
    if rank > 0:
        evecs[:, :rank] = Vt.T[:, :rank]
    # Fill remaining columns with an orthonormal completion if needed.
    if rank < max_components:
        evecs = _complete_orthonormal_basis(evecs, rank)
    return evals, evecs


def projector(U: np.ndarray) -> np.ndarray:
    """Return the orthogonal projector ``P = QQ.T`` onto the column space of ``U``."""
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError("U must be a 2D basis matrix of shape (D, d).")
    if U.shape[1] == 0:
        return np.zeros((U.shape[0], U.shape[0]), dtype=np.float64)
    Q, _ = np.linalg.qr(U, mode="reduced")
    return Q @ Q.T


def bootstrap_pca_at_point(
    X: np.ndarray,
    point_index: int,
    radius: float,
    d: int,
    n_bootstrap: int = 100,
    sample_size: Optional[int] = None,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> BootstrapPCAResult:
    """Bootstrap local PCA tangent estimates about ``X[point_index]``.

    For each bootstrap sample of the radius neighbourhood, compute PCA about
    the reference point ``p = X[point_index]``, form projectors onto the top
    ``d`` eigenspace, then average projectors and report projector variance.
    """
    X = np.asarray(X, dtype=np.float64)
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}.")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if d > X.shape[1]:
        raise ValueError(f"d={d} exceeds ambient dimension {X.shape[1]}.")

    neighbor_indices = radius_neighborhood(
        X, point_index=point_index, radius=radius, metric=metric, include_self=False
    )
    X_local = X[neighbor_indices]
    p = X[point_index]
    n_neighbors = X_local.shape[0]

    if sample_size is None:
        sample_size = n_neighbors
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {sample_size}.")

    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]

    bootstrap_eigenvalues = np.empty((n_bootstrap, d), dtype=np.float64)
    bootstrap_projectors = np.empty((n_bootstrap, n_features, n_features), dtype=np.float64)

    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, n_neighbors, size=sample_size)
        X_boot = X_local[boot_idx]
        evals, evecs = pca_about_reference(X_boot, p, max_components=d)
        bootstrap_eigenvalues[b] = evals
        bootstrap_projectors[b] = projector(evecs)

    eigenvalues_mean = bootstrap_eigenvalues.mean(axis=0)
    eigenvalues_var = bootstrap_eigenvalues.var(axis=0)

    mean_projector = bootstrap_projectors.mean(axis=0)
    # Mean tangent basis: top-d eigenspace of the soft-mean projector.
    # mean_projector is symmetric PSD; use eigh and reverse for descending order.
    evals_P, evecs_P = np.linalg.eigh(mean_projector)
    order = np.argsort(evals_P)[::-1]
    mean_basis = evecs_P[:, order[:d]]

    diffs = bootstrap_projectors - mean_projector
    # Frobenius norm squared per bootstrap, then average.
    projector_variance = float(np.mean(np.sum(diffs * diffs, axis=(1, 2))))
    normalized_projector_variance = projector_variance / d

    return BootstrapPCAResult(
        point_index=point_index,
        radius=radius,
        n_neighbors=n_neighbors,
        n_bootstrap=n_bootstrap,
        d=d,
        eigenvalues_mean=eigenvalues_mean,
        eigenvalues_var=eigenvalues_var,
        mean_projector=mean_projector,
        mean_basis=mean_basis,
        projector_variance=projector_variance,
        normalized_projector_variance=normalized_projector_variance,
        bootstrap_eigenvalues=bootstrap_eigenvalues,
        bootstrap_projectors=bootstrap_projectors,
    )


def summarize_result(result: BootstrapPCAResult) -> dict[str, float]:
    """Flatten a ``BootstrapPCAResult`` into a simple summary dictionary."""
    summary: dict[str, float] = {
        "point_index": float(result.point_index),
        "radius": float(result.radius),
        "n_neighbors": float(result.n_neighbors),
        "n_bootstrap": float(result.n_bootstrap),
        "d": float(result.d),
        "projector_variance": float(result.projector_variance),
        "normalized_projector_variance": float(result.normalized_projector_variance),
    }
    for i, (mean_i, var_i) in enumerate(
        zip(result.eigenvalues_mean, result.eigenvalues_var), start=1
    ):
        summary[f"eig_{i}_mean"] = float(mean_i)
        summary[f"eig_{i}_var"] = float(var_i)
    return summary


def _complete_orthonormal_basis(evecs: np.ndarray, rank: int) -> np.ndarray:
    """Extend the first ``rank`` columns of ``evecs`` to a full orthonormal set."""
    n_features, max_components = evecs.shape
    if rank <= 0:
        # Arbitrary orthonormal frame via QR of a random / identity seed.
        seed = np.eye(n_features, max_components, dtype=np.float64)
        Q, _ = np.linalg.qr(seed, mode="reduced")
        return Q

    # Start from existing columns and complete with standard basis directions.
    work = np.zeros((n_features, max_components), dtype=np.float64)
    work[:, :rank] = evecs[:, :rank]
    filled = rank
    eye = np.eye(n_features, dtype=np.float64)
    for j in range(n_features):
        if filled >= max_components:
            break
        candidate = eye[:, j]
        # Gram-Schmidt against existing columns.
        for k in range(filled):
            candidate = candidate - work[:, k] * np.dot(work[:, k], candidate)
        norm = np.linalg.norm(candidate)
        if norm > 1e-10:
            work[:, filled] = candidate / norm
            filled += 1
    if filled < max_components:
        # Extremely degenerate ambient space; fall back to QR on padded matrix.
        Q, _ = np.linalg.qr(work, mode="reduced")
        return Q
    return work
