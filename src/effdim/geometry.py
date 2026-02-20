import numpy as np
import faiss
from typing import Optional
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from sklearn.neighbors import kneighbors_graph

def compute_knn_distances(data: np.ndarray, k: int) -> np.ndarray:
    """
    Compute k nearest neighbors distances for each point in data.
    Returns squared distances.
    Excludes the point itself (distance 0).
    """
    n_samples, n_features = data.shape
    # Ensure data is float32 for FAISS and contiguous
    data = np.ascontiguousarray(data, dtype=np.float32)

    # Exact search using L2 distance
    index = faiss.IndexFlatL2(n_features)
    index.add(data)

    # Search for k+1 neighbors (including self)
    k_search = min(k + 1, n_samples)

    # distances_sq is (n_samples, k_search)
    # indices is (n_samples, k_search)
    distances_sq, _ = index.search(data, k_search)

    # The first column corresponds to the point itself (distance ~0)
    # We return the remaining k columns
    return distances_sq[:, 1:]


def mle_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None
) -> float:
    """
    Estimate intrinsic dimensionality using Levina-Bickel MLE.
    Includes protection against duplicate points (distance=0).
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples = data.shape[0]
    if n_samples < 2:
        return 0.0

    if precomputed_knn_dist_sq is None:
        # Cap k to available neighbors
        k = min(k, n_samples - 1)
        if k < 2:
            return 0.0

        # Get squared distances to k neighbors
        dist_sq = compute_knn_distances(data, k)
    else:
        dist_sq = precomputed_knn_dist_sq
        k = dist_sq.shape[1]
        if k < 2:
            return 0.0

    # Convert to Euclidean distances
    dist = np.sqrt(dist_sq)

    # Add epsilon to avoid division by zero or log(0)
    dist = dist + 1e-10

    # The k-th neighbor is at index k-1 (last column)
    r_k = dist[:, k-1]  # shape (n_samples,)
    r_j = dist[:, :k-1] # shape (n_samples, k-1)

    # Compute sum of log ratios: sum_{j=1}^{k-1} ln(r_k / r_j)
    # ln(r_k / r_j) = ln(r_k) - ln(r_j)

    # Broadcasting r_k to (n_samples, 1) to subtract
    log_r_k = np.log(r_k).reshape(-1, 1)
    log_r_j = np.log(r_j)

    sum_log_ratios = np.sum(log_r_k - log_r_j, axis=1)

    # Estimate for each point: d_i = (k - 1) / sum_log_ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        dim_estimates = (k - 1) / (sum_log_ratios + 1e-10)

    # Average over all points to get the estimate
    return float(np.mean(dim_estimates))


def two_nn_dimensionality(
    data: np.ndarray,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None
) -> float:
    """
    Estimate intrinsic dimensionality using Two-NN.
    Corrects the regression target to -log(1 - F(mu)).
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples = data.shape[0]
    if n_samples < 3:
        return 0.0

    if precomputed_knn_dist_sq is None:
        # Get squared distances to 2 neighbors (r1, r2)
        dist_sq = compute_knn_distances(data, 2)
    else:
        dist_sq = precomputed_knn_dist_sq

    if dist_sq.shape[1] < 2:
        return 0.0

    # dist_sq has columns: r1^2, r2^2
    r1 = np.sqrt(dist_sq[:, 0])
    r2 = np.sqrt(dist_sq[:, 1])

    # Add epsilon
    r1 = r1 + 1e-10
    r2 = r2 + 1e-10

    mu = r2 / r1

    # Sort mu values
    mu = np.sort(mu)

    # Drop last point to avoid F(mu) = 1 -> log(0)
    mu = mu[:-1]
    n_fit = len(mu)

    if n_fit == 0:
        return 0.0

    # x = ln(mu)
    x = np.log(mu)

    # y = -ln(1 - F(mu)) = -ln(1 - i/N)
    # where i is rank 1..n_fit
    # Note: original paper uses F(mu_i) = i/N where i=1..N
    # Since we dropped the last one, i goes 1..N-1
    i = np.arange(1, n_fit + 1)
    y = -np.log(1.0 - i / n_samples)

    # Linear regression through origin: y = d * x
    # d = (x . y) / (x . x)
    x_dot_x = np.dot(x, x)
    if x_dot_x == 0:
        return 0.0

    d = np.dot(x, y) / x_dot_x
    
    return float(d)


def danco_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None
) -> float:
    """
    Estimate intrinsic dimensionality using DANCo
    (Dimensionality from Angle and Norm Concentration).
    Exploits the concentration of angles between nearest neighbor vectors.
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples, n_features = data.shape
    if n_samples < 3:
        return 0.0

    if precomputed_knn_dist_sq is not None:
        k = precomputed_knn_dist_sq.shape[1]

    k = min(k, n_samples - 1)
    if k < 2:
        return 0.0

    # Build FAISS index for neighbor indices
    data_f32 = np.ascontiguousarray(data, dtype=np.float32)
    index = faiss.IndexFlatL2(n_features)
    index.add(data_f32)
    k_search = min(k + 1, n_samples)
    _, indices = index.search(data_f32, k_search)

    neighbor_indices = indices[:, 1:]
    k_actual = neighbor_indices.shape[1]

    if k_actual < 2:
        return 0.0

    # Compute vectors from each point to its neighbors
    vectors = data[neighbor_indices] - data[:, np.newaxis, :]

    # Normalize to unit vectors
    norms = np.linalg.norm(vectors, axis=2, keepdims=True) + 1e-10
    unit_vectors = vectors / norms

    # Compute pairwise cosines for each point
    cos_matrix = np.einsum('nik,njk->nij', unit_vectors, unit_vectors)

    # Extract upper triangle (excluding diagonal) for each point
    triu_idx = np.triu_indices(k_actual, k=1)
    cos_vals = cos_matrix[:, triu_idx[0], triu_idx[1]]

    mean_cos_sq = np.mean(cos_vals ** 2)
    if mean_cos_sq < 1e-10:
        return 0.0

    return float(1.0 / mean_cos_sq)


def mind_mli_dimensionality(
    data: np.ndarray,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None
) -> float:
    """
    Estimate intrinsic dimensionality using MiND-MLi
    (Maximum Likelihood on Minimum Distances, single neighbor).
    Uses the distribution of nearest neighbor distances.
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples = data.shape[0]
    if n_samples < 3:
        return 0.0

    if precomputed_knn_dist_sq is None:
        dist_sq = compute_knn_distances(data, 1)
    else:
        dist_sq = precomputed_knn_dist_sq

    # Convert to Euclidean distances (nearest neighbor only)
    dist = np.sqrt(dist_sq[:, 0]) + 1e-10

    r_max = np.max(dist)

    # Check if all distances are equal
    if np.all(np.abs(dist - dist[0]) < 1e-10):
        return 0.0

    # d = n / Σᵢ ln(r_max / rᵢ)
    log_ratios = np.log(r_max / dist)
    sum_log_ratios = np.sum(log_ratios)

    if sum_log_ratios < 1e-10:
        return 0.0

    return float(n_samples / sum_log_ratios)


def mind_mlk_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None
) -> float:
    """
    Estimate intrinsic dimensionality using MiND-MLk
    (Maximum Likelihood on Minimum Distances, k neighbors).
    Returns the median of per-point estimates for robustness.
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples = data.shape[0]
    if n_samples < 2:
        return 0.0

    if precomputed_knn_dist_sq is None:
        k = min(k, n_samples - 1)
        if k < 2:
            return 0.0
        dist_sq = compute_knn_distances(data, k)
    else:
        dist_sq = precomputed_knn_dist_sq
        k = dist_sq.shape[1]
        if k < 2:
            return 0.0

    # Convert to Euclidean distances
    dist = np.sqrt(dist_sq)

    # Add epsilon to avoid division by zero or log(0)
    dist = dist + 1e-10

    # The k-th neighbor is at index k-1 (last column)
    r_k = dist[:, k - 1].reshape(-1, 1)
    r_j = dist[:, :k - 1]

    log_r_k = np.log(r_k)
    log_r_j = np.log(r_j)

    sum_log_ratios = np.sum(log_r_k - log_r_j, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        dim_estimates = (k - 1) / (sum_log_ratios + 1e-10)

    return float(np.median(dim_estimates))


def ess_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None
) -> float:
    """
    Estimate intrinsic dimensionality using ESS
    (Expected Simplex Skewness).
    Analyzes the skewness of local simplices formed by nearest neighbors.
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples, n_features = data.shape
    if n_samples < 3:
        return 0.0

    if precomputed_knn_dist_sq is not None:
        k = precomputed_knn_dist_sq.shape[1]

    k = min(k, n_samples - 1)
    if k < 1:
        return 0.0

    # Build FAISS index for neighbor indices
    data_f32 = np.ascontiguousarray(data, dtype=np.float32)
    index = faiss.IndexFlatL2(n_features)
    index.add(data_f32)
    k_search = min(k + 1, n_samples)
    _, indices = index.search(data_f32, k_search)

    neighbor_indices = indices[:, 1:]
    k_actual = neighbor_indices.shape[1]

    if k_actual < 1:
        return 0.0

    # Compute vectors from each point to its neighbors
    vectors = data[neighbor_indices] - data[:, np.newaxis, :]

    # Normalize to unit vectors
    norms = np.linalg.norm(vectors, axis=2, keepdims=True) + 1e-10
    unit_vectors = vectors / norms

    # Compute mean of unit vectors for each point
    centroid = np.mean(unit_vectors, axis=1)

    # Squared norm of centroid
    S = np.sum(centroid ** 2, axis=1)
    S_avg = np.mean(S)

    if S_avg < 1e-10:
        return 0.0

    return float(1.0 / (k_actual * S_avg))


def tle_dimensionality(
    data: np.ndarray,
    k: int = 10,
    precomputed_knn_dist_sq: Optional[np.ndarray] = None
) -> float:
    """
    Estimate intrinsic dimensionality using TLE
    (Tight Localities Estimator).
    Maximizes likelihood on scale-normalized distances.
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples = data.shape[0]
    if n_samples < 2:
        return 0.0

    if precomputed_knn_dist_sq is None:
        k = min(k, n_samples - 1)
        if k < 2:
            return 0.0
        dist_sq = compute_knn_distances(data, k)
    else:
        dist_sq = precomputed_knn_dist_sq
        k = dist_sq.shape[1]
        if k < 2:
            return 0.0

    # Convert to Euclidean distances
    dist = np.sqrt(dist_sq)

    # Add epsilon to avoid division by zero or log(0)
    dist = dist + 1e-10

    r_k = dist[:, k - 1].reshape(-1, 1)
    r_j = dist[:, :k - 1]

    # Normalized distances
    u_j = r_j / r_k

    # Per-point estimate: d_i = -(k-1) / Σⱼ ln(u_j)
    log_u = np.log(u_j)
    sum_log_u = np.sum(log_u, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        dim_estimates = -(k - 1) / (sum_log_u - 1e-10)

    return float(np.mean(dim_estimates))


def gmst_dimensionality(
    data: np.ndarray,
    geodesic: bool = False
) -> float:
    """
    Estimate intrinsic dimensionality using GMST
    (Geodesic Minimum Spanning Tree).
    Estimates dimension from the scaling of MST length with sample size.
    """
    n_samples = data.shape[0]
    if n_samples < 10:
        return 0.0

    # Subsample sizes
    sizes = sorted(set([
        max(4, n_samples // 8),
        max(4, n_samples // 4),
        max(4, n_samples // 2),
        n_samples
    ]))

    if len(sizes) < 2:
        return 0.0

    rng = np.random.RandomState(42)
    log_n_list = []
    log_L_list = []

    for size in sizes:
        size = min(size, n_samples)
        if size == n_samples:
            idx = np.arange(n_samples)
        else:
            idx = rng.choice(n_samples, size=size, replace=False)

        subsample = data[idx]

        if geodesic:
            k_geo = min(10, size - 1)
            graph = kneighbors_graph(subsample, k_geo, mode='distance')
            graph = graph + graph.T
            dist_matrix = shortest_path(graph, directed=False)
            dist_matrix[np.isinf(dist_matrix)] = 0.0
        else:
            dist_matrix = squareform(pdist(subsample))

        mst = minimum_spanning_tree(dist_matrix)
        L = mst.sum()

        if L > 0:
            log_n_list.append(np.log(size))
            log_L_list.append(np.log(L))

    if len(log_n_list) < 2:
        return 0.0

    # Linear regression: ln(L) = alpha * ln(n) + c
    log_n_arr = np.array(log_n_list)
    log_L_arr = np.array(log_L_list)

    mean_x = np.mean(log_n_arr)
    mean_y = np.mean(log_L_arr)

    alpha = np.sum((log_n_arr - mean_x) * (log_L_arr - mean_y)) / (
        np.sum((log_n_arr - mean_x) ** 2) + 1e-10
    )

    # d = 1 / (1 - alpha)
    if abs(1.0 - alpha) < 1e-10:
        return 0.0

    return float(1.0 / (1.0 - alpha))
