import numpy as np
import faiss

def _compute_knn_distances(data: np.ndarray, k: int) -> np.ndarray:
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


def mle_dimensionality(data: np.ndarray, k: int = 10) -> float:
    """
    Estimate intrinsic dimensionality using Levina-Bickel MLE.
    Includes protection against duplicate points (distance=0).
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples = data.shape[0]
    if n_samples < 2:
        return 0.0

    # Cap k to available neighbors
    k = min(k, n_samples - 1)
    if k < 2:
        return 0.0

    # Get squared distances to k neighbors
    dist_sq = _compute_knn_distances(data, k)

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


def two_nn_dimensionality(data: np.ndarray) -> float:
    """
    Estimate intrinsic dimensionality using Two-NN.
    Corrects the regression target to -log(1 - F(mu)).
    Uses FAISS for fast nearest neighbor search.
    """
    n_samples = data.shape[0]
    if n_samples < 3:
        return 0.0

    # Get squared distances to 2 neighbors (r1, r2)
    dist_sq = _compute_knn_distances(data, 2)

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
