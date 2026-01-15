import numpy as np
from scipy.spatial import cKDTree

def knn_intrinsic_dimension(data: np.ndarray, k: int = 5, **kwargs) -> float:
    """
    Computes Intrinsic Dimension using Levina-Bickel MLE.
    
    Args:
        data: (N, D) array of points.
        k: Number of neighbors.
        
    Returns:
        float: Estimated dimension.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Data must be 2D array (N, D).")
        
    N = data.shape[0]
    if N < k + 1:
        raise ValueError(f"Not enough samples ({N}) for k={k}.")
        
    # Query k neighbors. k+1 because the point itself is included as distance 0.
    tree = cKDTree(data)
    dists, _ = tree.query(data, k=k+1)
    
    # Drop self
    neighbors_dists = dists[:, 1:] # (N, k) - these are 1st to kth neighbors
    
    # T_k is the distance to the k-th neighbor (last column)
    T_k = neighbors_dists[:, -1] # (N,)
    
    # T_j are distances 1 to k-1. (All columns excluding last)
    T_j = neighbors_dists[:, :-1] # (N, k-1)
    
    # Avoid log(0)
    epsilon = 1e-10
    T_k = np.maximum(T_k, epsilon)
    T_j = np.maximum(T_j, epsilon)
    
    # Log ratios: log(T_k / T_j)
    log_sum = np.sum(np.log(T_k[:, None]) - np.log(T_j)) # Sum over all i, j
    
    estimator = (N * (k - 1)) / log_sum
    return float(estimator)

def two_nn_intrinsic_dimension(data: np.ndarray, **kwargs) -> float:
    """
    Computes ID using Two-NN method (Facco et al., 2017).
    Uses ratio of 2nd to 1st neighbor distances.
    
    Args:
        data: (N, D) array.
        
    Returns:
        float: Estimated dimension.
    """
    data = np.asarray(data)
    N = data.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points for Two-NN.")
        
    tree = cKDTree(data)
    dists, _ = tree.query(data, k=3) # Self, 1st, 2nd
    
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    
    mask = r1 > 1e-10
    r1 = r1[mask]
    r2 = r2[mask]
    
    mu = r2 / r1
    
    if len(mu) == 0:
        return 0.0
        
    log_mu_sum = np.sum(np.log(mu))
    if log_mu_sum == 0:
        return 0.0
        
    d_hat = len(mu) / log_mu_sum
    return float(d_hat)

def lid_intrinsic_dimension(data: np.ndarray, k: int = 20, **kwargs) -> float:
    """
    Estimates intrinsic dimension using a global average of Local Intrinsic Dimensionality (LID).

    Formula: -1 / mean(log(r_i / r_k))
    where r_i are distances to neighbors 1..k, and r_k is distance to k-th neighbor.

    Args:
        data: (N, D) array.
        k: Number of neighbors.
    """
    data = np.asarray(data)
    N = data.shape[0]
    if N < k + 1:
        raise ValueError(f"Not enough samples ({N}) for k={k}.")

    tree = cKDTree(data)
    # query k+1 to get self + k neighbors
    dists, _ = tree.query(data, k=k+1)

    # Neighbors 1 to k
    neighbors_dists = dists[:, 1:] # (N, k)

    # r_k is the last column (k-th neighbor)
    r_k = neighbors_dists[:, -1] # (N,)

    # Avoid div by zero
    epsilon = 1e-10
    r_k = np.maximum(r_k, epsilon)
    neighbors_dists = np.maximum(neighbors_dists, epsilon)

    # ratios r_i / r_k. Broadcast r_k to (N, 1)
    ratios = neighbors_dists / r_k[:, None]

    # Log ratios
    log_ratios = np.log(ratios)

    # Mean over all N points and k neighbors
    mean_log_ratio = np.mean(log_ratios)

    if mean_log_ratio == 0:
        return 0.0

    return -1.0 / mean_log_ratio

def correlation_dimension(data: np.ndarray, n_steps: int = 10, **kwargs) -> float:
    """
    Estimates Intrinsic Dimension using Grassberger-Procaccia algorithm.
    Computes Correlation Integral C(r) ~ r^d.

    Args:
        data: (N, D) array.
        n_steps: Number of steps for r.
    """
    data = np.asarray(data)
    N = data.shape[0]
    if N < 2:
        return 0.0

    tree = cKDTree(data)

    # Estimate range of r based on Nearest Neighbor distances
    dists, _ = tree.query(data, k=2)
    nn_dists = dists[:, 1]
    nn_dists = nn_dists[nn_dists > 0]

    if len(nn_dists) == 0:
        return 0.0

    r_min = np.min(nn_dists)
    # Use max NN distance scaled up, or diameter.
    # To be robust, let's use the median * constant, or max.
    r_max = np.max(nn_dists) * 5.0

    if r_min == 0: r_min = 1e-9
    if r_max <= r_min: r_max = r_min * 10.0

    # Generate radii (log spaced)
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_steps)

    # Compute C(r)
    # count_neighbors returns number of pairs (i, j) such that dist(i,j) <= r
    counts = tree.count_neighbors(tree, radii)

    # Remove self-pairs (distance 0)
    counts = counts - N
    counts = np.maximum(counts, 1) # Avoid log(0)

    # Filter for valid range (unsaturated)
    limit = N * (N - 1)
    # We want region where counts are growing but not full
    mask = (counts > 0) & (counts < limit)

    if np.sum(mask) < 2:
        return 0.0

    x = np.log(radii[mask])
    y = np.log(counts[mask])

    # Linear regression
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)
