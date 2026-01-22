import numpy as np
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. MLE (Levina-Bickel) - Robust Version
# ==========================================
def mle_dimensionality(data: np.ndarray, k: int = 10) -> float:
    """
    Estimate intrinsic dimensionality using Levina-Bickel MLE.
    Includes protection against duplicate points (distance=0).
    """
    n_samples = data.shape[0]
    
    # Safety: We need k+1 neighbors because the 0th is the point itself.
    # If k >= n_samples, clamp it.
    k = min(k, n_samples - 1)
    if k < 2: return 0.0

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Drop the first column (distance to self, which is 0)
    distances = distances[:, 1:]
    
    # STABILITY FIX:
    # If data has duplicates, neighbors might be at distance 0.
    # We replace 0 with a tiny epsilon to avoid log(0) -> -inf
    distances += 1e-10

    # r_k is the distance to the k-th neighbor (last column)
    # r_j are the distances to the 1st... (k-1)-th neighbors
    r_k = distances[:, -1]
    r_j = distances[:, :-1]
    
    # Formula: d = 1 / [ (1/(k-1)) * sum( log(r_k / r_j) ) ]
    # Note: r_k is broadcasted across columns of r_j
    log_ratios = np.log(r_k[:, np.newaxis] / r_j)
    
    # Sum across neighbors (axis=1)
    sum_log_ratios = np.sum(log_ratios, axis=1)
    
    # Avoid division by zero if all points are identical
    inv_dim_estimates = (k - 1) / (sum_log_ratios + 1e-10)

    return np.mean(inv_dim_estimates)


# ==========================================
# 2. Two-NN (Facco et al.) - Corrected Math
# ==========================================
def two_nn_dimensionality(data: np.ndarray) -> float:
    """
    Estimate intrinsic dimensionality using Two-NN.
    Corrects the regression target to -log(1 - F(mu)).
    """
    n_samples = data.shape[0]
    if n_samples < 3: return 0.0

    # We only need the first 2 neighbors (plus self)
    nbrs = NearestNeighbors(n_neighbors=3).fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Drop self-distance
    distances = distances[:, 1:] + 1e-10  # Stability fix

    r1 = distances[:, 0]
    r2 = distances[:, 1]

    # Calculate mu = r2 / r1
    mu = r2 / r1
    
    # Sort mu for CDF estimation
    mu_sorted = np.sort(mu)

    # Empirical CDF F(mu) = i / N
    # We use range(1, N+1) for i
    i = np.arange(1, n_samples + 1)
    
    # MATH FIX:
    # Theoretical CDF: F(mu) = 1 - mu^(-d)
    # Linearized: -ln(1 - F(mu)) = d * ln(mu)
    # Y-axis: -ln(1 - i/N)
    # X-axis: ln(mu)
    
    # Problem: If i=N, 1 - i/N = 0, and log(0) fails.
    # Standard fix: discard the last data point (i=N) or use N+1 in denominator.
    # Facco's reference implementation often drops the last point.
    
    mu_fit = mu_sorted[:-1]
    i_fit = i[:-1]
    
    x = np.log(mu_fit)
    y = -np.log(1 - (i_fit / n_samples))

    # Linear regression through the origin is often preferred for 2NN,
    # but a standard least squares slope is robust enough.
    # We fit y = d * x. 
    # Solution for slope d = (x . y) / (x . x) (forcing intercept to 0)
    d = np.dot(x, y) / np.dot(x, x)

    return d


# ==========================================
# 3. Box-Counting - Optimized
# ==========================================
def box_counting_dimensionality(data: np.ndarray, box_sizes: np.ndarray = None) -> float:
    """
    Estimate Box-Counting Dimension.
    Optimized loop and bounds calculation.
    """
    if box_sizes is None:
        # Auto-generate logarithmic box sizes if none provided
        range_max = np.max(data) - np.min(data)
        box_sizes = np.geomspace(range_max / 100, range_max / 5, num=10)

    counts = []
    
    # Optimization: Compute bounds once outside the loop
    min_bounds = np.min(data, axis=0)
    
    for box_size in box_sizes:
        # Floor division to get grid coordinates
        # (data - min) / size -> integer coordinate
        box_indices = np.floor((data - min_bounds) / box_size).astype(int)
        
        # Use a set of tuples to count unique occupied boxes efficiently
        # map(tuple, ...) is faster than list comp for large arrays
        unique_boxes = len(set(map(tuple, box_indices)))
        counts.append(unique_boxes)

    log_box_sizes = np.log(box_sizes)
    log_counts = np.log(counts)

    # Fit line: log(N) = -d * log(epsilon) + C
    # Slope is -d
    A = np.vstack([log_box_sizes, np.ones(len(log_box_sizes))]).T
    slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]

    return -slope