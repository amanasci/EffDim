import numpy as np
from scipy.spatial import cKDTree

def knn_intrinsic_dimension(data: np.ndarray, k: int = 5) -> float:
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
    
    # dists has shape (N, k+1). Column 0 is distance to self (0).
    # We want neighbors 1 to k.
    # The Levina-Bickel estimator uses distances up to k.
    # Eq: inv( 1/(k-1) * sum_{j=1}^{k-1} log(Tk / Tj) ) ... wait average over N points first?
    # MacKay/Levina-Bickel:
    # For each point i: m_i = [ 1/(k-1) * sum_{j=1}^{k-1} log( T_k(x_i) / T_j(x_i) ) ]^-1
    # Global estimator is the average of m_i? Or average the inverse?
    # Levina-Bickel (2005) "Maximum Likelihood Estimation...":
    # \hat{m}_k = \left[ \frac{1}{N(k-1)} \sum_{i=1}^N \sum_{j=1}^{k-1} \ln \frac{T_k(x_i)}{T_j(x_i)} \right]^{-1}
    # Yes, one global inverse.
    
    # Drop self
    neighbors_dists = dists[:, 1:] # (N, k) - these are 1st to kth neighbors
    
    # T_k is the distance to the k-th neighbor (last column)
    T_k = neighbors_dists[:, -1] # (N,)
    
    # T_j are distances 1 to k-1. (All columns excluding last)
    T_j = neighbors_dists[:, :-1] # (N, k-1)
    
    # Avoid log(0) - unlikely if points distinct, but clear duplicates
    # If T_k or T_j is 0, we have duplicates.
    # Add epsilon? Or filter?
    # Simple fix: non-zero epsilon
    epsilon = 1e-10
    T_k = np.maximum(T_k, epsilon)
    T_j = np.maximum(T_j, epsilon)
    
    # Log ratios: log(T_k / T_j) = log(T_k) - log(T_j)
    # Broadcast T_k: (N, 1) - (N, k-1)
    log_sum = np.sum(np.log(T_k[:, None]) - np.log(T_j)) # Sum over all i, j
    
    # Denominator: N * (k-1) is outside the sum if we sum over everything
    # The formula is 1 / ( (1 / (N*(k-1))) * log_sum )
    # = N*(k-1) / log_sum
    
    # Actually, verify formula carefully.
    # sum_{i=1}^N sum_{j=1}^{k-1} ... is the total sum.
    # The term in bracket is Average of log ratios? No, explicit 1/N(k-1).
    # So MLE = 1 / ( Mean of Log Ratios ).
    
    estimator = (N * (k - 1)) / log_sum
    return float(estimator)

def two_nn_intrinsic_dimension(data: np.ndarray) -> float:
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
    
    # r1 is dists[:, 1], r2 is dists[:, 2]
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    
    # Filter valid ratios (r1 > 0, r2 > 0)
    # If duplicates, r1=0.
    mask = r1 > 1e-10
    r1 = r1[mask]
    r2 = r2[mask]
    
    # mu = r2 / r1
    mu = r2 / r1
    
    # Empirical cdf F(mu).
    # The paper simplifies to a linear fit or directly the formula:
    # d = N / sum_i ln(mu_i).
    # Wait, the paper "Estimating the intrinsic dimension of datasets by a minimal neighborhood information"
    # Section "The Two-NN estimator".
    # "d_hat = N / \sum_{i=1}^N \ln(\mu_i)"
    # This is assuming the distribution is Pareto with alpha=d.
    # Let's use this simple estimator.
    
    if len(mu) == 0:
        return 0.0
        
    log_mu_sum = np.sum(np.log(mu))
    if log_mu_sum == 0:
        return 0.0
        
    d_hat = len(mu) / log_mu_sum
    return float(d_hat)
