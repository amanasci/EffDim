import numpy as np
import math
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, gammaln
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
from scipy.stats import linregress

# ESS Constants (d=1 to 20, precomputed via 1e6 MC samples)
ESS_CONSTANTS = {
    1: {'E_V': 4.9940264850e-01, 'gamma': 9.1826394100},
    2: {'E_V': 1.4156819765e-01, 'gamma': 11.4195927240},
    3: {'E_V': 2.7610143551e-02, 'gamma': 13.0765130245},
    4: {'E_V': 4.0979287111e-03, 'gamma': 16.4555474529},
    5: {'E_V': 4.8969753325e-04, 'gamma': 15.3029841967},
    6: {'E_V': 4.9059175490e-05, 'gamma': 32.8464539856},
    7: {'E_V': 4.2127213066e-06, 'gamma': 15.8565967980},
    8: {'E_V': 3.1788635229e-07, 'gamma': 19.1386016239},
    9: {'E_V': 2.1337511251e-08, 'gamma': 24.1796122089},
    10: {'E_V': 1.2886580774e-09, 'gamma': 25.5996648416},
    11: {'E_V': 7.0860357322e-11, 'gamma': 22.4336720053},
    12: {'E_V': 3.5754276588e-12, 'gamma': 29.5277927056},
    13: {'E_V': 1.6620116517e-13, 'gamma': 27.5995509848},
    14: {'E_V': 7.1824775576e-15, 'gamma': 27.5019171119},
    15: {'E_V': 2.8966793941e-16, 'gamma': 26.3281790520},
    16: {'E_V': 1.0987080289e-17, 'gamma': 23.7684429769},
    17: {'E_V': 3.9106074114e-19, 'gamma': 23.0922087937},
    18: {'E_V': 1.3159935663e-20, 'gamma': 29.3806008432},
    19: {'E_V': 4.2016845618e-22, 'gamma': 25.4309704320},
    20: {'E_V': 1.2720637766e-23, 'gamma': 31.4301426372},
}

def knn_intrinsic_dimension(data: np.ndarray, k: int = 5) -> float:
    """Computes Intrinsic Dimension using Levina-Bickel MLE."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Data must be 2D array (N, D).")
    N = data.shape[0]
    if N < k + 1:
        raise ValueError(f"Not enough samples ({N}) for k={k}.")

    tree = cKDTree(data)
    dists, _ = tree.query(data, k=k+1)
    neighbors_dists = dists[:, 1:]
    T_k = neighbors_dists[:, -1]
    T_j = neighbors_dists[:, :-1]
    
    epsilon = 1e-10
    T_k = np.maximum(T_k, epsilon)
    T_j = np.maximum(T_j, epsilon)
    
    log_sum = np.sum(np.log(T_k[:, None]) - np.log(T_j))
    estimator = (N * (k - 1)) / log_sum
    return float(estimator)

def two_nn_intrinsic_dimension(data: np.ndarray) -> float:
    """Computes ID using Two-NN method (Facco et al., 2017)."""
    data = np.asarray(data)
    N = data.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points for Two-NN.")

    tree = cKDTree(data)
    dists, _ = tree.query(data, k=3)
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

def danco_intrinsic_dimension(data: np.ndarray, k: int = 20, d_max: int = None) -> float:
    """
    DANCo (Dimensionality from Angle and Norm Concentration).
    
    Args:
        data: (N, D) array.
        k: Neighborhood size. Defaults to min(20, N-1).
        d_max: Max dimension to scan. Defaults to D.
        
    Returns:
        float: Estimated dimension.
    """
    data = np.asarray(data)
    N, D = data.shape

    # Defaults and Validation
    if k is None:
        k = min(20, N - 1)
    if d_max is None:
        d_max = D

    if k < 5:
        raise ValueError(f"k must be >= 5 (got {k}).")
    if k >= N:
        raise ValueError(f"k must be < N (got {k}, N={N}).")

    tree = cKDTree(data)
    dists, indices = tree.query(data, k=k+1)

    # Neighbors (excluding self)
    # indices: (N, k+1). Column 0 is self.
    # dists: (N, k+1).
    nbr_indices = indices[:, 1:] # (N, k)
    nbr_dists = dists[:, 1:]     # (N, k)

    # Step 1: Vectors and Normalization
    # We need vectors v_{i,j} = x_{i,j} - x_i
    # Doing this in a loop to save memory (N*k*D can be large)

    # Precompute empirical distributions?
    # Norms:
    # r_{i,j} are nbr_dists.
    # Normalize by max in neighborhood: r_hat = r_{i,j} / r_{i,k}
    r_max = nbr_dists[:, -1][:, None] # (N, 1)
    r_max = np.maximum(r_max, 1e-10)
    r_hat = nbr_dists / r_max # (N, k)
    r_hat_flat = r_hat.flatten()
    r_hat_flat = r_hat_flat[r_hat_flat > 0] # Avoid exactly 0
    r_hat_flat = np.minimum(r_hat_flat, 1.0) # Clip

    # Angles:
    # Need pairwise angles between neighbors of i.
    # For each i, neighbors j=1..k. Pairwise (j, l).
    # This is heavy. Random subsample of angles?
    # "Compute ALL pairwise angles".
    # For k=20, 190 pairs per point. N=1000 -> 190,000 angles. Feasible.

    n_pairs = k * (k - 1) // 2

    # Ensure float dtype for angles
    dtype = data.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    all_angles = np.empty(N * n_pairs, dtype=dtype)

    # Indices of upper triangle
    tri_u_idx = np.triu_indices(k, k=1)

    # Process in batches to vectorize operations while keeping memory usage reasonable
    BATCH_SIZE = 1000

    for start_idx in range(0, N, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, N)
        # Current batch size (might be less than BATCH_SIZE for last batch)
        # b_sz = end_idx - start_idx

        # 1. Get neighbors for this batch
        # nbr_indices: (N, k) -> batch slice: (B, k)
        batch_nbrs_idx = nbr_indices[start_idx:end_idx]

        # 2. Get neighbor coordinates
        # data: (N, D). batch_nbrs: (B, k, D)
        batch_nbrs = data[batch_nbrs_idx]

        # 3. Get center point coordinates
        # batch_centers: (B, D) -> reshape to (B, 1, D) for broadcasting
        batch_centers = data[start_idx:end_idx][:, None, :]

        # 4. Compute vectors: neighbors - center
        # vecs: (B, k, D)
        vecs = batch_nbrs - batch_centers

        # 5. Normalize vectors
        # norms: (B, k, 1)
        norms = np.linalg.norm(vecs, axis=2, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs_hat = vecs / norms

        # 6. Compute pairwise dot products for each point in batch
        # We need (k, k) matrix for each item in batch.
        # vecs_hat: (B, k, D).
        # We want result (B, k, k) where res[b, i, j] = dot(vecs_hat[b, i], vecs_hat[b, j])
        # This is batch matrix multiplication: (B, k, D) @ (B, D, k)
        dots = np.matmul(vecs_hat, vecs_hat.transpose(0, 2, 1))

        # 7. Extract upper triangle angles
        # dots: (B, k, k).
        # We select specific indices from last two dimensions.
        # tri_u_idx is tuple of arrays (idx0, idx1).
        # We need to broadcast across batch dimension.
        # Result should be (B, n_pairs).
        batch_dots_tri = dots[:, tri_u_idx[0], tri_u_idx[1]]

        # 8. Compute angles
        batch_angles = np.arccos(np.clip(batch_dots_tri, -1.0, 1.0))

        # 9. Store in pre-allocated array
        start_out = start_idx * n_pairs
        end_out = end_idx * n_pairs
        all_angles[start_out:end_out] = batch_angles.ravel()

    # Binning for KL
    n_bins = 20

    # Empirical Histograms
    p_norm_emp, _ = np.histogram(r_hat_flat, bins=n_bins, range=(0, 1), density=True)
    p_angle_emp, _ = np.histogram(all_angles, bins=n_bins, range=(0, np.pi), density=True)

    # Avoid zeros
    epsilon = 1e-10
    p_norm_emp = np.maximum(p_norm_emp, epsilon)
    p_angle_emp = np.maximum(p_angle_emp, epsilon)

    # Bin centers
    r_centers = np.linspace(0, 1, n_bins+1)[:-1] + 0.5/n_bins
    theta_centers = np.linspace(0, np.pi, n_bins+1)[:-1] + 0.5*(np.pi/n_bins)

    def kl_divergence(p, q):
        return np.sum(p * np.log(p / q)) * (1/n_bins) # integration approx? No, if density=True, sum(p*dx)=1.
        # Here p, q are densities. KL = integral p log(p/q) dx.
        # dx is 1/n_bins (for norm) or pi/n_bins (for angle).
        # But simple sum check:
        # KL approx sum(p[i] * log(p[i]/q[i])) * dx

    dx_norm = 1.0 / n_bins
    dx_angle = np.pi / n_bins

    best_d = 1
    min_kl = float('inf')

    for d in range(1, d_max + 1):
        # Theoretical PDFs

        # Norm: d * r^(d-1)
        p_norm_theo = d * (r_centers ** (d - 1))

        # Angle: C_d * (sin theta)^(d-2)
        if d == 1:
            # For d=1, angles are 0 or pi. Continuous PDF is ill-defined.
            # We use a dummy uniform distribution but this will yield high KL
            # if empirical data is continuous (which it is for embedded d=1 manifolds usually).
            # If d=1 candidate is tested against D=3 data, empirical angles are in (0, pi).
            # Uniform theo (1/pi) vs Peaked empirical?
            # We'll use 1/pi.
            p_angle_theo = np.full_like(theta_centers, 1.0 / np.pi)
        else:
            c_d = (1.0 / np.sqrt(np.pi)) * gamma(d/2) / gamma((d-1)/2)
            p_angle_theo = c_d * (np.sin(theta_centers) ** (d - 2))

        p_norm_theo = np.maximum(p_norm_theo, epsilon)
        p_angle_theo = np.maximum(p_angle_theo, epsilon)

        # KL
        kl_norm = np.sum(p_norm_emp * np.log(p_norm_emp / p_norm_theo)) * dx_norm
        kl_angle = np.sum(p_angle_emp * np.log(p_angle_emp / p_angle_theo)) * dx_angle

        total_kl = kl_norm + kl_angle

        if total_kl < min_kl:
            min_kl = total_kl
            best_d = d

    return float(best_d)

def mind_mli_intrinsic_dimension(data: np.ndarray, k: int = 20) -> float:
    """
    MiND-MLi (Minimum Neighbor Distance).
    Corrected version using Hill estimator on nearest neighbor distance.
    """
    data = np.asarray(data)
    N = data.shape[0]

    # k is just for API compatibility? No, MiND-MLi is k=1 implicitly.
    # But usually we compute NN.

    tree = cKDTree(data)
    dists, _ = tree.query(data, k=2) # Self and 1st NN
    r_i = dists[:, 1] # Nearest neighbor distance

    epsilon = 1e-10
    r_i = np.maximum(r_i, epsilon)

    r_min = np.min(r_i)
    if r_min < epsilon:
        r_min = epsilon

    # Formula: d = ( 1/N * sum log(r_i / r_min) )^-1
    # Note: sum is strictly positive since r_i >= r_min.

    log_sum = np.sum(np.log(r_i / r_min))
    if log_sum == 0:
        return 0.0 # All distances equal?
        
    d_hat = N / log_sum
    return float(d_hat)

def mind_mlk_intrinsic_dimension(data: np.ndarray, k: int = 20) -> float:
    """
    MiND-MLk.
    Uses k-th nearest neighbor distances.
    """
    data = np.asarray(data)
    N = data.shape[0]

    if k < 1:
        raise ValueError("k must be >= 1.")

    tree = cKDTree(data)
    dists, _ = tree.query(data, k=k+1) # Self + k neighbors
    nbr_dists = dists[:, 1:] # (N, k)
    
    epsilon = 1e-10
    nbr_dists = np.maximum(nbr_dists, epsilon)
    
    r_ik = nbr_dists[:, -1] # (N,) - The k-th neighbor
    
    # Formula: d = ( 1/(Nk) * sum_i sum_j log(r_ik / r_ij) )^-1
    # r_ij are columns 0 to k-1.
    # log(r_ik / r_ij) = log(r_ik) - log(r_ij)
    # This is positive since r_ik >= r_ij.
    
    log_diffs = np.log(r_ik[:, None]) - np.log(nbr_dists)
    total_sum = np.sum(log_diffs)
    
    if total_sum == 0:
        return 0.0
        
    d_hat = (N * k) / total_sum
    return float(d_hat)

def ess_intrinsic_dimension(data: np.ndarray, d_max: int = None) -> float:
    """
    ESS (Expected Simplex Skewness).
    """
    data = np.asarray(data)
    N, D = data.shape

    if d_max is None:
        d_max = D
        
    # Cap d_max to available constants
    max_const = max(ESS_CONSTANTS.keys())
    d_scan_max = min(d_max, max_const)

    best_d = 1
    min_diff = float('inf')

    # We need to compute empirical skewness for each candidate d
    # Candidate d needs k = d + 1 neighbors.
    # We can query max needed neighbors once.
    k_max = d_scan_max + 1
    if N < k_max + 1:
        # Cannot compute for high d
        k_max = N - 1
        d_scan_max = k_max - 1
        if d_scan_max < 1:
            raise ValueError(f"Not enough samples ({N}) for ESS.")

    tree = cKDTree(data)
    dists, indices = tree.query(data, k=k_max+1)
    # indices: (N, k_max+1)

    for d in range(1, d_scan_max + 1):
        k = d + 1

        # Get neighbors for this d
        # We need vectors from i to neighbors.
        # Simplex vertices: {0, v_1, ..., v_d}.
        # Wait, the prompt said: "Choose k = d+1 neighbors. Form simplex with vectors v_1 ... v_d".
        # If we have k neighbors, we have vectors to them?
        # Usually ESS uses vectors to neighbors *minus* the point i.
        # "Form simplex with vectors v_1 ... v_d".
        # This implies we use d neighbors. But we selected k=d+1?
        # Maybe the point i itself is v_0=0?
        # Ref Johnsson 2015: "Consider a point x and its k nearest neighbors... form the simplex S spanned by the k vectors connecting x to its neighbors."
        # If simplex is in d dimensions, we need d vectors.
        # So we use d nearest neighbors.
        # The prompt says: "Choose k = d + 1 neighbors".
        # This might mean: Use neighbors 1 to d. (Total d).
        # Or use neighbors 1 to d+1? (Total d+1 vectors).
        # If we have d+1 vectors in R^D, and we project to tangent space?
        # "Volume: V_i = 1/d! |det([v_1 ... v_d])|".
        # This explicitly uses d vectors.
        # Why choose k=d+1? Maybe for robustness or the simplex uses d+1 points (0 + d neighbors)?
        # I will use the first d nearest neighbors to form the d vectors.

        nbr_idx = indices[:, 1:d+1] # (N, d)

        # Compute volumes
        # Batch volume computation?
        # V_i = 1/d! * sqrt(det(V.T @ V))

        # Construct V matrices for all i
        # This loop is heavy.

        vols = []
        fact_d = math.factorial(d)

        # Optimization: process in batches or loop
        for i in range(N):
            n_idxs = nbr_idx[i]
            vecs = data[n_idxs] - data[i] # (d, D)

            # ESS requires vectors in unit ball for comparison with constants.
            # Neighbors are in a ball of radius r_{i,d} (distance to d-th neighbor).
            # Normalize by this radius to map to unit ball.
            # The vectors are v_1 ... v_d. The d-th neighbor is the furthest.
            # Its distance is norm(vecs[-1]).

            max_r = np.linalg.norm(vecs[-1])
            max_r = max(max_r, 1e-10)

            vecs = vecs / max_r

            # V is (d, D). V V^T is (d, d) Gram matrix.
            gram = vecs @ vecs.T
            det = np.linalg.det(gram)
            if det < 0: det = 0
            vol = (1.0 / fact_d) * np.sqrt(det)
            vols.append(vol)

        vols = np.array(vols)
        vols = np.maximum(vols, 1e-20)

        gamma_emp = np.mean(ESS_CONSTANTS[d]['E_V'] / vols)
        gamma_theo = ESS_CONSTANTS[d]['gamma']

        diff = abs(gamma_emp - gamma_theo)
        if diff < min_diff:
            min_diff = diff
            best_d = d

    return float(best_d)

def tle_intrinsic_dimension(data: np.ndarray, k: int = 20, d_max: int = None) -> float:
    """
    TLE (Tight Localities Estimator).
    """
    data = np.asarray(data)
    N, D = data.shape

    if d_max is None:
        d_max = D

    if k < 2:
        raise ValueError("k must be >= 2 for TLE.")

    tree = cKDTree(data)
    dists, _ = tree.query(data, k=k+1)
    nbr_dists = dists[:, 1:] # (N, k)

    epsilon = 1e-10
    nbr_dists = np.maximum(nbr_dists, epsilon)

    r_ik = nbr_dists[:, -1] # (N,)

    # Scale invariance: Normalize by r_{i,k}
    # r_hat = r / r_k
    # This ensures TLE does not depend on unit scale.
    # The formula provided: - d log r_{i,k} + (d-1) sum log r_{i,j}
    # With normalized dists, r_{i,k} becomes 1, log is 0.
    # r_{i,j} becomes r_{i,j}/r_{i,k}.

    # Normalize distances
    r_ik_col = r_ik[:, None]
    nbr_dists_norm = nbr_dists / np.maximum(r_ik_col, 1e-10)

    sum_log_rij = np.sum(np.log(nbr_dists_norm + 1e-20), axis=1) # (N,)

    # With normalized distances:
    # L_i(d) = k log d - d * log(1) + (d-1) sum log(r_j/r_k)
    #        = k log d + (d-1) sum log(r_j/r_k)

    S2 = np.sum(sum_log_rij)

    best_d = 1
    max_lik = -float('inf')

    for d in range(1, d_max + 1):
        # L = N*k*log(d) + (d-1)*S2
        # Note: S2 is sum of log ratios (negative).
        ll = N * k * np.log(d) + (d - 1) * S2

        if ll > max_lik:
            max_lik = ll
            best_d = d

    return float(best_d)

def gmst_intrinsic_dimension(data: np.ndarray, k: int = 5, mode: str = 'euclidean') -> float:
    """
    GMST (Geodesic Minimum Spanning Tree).

    Args:
        data: (N, D) array.
        k: k for k-NN graph (only used if mode='geodesic').
        mode: 'euclidean' (default) or 'geodesic'.
    """
    data = np.asarray(data)
    N = data.shape[0]

    if N < 20:
        raise ValueError(f"Not enough samples ({N}) for GMST. Need >= 20.")

    geo_dists = None

    if mode == 'geodesic':
        # 1. Build k-NN graph (weighted Euclidean)
        tree = cKDTree(data)
        dists, indices = tree.query(data, k=k+1)

        row_ind = np.repeat(np.arange(N), k)
        col_ind = indices[:, 1:].flatten()
        data_weights = dists[:, 1:].flatten()

        adj = csr_matrix((data_weights, (row_ind, col_ind)), shape=(N, N))
        adj = adj + adj.T

        # 2. Geodesic distances (Dense NxN)
        geo_dists = shortest_path(adj, directed=False, return_predecessors=False)
        if np.isinf(geo_dists).any():
            pass

    # Subsampling
    start_n = int(0.2 * N)
    n_values = np.unique(np.linspace(start_n, N, num=10, dtype=int))
    n_values = n_values[n_values >= start_n]
    if len(n_values) < 5:
        n_values = np.unique(np.linspace(start_n, N, num=max(5, (N-start_n)//2), dtype=int))

    lengths = []
    final_ns = []

    for n in n_values:
        if n < 2: continue

        idx = np.random.choice(N, n, replace=False)

        if mode == 'geodesic':
             # Use precomputed geodesic matrix
             sub_dists = geo_dists[np.ix_(idx, idx)]
             mst = minimum_spanning_tree(sub_dists)
             L_n = mst.data.sum()
        else: # euclidean
             # Compute Euclidean MST on subsample
             sub_data = data[idx]
             # For N up to 1000, pdist is fine (1000*999/2 = 500k floats).
             dmat = squareform(pdist(sub_data))
             mst = minimum_spanning_tree(dmat)
             L_n = mst.data.sum()

        lengths.append(L_n)
        final_ns.append(n)

    # Regression
    # log(L) = a + b * log(N)
    # b = (d-1)/d -> bd = d-1 -> d(1-b) = 1 -> d = 1/(1-b)

    log_n = np.log(final_ns)
    log_l = np.log(lengths)

    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_l)

    b = slope
    if b >= 1.0:
        # Singularity.
        return float(D) # Fallback?

    d_hat = 1.0 / (1.0 - b)
    return float(d_hat)
