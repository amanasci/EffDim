import numpy as np
from scipy import linalg, sparse
from typing import Union

def get_singular_values(data: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
    """
    Standardizes input data into Singular Values.
    
    Args:
        data: Input data.
            - (N, D) array: Interpreted as raw data. Returns singular values.
            - (N, N) symmetric matrix: Interpreted as Covariance. Returns sqrt(abs(eigenvalues)).
    
    Returns:
        np.ndarray: 1D array of singular values, sorted descending.
    """
    data = np.asarray(data)
    
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional.")
    
    N, D = data.shape
    
    # Heuristic for Symmetric Matrix (Covariance/Kernel)
    if N == D and np.allclose(data, data.T):
        vals = linalg.eigvalsh(data)
        # Eigenvalues of Covariance are Variance = s^2.
        # So s = sqrt(vals).
        # We take abs just in case of numerical noise, though cov should be pos def.
        return np.sqrt(np.abs(vals))[::-1]
        
    # (N, D) Data Matrix -> SVD
    # For large matrices, this is slow. v0.2 will add randomized SVD.
    _, s, _ = linalg.svd(data, full_matrices=False)
    return s
