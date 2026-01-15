import numpy as np
from scipy import linalg, sparse
from typing import Union

def get_singular_values(data: Union[np.ndarray, sparse.spmatrix], is_spectrum: bool = False, return_variance: bool = False) -> np.ndarray:
    """
    Standardizes input data into Singular Values or Variance Spectrum.
    
    Args:
        data: Input data.
            - (N, D) array: Interpreted as raw data.
            - (N, N) symmetric matrix: Interpreted as Covariance.
            - (D,) array: Interpreted as pre-computed spectrum (if is_spectrum=True).
        is_spectrum: If True, treats input data as a pre-computed spectrum.
                     The function assumes the input matches the requested form (singular or variance)
                     and simply sorts it descending.
        return_variance: If True, returns eigenvalues (variance, s^2).
                         If False, returns singular values (s).
                         Ignored if is_spectrum=True (user responsibility).
    
    Returns:
        np.ndarray: 1D array of spectrum, sorted descending.
    """
    data = np.asarray(data)
    
    if is_spectrum:
        if data.ndim != 1:
            raise ValueError("Input data must be 1-dimensional when is_spectrum=True.")
        # Return sorted descending
        return np.sort(np.abs(data))[::-1]

    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional.")
    
    N, D = data.shape
    
    # Heuristic for Symmetric Matrix (Covariance/Kernel)
    if N == D and np.allclose(data, data.T):
        vals = linalg.eigvalsh(data)
        # Eigenvalues of Covariance are Variance = s^2.
        # We take abs just in case of numerical noise, though cov should be pos def.
        vals = np.abs(vals)

        # Sort descending
        vals = np.sort(vals)[::-1]

        if return_variance:
            return vals
        else:
            return np.sqrt(vals)
        
    # (N, D) Data Matrix -> SVD
    # For large matrices, this is slow. v0.2 will add randomized SVD.
    _, s, _ = linalg.svd(data, full_matrices=False)
    # s is already sorted descending by linalg.svd

    if return_variance:
        return s**2

    return s
