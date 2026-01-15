import numpy as np

def _normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Normalizes the spectrum to a probability distribution."""
    # Ensure non-negative just in case
    s = np.abs(spectrum)
    total = np.sum(s)
    if total == 0:
        return np.ones_like(s) / len(s) # Uniform if all zero
    return s / total

def pca_explained_variance(spectrum: np.ndarray, threshold: float = 0.95, **kwargs) -> float:
    """
    Returns the number of components needed to explain `threshold` fraction of variance.
    
    Args:
        spectrum: Variance spectrum (eigenvalues of covariance matrix).
        threshold: Variance threshold (0.0 to 1.0).
    """
    total_var = np.sum(spectrum)
    if total_var == 0:
        return 0.0
        
    cumsum = np.cumsum(spectrum)
    # Find index where cumsum >= threshold * total_var
    idx = np.searchsorted(cumsum, threshold * total_var)
    return float(idx + 1)

def participation_ratio(spectrum: np.ndarray, **kwargs) -> float:
    """
    Computes the Participation Ratio (PR).
    PR = (Sum lambda)^2 / Sum (lambda^2)
    
    Args:
        spectrum: Variance spectrum (eigenvalues of covariance matrix).
    """
    s_sum = np.sum(spectrum)
    s_sq_sum = np.sum(spectrum**2)
    if s_sq_sum == 0:
        return 0.0
    return (s_sum**2) / s_sq_sum

def shannon_effective_dimension(spectrum: np.ndarray, **kwargs) -> float:
    """
    Computes Shannon Effective Dimension: exp(Entropy).
    H = - sum p_i log p_i
    where p_i = lambda_i / sum(lambda)

    Args:
        spectrum: Variance spectrum (eigenvalues of covariance matrix).
    """
    p = _normalize_spectrum(spectrum)
    # Filter zeros for log
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def renyi_effective_dimension(spectrum: np.ndarray, alpha: float = 2.0, **kwargs) -> float:
    """
    Computes Rényi Effective Dimension (Generalized).

    Args:
        spectrum: Variance spectrum (eigenvalues of covariance matrix).
        alpha: Order of Rényi entropy.
    """
    if alpha == 1:
        return shannon_effective_dimension(spectrum)
        
    p = _normalize_spectrum(spectrum)
    p_alpha = np.sum(p**alpha)
    if p_alpha == 0:
        return 0.0
        
    entropy = (1 / (1 - alpha)) * np.log(p_alpha)
    return np.exp(entropy)

def effective_rank(spectrum: np.ndarray, **kwargs) -> float:
    """
    Computes Effective Rank (Roy & Vetterli, 2007).
    This is effectively the Shannon Effective Dimension of the normalized spectrum.

    Args:
        spectrum: Variance spectrum (eigenvalues of covariance matrix).
    """
    return shannon_effective_dimension(spectrum)

def geometric_mean_dimension(spectrum: np.ndarray, **kwargs) -> float:
    """
    Computes a dimension based on the ratio of arithmetic mean to geometric mean.

    Args:
        spectrum: Variance spectrum (eigenvalues of covariance matrix).
    """
    # Filter strict positives
    s = spectrum[spectrum > 0]
    if len(s) == 0:
        return 0.0
        
    arithmetic = np.mean(s)
    geometric = np.exp(np.mean(np.log(s)))
    
    return arithmetic / geometric if geometric > 0 else 0.0

def stable_rank(spectrum: np.ndarray, **kwargs) -> float:
    """
    Computes Stable Rank.
    stable_rank = sum(lambda) / max(lambda)

    Args:
        spectrum: Variance spectrum (eigenvalues of covariance matrix).
    """
    s_sum = np.sum(spectrum)
    s_max = np.max(spectrum)

    if s_max == 0:
        return 0.0

    return s_sum / s_max

def regularized_trace_ratio(spectrum: np.ndarray, z: float = 1e-5, **kwargs) -> float:
    """
    Computes Regularized Effective Dimension (Fisher-based / Trace Ratio).
    N_eff = Sum (lambda_i / (lambda_i + z))

    Args:
        spectrum: Variance spectrum (eigenvalues of covariance matrix).
        z: Regularization parameter.
    """
    return np.sum(spectrum / (spectrum + z))
