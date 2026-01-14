import numpy as np

def _normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Normalizes the spectrum to a probability distribution."""
    # Ensure non-negative just in case
    s = np.abs(spectrum)
    total = np.sum(s)
    if total == 0:
        return np.ones_like(s) / len(s) # Uniform if all zero
    return s / total

def pca_explained_variance(spectrum: np.ndarray, threshold: float = 0.95) -> float:
    """
    Returns the number of components needed to explain `threshold` fraction of variance.
    Note: For singular values s, variance is proportional to s^2.
    """
    # If input is singular values (from SVD), eigenvalues are s^2.
    # If input is eigenvalues (from Covariance), they are variance already.
    # The adapter returns singular values for data matrix, eigenvalues for covariance.
    # This ambiguity needs handling.
    # Assumption for v0.1: The 'spectrum' passed here is assumed to be the "importance" metric directly.
    # HOWEVER, standard PCA explained variance is on Eigenvalues of Covariance (s^2 / (N-1)).
    # If users pass singular values (s), we should square them to get variance.
    # But if they pass eigenvalues, we shouldn't. 
    # Let's assume the adapter output 's' or 'vals' is the "magnitude of the mode".
    # For now, I will treat them as "magnitudes". If they are singular values, energy is s^2.
    # If they are eigenvalues of covariance, energy is lambda.
    # This is a tricky design point. 
    # DECISION: I will assume the input to these functions is strictly the "Eigenvalues of the Correlation/Covariance Matrix" or equivalent "Power".
    # So if we had singular values s, we should convert to s^2 before calling this if we want "Explained Variance".
    # BUT, to keep it simple, I will modify `adapters.py` later to always return "Power/Variance" spectrum?
    # No, SVD returns singular values. 
    # Let's add a 'squared' argument or assume the user handles it? 
    # No, ease of use.
    # I'll implement a helper that treats them as singular values by default (squaring them) if they seem to be s-values? 
    # Or I'll just document: "Expects eigenvalues (variance)".
    
    # Actually, for PR and Entropy, we often operate on eigenvalues of covariance matrix.
    # So, I should probably enforce that `components are energies`.
    
    # Let's treat the input `spectrum` as strictly "Variance/Energy" distribution.
    # I will update `adapters.py` to optionally return squared values, or I handle it here.
    # Let's assume they are VARIANCES (Eigenvalues).
    
    total_var = np.sum(spectrum)
    if total_var == 0:
        return 0.0
        
    cumsum = np.cumsum(spectrum)
    # Find index where cumsum >= threshold * total_var
    idx = np.searchsorted(cumsum, threshold * total_var)
    return float(idx + 1)

def participation_ratio(spectrum: np.ndarray) -> float:
    """
    Computes the Participation Ratio (PR).
    PR = (Sum lambda)^2 / Sum (lambda^2)
    
    Ref: Recanatesi et al.
    """
    # PR is usually defined on the eigenvalues of the covariance matrix.
    # If spectrum are these eigenvalues:
    s_sum = np.sum(spectrum)
    s_sq_sum = np.sum(spectrum**2)
    if s_sq_sum == 0:
        return 0.0
    return (s_sum**2) / s_sq_sum

def shannon_effective_dimension(spectrum: np.ndarray) -> float:
    """
    Computes Shannon Effective Dimension: exp(Entropy).
    H = - sum p_i log p_i
    where p_i = lambda_i / sum(lambda)
    """
    p = _normalize_spectrum(spectrum)
    # Filter zeros for log
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def renyi_effective_dimension(spectrum: np.ndarray, alpha: float = 2.0) -> float:
    """
    Computes Rényi Effective Dimension (Generalized).
    For alpha=1 -> Shannon.
    For alpha=2 -> Connected to Participation Ratio?
      R_2 = 1/(1-2) * log(sum p^2) = -log(sum p^2)
      Exp(R_2) = 1 / sum p^2.
      PR = (sum lambda)^2 / sum lambda^2 = 1 / sum (lambda/sum lambda)^2 = 1 / sum p^2.
      So Exp(Renyi_2) is exactly Participation Ratio!
    """
    if alpha == 1:
        return shannon_effective_dimension(spectrum)
        
    p = _normalize_spectrum(spectrum)
    p_alpha = np.sum(p**alpha)
    if p_alpha == 0:
        return 0.0
        
    entropy = (1 / (1 - alpha)) * np.log(p_alpha)
    return np.exp(entropy)

def effective_rank(spectrum: np.ndarray) -> float:
    """
    Computes Effective Rank (Roy & Vetterli, 2007).
    This is effectively the Shannon Effective Dimension of the normalized spectrum.
    Alias for shannon_effective_dimension.
    """
    return shannon_effective_dimension(spectrum)

def geometric_mean_dimension(spectrum: np.ndarray) -> float:
    """
    Computes a dimension based on the ratio of arithmetic mean to geometric mean.
    """
    # Filter strict positives
    s = spectrum[spectrum > 0]
    if len(s) == 0:
        return 0.0
        
    arithmetic = np.mean(s)
    geometric = np.exp(np.mean(np.log(s)))
    
    # This ratio is 1 if all equal (max dim), and small if sparse.
    # Not a standard 'dimension' count scalar like 5.4, but a ratio.
    # However, some define a dimension proxy from it.
    # For now, I'll return the raw ratio as a placeholder or looks for a specific 'Dimension' formula using it.
    # Vardi's "The effective dimension..."?
    # I will just return the ratio for now.
    return arithmetic / geometric if geometric > 0 else 0.0
