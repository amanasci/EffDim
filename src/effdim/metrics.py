import numpy as np



def pca_explained_variance(spectrum: np.ndarray, threshold: float = 0.95) -> int:
    """
    Compute the number of principal components required to explain a given
    threshold of variance.

    Parameters:
    -----------
    spectrum : np.ndarray
        Array of eigenvalues (explained variance) from PCA.
    threshold : float
        The cumulative variance threshold to reach (between 0 and 1).

    Returns:
    --------
    int
        Number of principal components needed to reach the threshold.
    """
    total_variance = np.sum(spectrum)
    if total_variance == 0:
        return 1
    cumulative_variance = np.cumsum(spectrum)
    explained_variance_ratio = cumulative_variance / total_variance

    num_components = int(np.searchsorted(explained_variance_ratio, threshold) + 1)
    return num_components

def participation_ratio(spectrum: np.ndarray) -> float:
    """
    Compute the Participation Ratio (PR) of the given spectrum.

    Parameters:
    -----------
    spectrum : np.ndarray
        Array of eigenvalues.

    Returns:
    --------
    float
        Participation Ratio value.
    """
    numerator = (np.sum(spectrum)) ** 2
    denominator = np.sum(spectrum ** 2)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Compute the Shannon Entropy of the given probability distribution.

    Parameters:
    -----------
    probabilities : np.ndarray
        Array of probabilities.

    Returns:
    --------
    float
        Shannon Entropy value.
    """
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log(probabilities))
    d_eff = np.exp(entropy)
    return d_eff
    
def renyi_eff_dimensionality(probabilities: np.ndarray, alpha: float) -> float:
    """
    Compute the Rényi Effective Dimensionality of the given probability distribution.

    Parameters:
    -----------
    probabilities : np.ndarray
        Array of probabilities.
    alpha : float
        Order of the Rényi entropy (alpha > 0 and alpha != 1).

    Returns:
    --------
    float
        Rényi Effective Dimensionality value.
    """
    if alpha <= 0 or alpha == 1:
        raise ValueError("Alpha must be greater than 0 and not equal to 1.")

    sum_probs_alpha = np.sum(probabilities ** alpha)
    if sum_probs_alpha == 0:
        return 0.0

    d_eff = sum_probs_alpha ** (1 / (1 - alpha))
    return d_eff

def geometric_mean_eff_dimensionality(spectrum: np.ndarray) -> float:
    """
    Compute the Geometric Mean Effective Dimensionality of the given spectrum.

    Parameters:
    -----------
    spectrum : np.ndarray
        Array of eigenvalues.

    Returns:
    --------
    float
        Geometric Mean Effective Dimensionality value.
    """
    positive_spectrum = spectrum[spectrum > 0]
    if len(positive_spectrum) == 0:
        return 0.0

    # Calculate the arithmetic mean of the positive spectrum
    am = np.mean(positive_spectrum)
    # Calculate the geometric mean of the positive spectrum
    gm = np.exp(np.mean(np.log(positive_spectrum)))
    d_eff = (am / gm)
    
    return d_eff

def stable_rank(spectrum: np.ndarray) -> float:
    """
    Compute the Stable Rank of the given spectrum.

    Parameters:
    -----------
    spectrum : np.ndarray
        Array of eigenvalues.

    Returns:
    --------
    float
        Stable Rank value.
    """
    if len(spectrum) == 0:
        return 0.0
    max_eig = np.max(spectrum)
    if max_eig == 0:
        return 0.0
    return float(np.sum(spectrum) / max_eig)

def numerical_rank(singular_values: np.ndarray, epsilon: float = None) -> int:
    """
    Compute the Numerical Rank (Epsilon-Rank) of the given singular values.

    Parameters:
    -----------
    singular_values : np.ndarray
        Array of singular values.
    epsilon : float, optional
        Threshold. If None, defaults to machine precision times the largest
        singular value.

    Returns:
    --------
    int
        Numerical rank.
    """
    if len(singular_values) == 0:
        return 0
    if epsilon is None:
        epsilon = float(np.finfo(singular_values.dtype).eps * np.max(singular_values) * len(singular_values))
    return int(np.sum(singular_values > epsilon))

def cumulative_eigenvalue_ratio(probabilities: np.ndarray) -> float:
    """
    Compute the Cumulative Eigenvalue Ratio (CER) of the given spectrum.

    Parameters:
    -----------
    probabilities : np.ndarray
        Array of probabilities (normalized eigenvalues).

    Returns:
    --------
    float
        CER value.
    """
    D = len(probabilities)
    if D == 0:
        return 0.0
    elif D == 1:
        return float(probabilities[0])
    
    weights = (D - 1 - np.arange(D)) / (D - 1)
    return float(np.sum(weights * probabilities))