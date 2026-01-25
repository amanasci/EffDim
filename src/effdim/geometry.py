import numpy as np
from effdim import _rust


# ==========================================
# 1. MLE (Levina-Bickel) - Robust Version
# ==========================================
def mle_dimensionality(data: np.ndarray, k: int = 10) -> float:
    """
    Estimate intrinsic dimensionality using Levina-Bickel MLE.
    Includes protection against duplicate points (distance=0).
    Uses fast Rust implementation.
    """
    return _rust.mle_dimensionality(data, k)


# ==========================================
# 2. Two-NN (Facco et al.) - Corrected Math
# ==========================================
def two_nn_dimensionality(data: np.ndarray) -> float:
    """
    Estimate intrinsic dimensionality using Two-NN.
    Corrects the regression target to -log(1 - F(mu)).
    Uses fast Rust implementation.
    """
    return _rust.two_nn_dimensionality(data)


# ==========================================
# 3. Box-Counting - Optimized
# ==========================================
def box_counting_dimensionality(data: np.ndarray, box_sizes: np.ndarray = None) -> float:
    """
    Estimate Box-Counting Dimension.
    Optimized loop and bounds calculation.
    Uses fast Rust implementation.
    """
    if box_sizes is None:
        # Auto-generate logarithmic box sizes if none provided
        range_max = np.max(data) - np.min(data)
        box_sizes = np.geomspace(range_max / 100, range_max / 5, num=10)
    
    return _rust.box_counting_dimensionality(data, box_sizes)
