from typing import Union, List, Dict, Any, Optional
import numpy as np
from . import adapters
from . import metrics
from . import geometry

# Map method names to function calls and their expected input type
# 'variance': pass s**2 (eigenvalues of covariance)
# 'singular': pass s
# 'geometric': pass raw data (N, D)
METHOD_CONFIG = {
    'pca': {'func': metrics.pca_explained_variance, 'input_type': 'variance'},
    'participation_ratio': {'func': metrics.participation_ratio, 'input_type': 'variance'},
    'shannon': {'func': metrics.shannon_effective_dimension, 'input_type': 'variance'},
    'renyi': {'func': metrics.renyi_effective_dimension, 'input_type': 'variance'},
    'effective_rank': {'func': metrics.effective_rank, 'input_type': 'variance'},
    'geometric_mean': {'func': metrics.geometric_mean_dimension, 'input_type': 'singular'},

    # New Spectral
    'stable_rank': {'func': metrics.stable_rank, 'input_type': 'variance'},
    'regularized': {'func': metrics.regularized_trace_ratio, 'input_type': 'variance'},

    # Geometric
    'knn': {'func': geometry.knn_intrinsic_dimension, 'input_type': 'geometric'},
    'twonn': {'func': geometry.two_nn_intrinsic_dimension, 'input_type': 'geometric'},
    'lid': {'func': geometry.lid_intrinsic_dimension, 'input_type': 'geometric'},
    'correlation': {'func': geometry.correlation_dimension, 'input_type': 'geometric'},

    # Aliases
    'erank': {'func': metrics.effective_rank, 'input_type': 'variance'},
    'pr': {'func': metrics.participation_ratio, 'input_type': 'variance'},
    'entropy': {'func': metrics.shannon_effective_dimension, 'input_type': 'variance'},
    'mle': {'func': geometry.knn_intrinsic_dimension, 'input_type': 'geometric'},
}

def compute(data: Union[np.ndarray, Any], method: str = 'participation_ratio', is_spectrum: bool = False, **kwargs) -> float:
    """
    Computes effective dimension using the specified method.
    
    Args:
        data: Input data or pre-computed spectrum (if is_spectrum=True).
              If is_spectrum=True, data is assumed to be Singular Values (s).
        method: Method name.
        is_spectrum: Set to True if data is a 1D array of singular values.
        **kwargs: Arguments passed to the estimator.
        
    Returns:
        float: Estimated effective dimension.
    """
    method = method.lower()
    
    config = METHOD_CONFIG.get(method)
    if not config:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHOD_CONFIG.keys())}")
        
    input_type = config['input_type']
    
    # Branching logic for Data
    if input_type == 'geometric':
        if is_spectrum:
            raise ValueError(f"Method '{method}' is geometric and cannot use pre-computed spectrum.")
        return config['func'](data, **kwargs)
        
    # Spectral methods need singular values
    # We use adapters to get 's'. If is_spectrum=True, we assume data IS 's'.
    s = adapters.get_singular_values(data, is_spectrum=is_spectrum, return_variance=False)
    
    if input_type == 'variance':
        spectrum = s**2
    else: # 'singular'
        spectrum = s
        
    return config['func'](spectrum, **kwargs)

def analyze(data: Union[np.ndarray, Any], methods: Optional[List[str]] = None, is_spectrum: bool = False, **kwargs) -> Dict[str, float]:
    """
    Computes multiple effective dimension metrics.
    
    Args:
        data: Input data or pre-computed spectrum (Singular Values) if is_spectrum=True.
        methods: List of methods to compute. Defaults to generic set.
        is_spectrum: Set to True if data is a 1D array of singular values.
        **kwargs: Shared kwargs.
    
    Returns:
        Dict[str, float]: Dictionary of results.
    """
    if methods is None:
        methods = ['participation_ratio', 'shannon', 'effective_rank']
        
    results = {}
    
    s = None
    s_sq = None
    
    # Check if we need spectral computation
    needs_spectral = False
    for m in methods:
        m_cleaned = m.lower()
        cfg = METHOD_CONFIG.get(m_cleaned)
        if cfg and cfg['input_type'] in ['variance', 'singular']:
            needs_spectral = True
            break
            
    if needs_spectral:
        # Compute/Fetch Singular Values once
        s = adapters.get_singular_values(data, is_spectrum=is_spectrum, return_variance=False)
        s_sq = s**2
    
    for method_name in methods:
        orig_name = method_name
        method_name = method_name.lower()
        
        config = METHOD_CONFIG.get(method_name)
        if not config:
             results[orig_name] = np.nan
             continue
        
        input_type = config['input_type']
        
        if input_type == 'geometric':
             if is_spectrum:
                 # Cannot compute geometric on spectrum
                 val = np.nan
             else:
                 val = config['func'](data, **kwargs)
        elif input_type == 'variance':
            val = config['func'](s_sq, **kwargs)
        else: # singular
            val = config['func'](s, **kwargs)
            
        results[orig_name] = val
        
    return results
