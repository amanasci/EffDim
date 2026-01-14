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
    'effective_rank': {'func': metrics.effective_rank, 'input_type': 'singular'},
    'geometric_mean': {'func': metrics.geometric_mean_dimension, 'input_type': 'singular'},
    # Geometric
    'knn': {'func': geometry.knn_intrinsic_dimension, 'input_type': 'geometric'},
    'twonn': {'func': geometry.two_nn_intrinsic_dimension, 'input_type': 'geometric'},
    # Aliases
    'erank': {'func': metrics.effective_rank, 'input_type': 'singular'},
    'pr': {'func': metrics.participation_ratio, 'input_type': 'variance'},
    'entropy': {'func': metrics.shannon_effective_dimension, 'input_type': 'variance'},
}

def compute(data: Union[np.ndarray, Any], method: str = 'participation_ratio', **kwargs) -> float:
    """
    Computes effective dimension using the specified method.
    
    Args:
        data: Input data.
        method: Method name.
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
        # Geometric methods need raw data (N, D) or distance matrix
        # For now, geometry.py assumes (N, D) points.
        return config['func'](data, **kwargs)
        
    # Spectral methods need singular values
    s = adapters.get_singular_values(data)
    
    if input_type == 'variance':
        spectrum = s**2
    else: # 'singular'
        spectrum = s
        
    return config['func'](spectrum, **kwargs)

def analyze(data: Union[np.ndarray, Any], methods: Optional[List[str]] = None, **kwargs) -> Dict[str, float]:
    """
    Computes multiple effective dimension metrics.
    
    Args:
        data: Input data.
        methods: List of methods to compute. Defaults to generic set.
        **kwargs: Shared kwargs (e.g. threshold=0.95). 
                  Note: Specific kwargs for specific methods not easily supported in this simple API.
    
    Returns:
        Dict[str, float]: Dictionary of results.
    """
    if methods is None:
        methods = ['participation_ratio', 'shannon', 'effective_rank']
        
    results = {}
    
    # Cache singular values to avoid re-computing SVD for each method
    # But compute() calls adapters.get_singular_values() every time.
    # Optimization: We should split the logic.
    # For now, simplistic approach is fine. For large data, we should optimize.
    # Let's optimize:
    
    # Optimize: Compute valid inputs once
    s = None
    s_sq = None
    
    # Check if we need spectral computation at all
    needs_spectral = False
    for m in methods:
        m_cleaned = m.lower()
        if m_cleaned == 'pr': m_cleaned = 'participation_ratio'
        if m_cleaned == 'entropy': m_cleaned = 'shannon'
        
        cfg = METHOD_CONFIG.get(m_cleaned)
        if cfg and cfg['input_type'] in ['variance', 'singular']:
            needs_spectral = True
            break
            
    if needs_spectral:
        s = adapters.get_singular_values(data)
        s_sq = s**2
    
    for method_name in methods:
        orig_name = method_name
        method_name = method_name.lower()
        
        config = METHOD_CONFIG.get(method_name)
        if not config:
            # Check aliases in METHOD_CONFIG directly or manual map above?
            # Creating a standard clean name helper would be better but keeping it simple.
            # METHOD_CONFIG now has aliases.
            if method_name not in METHOD_CONFIG:
                 results[orig_name] = np.nan
                 continue
            config = METHOD_CONFIG[method_name] # Retrieve again if needed
        
        input_type = config['input_type']
        
        if input_type == 'geometric':
             val = config['func'](data, **kwargs)
        elif input_type == 'variance':
            val = config['func'](s_sq, **kwargs)
        else: # singular
            val = config['func'](s, **kwargs)
            
        results[orig_name] = val
        
    return results
