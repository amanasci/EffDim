from typing import Any, Dict, List, Union

import numpy as np

from effdim._native import compute_dim as _native_compute_dim


def compute_dim(data: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, Any]:
    """
    Compute the effective dimensionality of the given data using the specified method.

    Parameters:
    -----------
    data : Union[np.ndarray, List[np.ndarray]]
        Input data. Can be a single numpy array or a list of numpy arrays.
    Returns: dict
        A dictionary containing the results of the effective dimensionality computation.
    """
    # Getting the data and then converting to numpy array if it's a list
    if isinstance(data, list):
        data = np.vstack(data)
    elif not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array or a list of numpy arrays.")

    return dict(_native_compute_dim(np.asarray(data, dtype=np.float64)))
