"""
EffDim: Unified Effective Dimensionality Library.

A research-oriented Python library for computing effective dimensionality
across diverse data modalities using various established methods.

I guess we can just from one single function for now.

function compute(data) -> dictionary of results



"""

from .api import compute

# Version
__version__ = "0.1.0"

# Public API
__all__ = [
    '__version__',
]
