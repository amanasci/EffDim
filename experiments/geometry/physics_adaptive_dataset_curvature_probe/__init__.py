"""Adaptive per-dataset curvature–physics-label probe.

Geometry ranges are chosen from embeddings only, then frozen before any
label association. Discovery is the completed ViT-B / mag_r_desi rank sweep.
"""

from .pipeline import AdaptiveProbeConfig

__all__ = ["AdaptiveProbeConfig"]
