# API Reference

## Main Interface

::: effdim.api
    options:
      members:
        - compute_dim

## Metrics (Spectral)

::: effdim.metrics
    options:
      heading_level: 3

## Geometry (Spatial)

!!! info "Rust Acceleration"
    The geometry functions (`mle_dimensionality`, `two_nn_dimensionality`, `box_counting_dimensionality`) automatically use a high-performance Rust implementation when available, providing 10-50x speedup for large datasets.

::: effdim.geometry
    options:
      heading_level: 3
