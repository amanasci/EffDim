# API Reference

## Primary public API

The supported public surface is `effdim.compute_dim` (plus `__version__`). Call `compute_dim` to run the full estimator suite and receive a flat results dict. Geometry and metrics modules below are **compatibility shims**: they keep import paths for advanced callers but are backed by Rust via `effdim._native` and are **not** listed in `__all__`.

## Main Interface

::: effdim.api
    options:
      members:
        - compute_dim

## Metrics (Spectral) — Rust-backed shim

`effdim.metrics` re-exports spectral helpers as thin wrappers over `_native`. Prefer `compute_dim` for the supported public path.

::: effdim.metrics
    options:
      heading_level: 3

## Geometry (Spatial) — Rust-backed shim

`effdim.geometry` re-exports neighborhood / manifold estimators as thin wrappers over `_native`. Prefer `compute_dim` for the supported public path.

::: effdim.geometry
    options:
      heading_level: 3
