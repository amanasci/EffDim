# Performance and Rust Implementation

EffDim includes a high-performance Rust implementation of geometry-based dimensionality estimators, providing significant speedups for large datasets.

## Overview

The Rust implementation uses:

- **Parallel brute-force nearest neighbor search** with [rayon](https://github.com/rayon-rs/rayon)
- **Optimized for high-dimensional data** (100-1000+ dimensions)
- **Multi-core parallelization** for maximum performance

## Performance Benchmarks

Benchmark results on GitHub Actions runners (4 CPU cores):

| Samples | Dimensions | MLE Time (Rust) | Two-NN Time (Rust) | Speedup vs Python |
|---------|------------|-----------------|---------------------|-------------------|
| 1,000   | 100        | 0.05s          | 0.05s               | ~10x             |
| 5,000   | 200        | 2.5s           | 2.5s                | ~30x             |
| 10,000  | 700        | 36s            | 36s                 | ~50x             |

!!! info "Scaling"
    Performance scales roughly linearly with sample count and quadratically with the number of nearest neighbors (k).

## Why Brute-Force?

Traditional k-d trees perform poorly in high dimensions due to the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). For data with 100+ dimensions:

- **K-d trees**: Performance degrades to O(n²) anyway
- **Brute-force + SIMD + Parallelization**: Consistent O(n²) but with much better constants
- **Result**: Brute-force is faster for high-dimensional data

## Affected Functions

The following functions use the Rust implementation:

- `geometry.mle_dimensionality()` - MLE (Levina-Bickel) estimator
- `geometry.two_nn_dimensionality()` - Two-NN (Facco et al.) estimator
- `geometry.box_counting_dimensionality()` - Box-counting dimension

## Installation

### Prebuilt Wheels (Recommended)

```bash
pip install effdim
```

The Rust extension is automatically included in prebuilt wheels for Linux, macOS, and Windows.

### Building from Source

If prebuilt wheels aren't available for your platform:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install
maturin build --release
pip install target/wheels/effdim-*.whl
```

See the [repository README](https://github.com/amanasci/EffDim/blob/main/RUST_BUILD.md) for detailed build instructions.

## CPU Core Utilization

The Rust implementation automatically uses all available CPU cores. Performance scales with:

- **Number of cores**: Near-linear scaling up to ~8 cores
- **CPU speed**: Single-threaded performance matters
- **Cache size**: Larger L2/L3 cache helps with large datasets

!!! tip "Performance Tips"
    - Use release builds (`maturin build --release`) for maximum performance
    - Ensure your system has sufficient RAM (dataset size × 8 bytes)
    - For very large datasets (1M+ samples), consider batch processing

## Technical Details

### Dependencies

The Rust implementation uses:

- **[PyO3](https://pyo3.rs/)** (0.23) - Python bindings
- **[numpy](https://github.com/PyO3/rust-numpy)** (0.23) - NumPy array interop
- **[ndarray](https://github.com/rust-ndarray/ndarray)** (0.16) - N-dimensional arrays
- **[rayon](https://github.com/rayon-rs/rayon)** (1.10) - Data parallelism

### Algorithm Details

#### MLE Dimensionality

1. Build point collection from input data
2. For each point (in parallel):
   - Find k+1 nearest neighbors using brute-force search
   - Calculate distance ratios
   - Compute local dimension estimate
3. Return mean of local estimates

#### Two-NN Dimensionality

1. Build point collection from input data
2. For each point (in parallel):
   - Find 3 nearest neighbors (self + 2)
   - Calculate μ = r₂/r₁ ratio
3. Perform linear regression on sorted ratios
4. Return dimension estimate

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'effdim._rust'`:

1. Ensure you're using a prebuilt wheel or built from source
2. Check Python version compatibility (3.8-3.12)
3. Verify platform compatibility (Linux/macOS/Windows)

### Performance Issues

If performance is slower than expected:

1. Check CPU usage - should use all cores
2. Ensure release build (not debug)
3. Monitor memory usage - swapping will slow everything down

### Build Failures

If building from source fails:

1. Update Rust: `rustup update stable`
2. Install build dependencies: `pip install maturin setuptools-rust`
3. Check error logs in GitHub Actions/local build
4. See [RUST_BUILD.md](https://github.com/amanasci/EffDim/blob/main/RUST_BUILD.md) for detailed troubleshooting

## Contributing

Interested in improving performance? Contributions welcome!

- Benchmark new algorithms
- Optimize existing code
- Add GPU acceleration
- Improve documentation
