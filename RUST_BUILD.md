# Building the Rust Extension

EffDim now includes a Rust implementation of geometry functions for improved performance on large datasets.

## Prerequisites

1. **Rust toolchain** (1.70+):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Maturin**:
   ```bash
   pip install maturin
   ```

## Building

To build the Rust extension:

```bash
# Build in release mode (optimized)
maturin build --release

# Install the built wheel
pip install target/wheels/effdim-*.whl --force-reinstall
```

For development:

```bash
# Build and install in development mode (requires virtualenv)
maturin develop --release
```

## Architecture

The Rust implementation provides:

- **Parallel brute-force nearest neighbor search** using `rayon` for multi-core performance
- **Optimized for high-dimensional data** (100-1000+ dimensions)
- **Automatic fallback** to Python implementation if Rust module is not available

### Performance Characteristics

The Rust implementation uses parallel brute-force nearest neighbor search, which:
- Scales well with CPU cores
- Works efficiently for high-dimensional data (where k-d trees perform poorly)
- Provides 10-50x speedup over Python for medium datasets (1k-10k samples)

**Benchmark results (on GitHub Actions runners):**
- 1,000 samples × 100 dims: ~0.05s (MLE & Two-NN)
- 5,000 samples × 200 dims: ~2.5s (MLE & Two-NN)
- 10,000 samples × 700 dims: ~36s (MLE & Two-NN)

For very large datasets (100k+ samples), times scale roughly linearly with sample count.

## Files

- `Cargo.toml` - Rust package configuration
- `src_rust/lib.rs` - Rust implementation of geometry functions
- `src/effdim/geometry.py` - Python wrapper with automatic Rust/Python fallback

## Dependencies

The Rust implementation uses:
- **pyo3** (0.23) - Python bindings
- **numpy** (0.23) - NumPy array interop
- **ndarray** (0.16) - N-dimensional arrays
- **rayon** (1.10) - Data parallelism

All dependencies are automatically fetched by Cargo during build.
