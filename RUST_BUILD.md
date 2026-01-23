# Building the Rust Extension

EffDim now includes a Rust implementation of geometry functions for improved performance on large datasets.

## Installation

### From PyPI (Recommended)

Prebuilt wheels are available for Linux, macOS, and Windows:

```bash
pip install effdim
```

The Rust extension is automatically included in the prebuilt wheels. No additional setup required!

### From Source

If prebuilt wheels are not available for your platform, you can build from source.

## Building from Source

### Prerequisites

1. **Rust toolchain** (1.70+):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Maturin**:
   ```bash
   pip install maturin
   ```

## Building

### Build and Install

To build the Rust extension from source:

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

## CI/CD and Release Process

### Automated Builds

The project uses GitHub Actions with a maturin-generated CI workflow to automatically build prebuilt wheels for multiple platforms:

- **Platforms**: 
  - Linux: x86_64, aarch64 (manylinux & musllinux)
  - Windows: x64, x86
  - macOS: x86_64 (Intel), aarch64 (Apple Silicon)
- **Python versions**: 3.8-3.12 (via `--find-interpreter`)
- **Trigger**: Automatic on pushes to main, PRs, and version tags

### Publishing a New Version

To publish a new version to PyPI:

1. Update the version in `pyproject.toml`:
   ```toml
   [project]
   version = "0.1.1"  # Update this
   ```

2. Commit the change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git push origin main
   ```

3. Create and push a version tag:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

4. GitHub Actions will automatically:
   - Build wheels for all platforms and architectures
   - Build source distribution (sdist)
   - Generate build attestations
   - Upload all artifacts to PyPI

### Manual Build Workflow

You can also trigger builds manually from the GitHub Actions tab using the `workflow_dispatch` event.

### Supported Platforms

The prebuilt wheels support:

| Platform | Architectures | Notes |
|----------|---------------|-------|
| Linux    | x86_64, aarch64 | manylinux & musllinux |
| macOS    | x86_64, aarch64 | macOS 13+ (Intel), macOS 14+ (Apple Silicon) |
| Windows  | x64, x86 | Windows 10+ |

Python versions 3.8-3.12 are automatically detected and built for each platform.

If your platform is not listed, you can build from source (see above).
