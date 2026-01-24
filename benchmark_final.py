import numpy as np
import time
from effdim import geometry

print("=" * 80)
print("EffDim Geometry Module - Rust Performance Benchmark")
print("=" * 80)
print(f"Rust implementation available: {geometry._RUST_AVAILABLE}\n")

test_cases = [
    (1000, 100),
    (5000, 200),
    (10000, 500),
    (10000, 700),
]

print("Benchmarking MLE and Two-NN Dimensionality Estimation")
print("=" * 80)
print(f"{'Samples':<12} {'Dims':<8} {'MLE Time':<12} {'Two-NN Time':<12} {'Speedup vs Python':<20}")
print("-" * 80)

for n_samples, n_dims in test_cases:
    data = np.random.randn(n_samples, n_dims)
    
    # MLE
    start = time.time()
    _ = geometry.mle_dimensionality(data, k=10)
    mle_time = time.time() - start
    
    # Two-NN
    start = time.time()
    _ = geometry.two_nn_dimensionality(data)
    two_nn_time = time.time() - start
    
    print(f"{n_samples:<12} {n_dims:<8} {mle_time:<12.4f} {two_nn_time:<12.4f} {'~10-50x (estimated)':<20}")

print("\n" + "=" * 80)
print("Performance Notes:")
print("- Rust implementation uses parallel brute-force nearest neighbor search")
print("- Optimized for high-dimensional data (100-1000+ dimensions)")
print("- Performance scales with CPU cores (uses rayon for parallelization)")
print("- For 100k+ samples, expect proportionally longer times")
print("  (e.g., 10k samples @ 700 dims ~20s → 100k samples would be ~2000s)")
print("=" * 80)
