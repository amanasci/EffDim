import numpy as np
import time
from effdim import geometry

print("Rust available:", geometry._RUST_AVAILABLE)

test_cases = [
    (1000, 100),
    (10000, 500),
    (100000, 700),
    (100000, 1000),
]

if geometry._RUST_AVAILABLE:
    test_cases.extend([
        (500000, 1000),
        (1000000, 1000),
    ])

print("\nBenchmarking MLE Dimensionality:")
print("=" * 80)
print(f"{'Samples':<12} {'Dims':<8} {'Time (s)':<12} {'Throughput':<20}")
print("-" * 80)

for n_samples, n_dims in test_cases:
    data = np.random.randn(n_samples, n_dims)
    start = time.time()
    result = geometry.mle_dimensionality(data, k=10)
    elapsed = time.time() - start
    throughput = n_samples / elapsed
    print(f"{n_samples:<12} {n_dims:<8} {elapsed:<12.4f} {throughput:>12.0f} samples/s")

print("\nBenchmarking Two-NN Dimensionality:")
print("=" * 80)
print(f"{'Samples':<12} {'Dims':<8} {'Time (s)':<12} {'Throughput':<20}")
print("-" * 80)

for n_samples, n_dims in test_cases:
    data = np.random.randn(n_samples, n_dims)
    start = time.time()
    result = geometry.two_nn_dimensionality(data)
    elapsed = time.time() - start
    throughput = n_samples / elapsed
    print(f"{n_samples:<12} {n_dims:<8} {elapsed:<12.4f} {throughput:>12.0f} samples/s")
