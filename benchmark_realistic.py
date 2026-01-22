import numpy as np
import time
from effdim import geometry

print("Rust available:", geometry._RUST_AVAILABLE)
print("\nBenchmarking with realistic test cases")
print("=" * 80)

test_cases = [
    (1000, 50),
    (5000, 100),
    (10000, 100),
    (10000, 200),
    (50000, 100),
]

print(f"\n{'Samples':<12} {'Dims':<8} {'MLE Time (s)':<15} {'Two-NN Time (s)':<15}")
print("-" * 80)

for n_samples, n_dims in test_cases:
    data = np.random.randn(n_samples, n_dims)
    
    start = time.time()
    _ = geometry.mle_dimensionality(data, k=10)
    mle_time = time.time() - start
    
    start = time.time()
    _ = geometry.two_nn_dimensionality(data)
    two_nn_time = time.time() - start
    
    print(f"{n_samples:<12} {n_dims:<8} {mle_time:<15.4f} {two_nn_time:<15.4f}")

print("\n" + "=" * 80)
