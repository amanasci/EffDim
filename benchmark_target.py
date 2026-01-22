import numpy as np
import time
from effdim import geometry

print("Rust available:", geometry._RUST_AVAILABLE)
print("\nTesting the user's specific case: 100,000 samples with 700+ dimensions")
print("=" * 80)

# User's specific case
n_samples = 100000
n_dims = 700

print(f"\nGenerating data: {n_samples} samples × {n_dims} dimensions...")
data = np.random.randn(n_samples, n_dims)

print("\nTesting MLE dimensionality (k=10)...")
start = time.time()
result_mle = geometry.mle_dimensionality(data, k=10)
elapsed_mle = time.time() - start
print(f"  Result: {result_mle:.4f}")
print(f"  Time: {elapsed_mle:.4f} seconds")
print(f"  Throughput: {n_samples/elapsed_mle:.0f} samples/second")

print("\nTesting Two-NN dimensionality...")
start = time.time()
result_two_nn = geometry.two_nn_dimensionality(data)
elapsed_two_nn = time.time() - start
print(f"  Result: {result_two_nn:.4f}")
print(f"  Time: {elapsed_two_nn:.4f} seconds")
print(f"  Throughput: {n_samples/elapsed_two_nn:.0f} samples/second")

print("\n" + "=" * 80)
print("Summary:")
print(f"  MLE took {elapsed_mle:.2f}s for 100k samples × 700 dims")
print(f"  Two-NN took {elapsed_two_nn:.2f}s for 100k samples × 700 dims")
