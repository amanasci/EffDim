#!/usr/bin/env python3
"""
Benchmark script for testing geometry.py performance with Rust implementation.
Tests performance on large datasets with high dimensions.
"""

import numpy as np
import time
from effdim import geometry

def benchmark_mle(n_samples, n_dims, k=10):
    """Benchmark MLE dimensionality estimation."""
    print(f"\n=== MLE Dimensionality ({n_samples} samples, {n_dims} dimensions, k={k}) ===")
    
    # Generate random data
    data = np.random.randn(n_samples, n_dims)
    
    # Benchmark
    start = time.time()
    result = geometry.mle_dimensionality(data, k=k)
    elapsed = time.time() - start
    
    print(f"Result: {result:.4f}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Throughput: {n_samples / elapsed:.0f} samples/sec")
    
    return elapsed, result

def benchmark_two_nn(n_samples, n_dims):
    """Benchmark Two-NN dimensionality estimation."""
    print(f"\n=== Two-NN Dimensionality ({n_samples} samples, {n_dims} dimensions) ===")
    
    # Generate random data
    data = np.random.randn(n_samples, n_dims)
    
    # Benchmark
    start = time.time()
    result = geometry.two_nn_dimensionality(data)
    elapsed = time.time() - start
    
    print(f"Result: {result:.4f}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Throughput: {n_samples / elapsed:.0f} samples/sec")
    
    return elapsed, result

def benchmark_box_counting(n_samples, n_dims):
    """Benchmark box-counting dimensionality estimation."""
    print(f"\n=== Box-Counting Dimensionality ({n_samples} samples, {n_dims} dimensions) ===")
    
    # Generate random data
    data = np.random.randn(n_samples, n_dims)
    
    # Benchmark
    start = time.time()
    result = geometry.box_counting_dimensionality(data)
    elapsed = time.time() - start
    
    print(f"Result: {result:.4f}")
    print(f"Time: {elapsed:.4f}s")
    print(f"Throughput: {n_samples / elapsed:.0f} samples/sec")
    
    return elapsed, result

def main():
    print("=" * 80)
    print("EffDim Geometry Module Benchmark")
    print("=" * 80)
    
    # Check if Rust implementation is available
    print(f"\nRust implementation available: {geometry._RUST_AVAILABLE}")
    
    # Test cases: progressively larger datasets
    test_cases = [
        (1000, 100),      # Small: 1K samples, 100 dims
        (10000, 500),     # Medium: 10K samples, 500 dims
        (100000, 700),    # Large: 100K samples, 700 dims (user's case)
        (100000, 1000),   # Large: 100K samples, 1000 dims
    ]
    
    # Only run huge datasets if Rust is available (too slow in Python)
    if geometry._RUST_AVAILABLE:
        test_cases.extend([
            (1000000, 1000),  # Huge: 1M samples, 1000 dims
            (2000000, 1000),  # Huge: 2M samples, 1000 dims
        ])
    
    results = []
    
    for n_samples, n_dims in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {n_samples} samples × {n_dims} dimensions")
        print(f"{'=' * 80}")
        
        try:
            # MLE benchmarks
            mle_time, mle_result = benchmark_mle(n_samples, n_dims, k=10)
            
            # Two-NN benchmarks
            two_nn_time, two_nn_result = benchmark_two_nn(n_samples, n_dims)
            
            # Box-counting benchmarks (skip for very large datasets)
            if n_samples <= 100000:
                box_time, box_result = benchmark_box_counting(n_samples, n_dims)
            else:
                box_time = None
                print("\n=== Box-Counting skipped for very large datasets ===")
            
            results.append({
                'n_samples': n_samples,
                'n_dims': n_dims,
                'mle_time': mle_time,
                'two_nn_time': two_nn_time,
                'box_time': box_time,
            })
            
        except Exception as e:
            print(f"\n ERROR: {e}")
            continue
    
    # Summary
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Samples':<12} {'Dims':<8} {'MLE (s)':<12} {'Two-NN (s)':<12} {'Box (s)':<12}")
    print("-" * 80)
    
    for r in results:
        box_str = f"{r['box_time']:.4f}" if r['box_time'] is not None else "N/A"
        print(f"{r['n_samples']:<12} {r['n_dims']:<8} {r['mle_time']:<12.4f} {r['two_nn_time']:<12.4f} {box_str:<12}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
