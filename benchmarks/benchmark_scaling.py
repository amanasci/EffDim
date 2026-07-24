"""
Benchmark Scaling Script

This script benchmarks the speed and accuracy of the effective dimensionality 
computation (`compute_dim`) across a 2D grid of parameters:
- N: Number of vectors (samples)
- D: Vector size (features)

Methodology:
1. For each combination of N and D, it generates a synthetic dataset with a known 
   low-rank structure (true underlying dimensionality) and adds a small amount of noise.
2. It measures the execution time of `compute_dim`.
3. It extracts the estimated effective dimensionality (using participation ratio as a baseline) 
   and calculates the absolute error against the true rank.
4. It outputs the tabular results to a CSV and generates heatmaps showing how speed 
   and accuracy scale across the N x D grid.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os
from pathlib import Path
from tqdm import tqdm
from effdim.api import compute_dim

def generate_data(n_samples, n_features, rank, noise_level=1e-5):
    """Generate synthetic data with a known effective dimensionality."""
    # A low-rank underlying structure
    A = np.random.randn(n_samples, rank)
    B = np.random.randn(rank, n_features)
    data = A @ B
    
    # Add noise
    if noise_level > 0:
        data += noise_level * np.random.randn(n_samples, n_features)
    return data

def run_benchmarks():
    # Define grid of parameters
    N_values = [100, 500, 1000, 5000] # Number of vectors
    D_values = [10, 50, 100, 500]     # Vector size
    
    # Keep true rank constant or proportional
    true_rank = 5 
    
    results = []

    print(f"{'N':<6} | {'D':<6} | {'Time (s)':<10} | {'True Dim':<10} | {'Est Dim':<10} | {'Error'}")
    print("-" * 65)

    # Use tqdm for progress tracking
    total_iterations = len(N_values) * len(D_values)
    with tqdm(total=total_iterations, desc="Benchmarking Scaling", unit="run") as pbar:
        for n, d in product(N_values, D_values):
            # We can't have a rank higher than min(n, d)
            actual_rank = min(true_rank, n, d)
            
            # 1. Setup
            pbar.set_postfix_str(f"Generating data for N={n}, D={d}...")
            data = generate_data(n, d, rank=actual_rank)
            
            # 2. Benchmark Speed
            pbar.set_postfix_str(f"Computing dimensions for N={n}, D={d}...")
            start_time = time.perf_counter()
            
            # Compute effective dimensionality
            all_dims = compute_dim(data)
            
            # Choose one metric to use as our 'estimated dim' baseline (e.g. participation_ratio)
            estimated_dim = all_dims.get("participation_ratio", 0)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # 3. Benchmark Accuracy
            accuracy_error = abs(estimated_dim - actual_rank)
            
            # Print row nicely using tqdm.write so it doesn't break the progress bar
            tqdm.write(f"{n:<6} | {d:<6} | {duration:<10.4f} | {actual_rank:<10} | {estimated_dim:<10.2f} | {accuracy_error:.2f}")
            
            results.append({
                'N': n,
                'D': d,
                'Time_Seconds': duration,
                'True_Dim': actual_rank,
                'Estimated_Dim': estimated_dim,
                'Error': accuracy_error
            })
            
            pbar.update(1)

    # Create results directory if it doesn't exist
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Convert to DataFrame for easy plotting/saving
    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")
    
    return df, output_dir

def plot_results(df, output_dir):
    """Generate heatmaps for Speed and Accuracy."""
    plt.figure(figsize=(14, 5))
    
    # Plot Speed
    plt.subplot(1, 2, 1)
    speed_pivot = df.pivot(index='N', columns='D', values='Time_Seconds')
    sns.heatmap(speed_pivot, annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title("Execution Time (seconds)")
    plt.xlabel("Vector Size (D)")
    plt.ylabel("Number of Vectors (N)")
    
    # Plot Accuracy (Error)
    plt.subplot(1, 2, 2)
    acc_pivot = df.pivot(index='N', columns='D', values='Error')
    sns.heatmap(acc_pivot, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Absolute Error in Dimensionality")
    plt.xlabel("Vector Size (D)")
    plt.ylabel("Number of Vectors (N)")
    
    plt.tight_layout()
    plot_path = output_dir / "benchmark_scaling.png"
    plt.savefig(plot_path)
    print(f"Saved plots to {plot_path}")

if __name__ == "__main__":
    df_results, out_dir = run_benchmarks()
    plot_results(df_results, out_dir)
