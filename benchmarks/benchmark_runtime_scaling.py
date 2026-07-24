"""
Runtime Scaling Benchmark (Fixed Vector Size)

This script isolates the effect of the number of vectors (N) on the execution 
time of the dimensionality computation. 

Methodology:
1. The vector size (D) and the true underlying rank are fixed (default to 1024).
2. The number of vectors (N) is scaled from a small number (e.g., 100) up to 
   a large number (e.g., 10,000).
3. For each N, synthetic data is generated with the fixed D and rank.
4. The execution time of `compute_dim` is recorded, along with all the various 
   predicted dimensionality metrics it outputs.
5. The results are saved to a CSV, and a figure with two plots is generated:
   - A line plot showing how the runtime scales with N.
   - A line plot showing how each metric's predicted rank scales with N.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from effdim.api import compute_dim

def generate_data(n_samples, n_features, rank, noise_level=1e-5):
    """Generate synthetic data with a known effective dimensionality."""
    # A low-rank underlying structure (or full rank if rank >= min(n_samples, n_features))
    A = np.random.randn(n_samples, rank)
    B = np.random.randn(rank, n_features)
    data = A @ B
    
    # Add noise
    if noise_level > 0:
        data += noise_level * np.random.randn(n_samples, n_features)
    return data

def run_benchmarks():
    # Fixed parameters
    fixed_D = 1024
    true_rank = 512 
    
    # Scale N up to 12,800 by doubling
    N_values = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    
    results = []

    print(f"{'N':<8} | {'D':<6} | {'Time (s)':<10}")
    print("-" * 32)

    with tqdm(total=len(N_values), desc="Benchmarking Runtime vs N", unit="run") as pbar:
        for n in N_values:
            # Rank cannot exceed min(N, D)
            actual_rank = min(true_rank, n, fixed_D)
            
            # 1. Setup
            pbar.set_postfix_str(f"Generating data for N={n}...")
            data = generate_data(n, fixed_D, rank=actual_rank)
            
            # 2. Benchmark Speed
            pbar.set_postfix_str(f"Computing dimensions for N={n}...")
            start_time = time.perf_counter()
            
            # Compute effective dimensionality
            all_dims = compute_dim(data)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            tqdm.write(f"{n:<8} | {fixed_D:<6} | {duration:<10.4f}")
            
            # Compile row
            row = {
                'N': n,
                'D': fixed_D,
                'Time_Seconds': duration,
                'True_Rank': actual_rank
            }
            # Add all metrics to the row (commented out for now)
            # for metric_name, val in all_dims.items():
            #     # Some metrics might return arrays or dicts (though compute_dim usually returns floats)
            #     if isinstance(val, (int, float, np.number)):
            #         row[metric_name] = val
                    
            results.append(row)
            pbar.update(1)

    # Create results directory if it doesn't exist
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_runtime_scaling.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")
    
    return df, output_dir

def plot_results(df, output_dir):
    """Generate line plots showing runtime scaling with N."""
    # Standard academic formatting
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # --- Plot 1: Runtime (Log-Log) ---
    sns.lineplot(data=df, x='N', y='Time_Seconds', marker='o', linewidth=2, markersize=8, ax=ax, label="Measured Runtime")
    
    # Calculate empirical computational complexity (gradient in log-log space)
    log_n = np.log2(df['N'])
    log_t = np.log2(df['Time_Seconds'])
    m, c = np.polyfit(log_n, log_t, 1)
    
    # Add linear fit line
    fit_t = (2**c) * (df['N'] ** m)
    sns.lineplot(x=df['N'], y=fit_t, ax=ax, color='red', linestyle='--', linewidth=2, label=f'Linear Fit: $O(N^{{{m:.2f}}})$')
    
    ax.set_title(f"Empirical Runtime Complexity (D={df['D'].iloc[0]})", fontweight='bold')
    ax.set_xlabel("Number of Vectors ($N$)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.legend(loc='upper left')

    # --- Plot 2: Predicted Ranks --- (Commented out temporarily)
    # exclude_cols = {'N', 'D', 'Time_Seconds'}
    # metric_cols = [col for col in df.columns if col not in exclude_cols]
    # df_melted = df.melt(id_vars=['N'], value_vars=metric_cols, var_name='Metric', value_name='Predicted_Rank')
    # sns.lineplot(data=df_melted, x='N', y='Predicted_Rank', hue='Metric', marker='o', linewidth=2, ax=axes[1])
    # target_rank = df['True_Rank'].max()
    # axes[1].axhline(y=target_rank, color='black', linestyle='--', linewidth=2, label=f'Target True Rank ({target_rank})')
    # axes[1].set_title("Predicted Rank vs $N$", fontweight='bold')
    # axes[1].set_xlabel("Number of Vectors ($N$)")
    # axes[1].set_ylabel("Predicted Rank")
    # axes[1].set_xscale('log', base=2)
    # axes[1].set_ylim(0, 1024)
    # axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Explicitly set the x-ticks and format y-ticks to avoid 2^X notation
    ax.set_xticks(df['N'])
    ax.set_xticklabels(df['N'], rotation=45)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plot_path = output_dir / "benchmark_runtime_scaling.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    df_results, out_dir = run_benchmarks()
    plot_results(df_results, out_dir)
