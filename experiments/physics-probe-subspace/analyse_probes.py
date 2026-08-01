#!/usr/bin/env python3
"""Analysis script for Physics Probe Subspace.

Generates:
1. R^2 comparison bar chart between Model A and Model B.
2. Canonical angles between the two probe subspaces.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_principal_angles(Q_A: np.ndarray, Q_B: np.ndarray) -> np.ndarray:
    """Compute principal angles between two subspaces defined by orthonormal bases Q_A and Q_B."""
    # SVD of Q_A^T Q_B
    M = Q_A.T @ Q_B
    U, S, Vt = np.linalg.svd(M)
    # S contains the cosines of the principal angles
    S = np.clip(S, 0.0, 1.0)
    angles_rad = np.arccos(S)
    return np.degrees(angles_rad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to results.json")
    args = parser.parse_args()
    
    res_path = Path(args.results)
    with open(res_path, "r") as f:
        results = json.load(f)
        
    out_dir = res_path.parent
    npz_path = out_dir / "probe_weights.npz"
    
    if npz_path.exists():
        data = np.load(npz_path)
        Q_A = data["Q_A"]
        Q_B = data["Q_B"]
        angles = compute_principal_angles(Q_A, Q_B)
        
        plt.figure(figsize=(8, 5))
        plt.plot(angles, marker='o')
        plt.title("Principal Angles between Physical Subspaces")
        plt.xlabel("Dimension index")
        plt.ylabel("Angle (degrees)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "principal_angles.png")
        plt.close()
        print(f"Mean principal angle: {angles.mean():.2f}°")
        print(f"Saved principal_angles.png")
        
    # Plot R2 comparisons
    stats_a = results.get("stats_a", {})
    stats_b = results.get("stats_b", {})
    
    probes = list(stats_a.keys())
    if probes:
        r2_a = [stats_a[k]["r2_cv"] for k in probes]
        r2_b = [stats_b[k]["r2_cv"] if k in stats_b else 0 for k in probes]
        
        x = np.arange(len(probes))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, r2_a, width, label=results["model_a"])
        plt.bar(x + width/2, r2_b, width, label=results["model_b"])
        
        plt.ylabel('R^2 (5-fold CV)')
        plt.title('Probe Accuracy Comparison')
        plt.xticks(x, probes, rotation=45, ha="right")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(out_dir / "probe_r2_comparison.png")
        plt.close()
        print(f"Saved probe_r2_comparison.png")

if __name__ == "__main__":
    main()
