#!/usr/bin/env python3
"""Main script for Physics Probe Subspace experiments.

1. Loads embeddings for two models.
2. Streams Smith42/galaxies test labels and matches them to embeddings.
3. Splits into train/test (70/30).
4. Trains M linear probes on the train set for each model.
5. Extracts normal vectors to form a subspace, orthormalises via QR.
6. Projects test data onto subspace and computes mKNN.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import (
    ALL_PROBES,
    DEFAULT_11_PROBES,
    INDEPENDENT_PROBES,
    load_embeddings,
    load_physics_labels,
    platonic_root,
)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, 1e-12)


@torch.inference_mode()
def knn_euclidean(Z: torch.Tensor, k: int, row_batch: int = 512) -> torch.Tensor:
    """Compute exact k-NN indices using Euclidean distance."""
    n = Z.shape[0]
    idx = torch.empty((n, k), dtype=torch.long, device=Z.device)
    for i in range(0, n, row_batch):
        end = min(i + row_batch, n)
        # squared euclidean dist: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        x = Z[i:end]
        x2 = (x ** 2).sum(dim=1, keepdim=True)
        z2 = (Z ** 2).sum(dim=1, keepdim=True).T
        dist2 = x2 + z2 - 2 * (x @ Z.T)
        dist2.clamp_min_(0)
        
        # We want the k smallest distances, excluding self (which has dist=0)
        # Fill self-distance with infinity
        arange = torch.arange(i, end, device=Z.device)
        dist2[torch.arange(end - i), arange] = float('inf')
        
        topk = dist2.topk(k, dim=1, largest=False)
        idx[i:end] = topk.indices
    return idx


@torch.inference_mode()
def knn_cosine(Z: torch.Tensor, k: int, row_batch: int = 512) -> torch.Tensor:
    """Compute exact k-NN indices using Cosine similarity."""
    Z_norm = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = Z_norm.shape[0]
    idx = torch.empty((n, k), dtype=torch.long, device=Z_norm.device)
    for i in range(0, n, row_batch):
        end = min(i + row_batch, n)
        sim = Z_norm[i:end] @ Z_norm.T
        
        # Exclude self
        arange = torch.arange(i, end, device=Z_norm.device)
        sim[torch.arange(end - i), arange] = -float('inf')
        
        topk = sim.topk(k, dim=1, largest=True)
        idx[i:end] = topk.indices
    return idx


def mknn_overlap(knn1: torch.Tensor, knn2: torch.Tensor) -> float:
    """Compute mean k-nearest neighbor overlap."""
    n, k = knn1.shape
    # Count intersections row by row.
    # We can do this on CPU efficiently, or using broadcasting on GPU.
    # For k=10, looping on CPU over tensor.tolist() is okay for n=10000.
    overlap = 0
    k1_list = knn1.cpu().tolist()
    k2_list = knn2.cpu().tolist()
    for row1, row2 in zip(k1_list, k2_list):
        overlap += len(set(row1) & set(row2))
    return float(overlap) / (n * k)


def train_probes(Z: np.ndarray, y_dict: dict[str, np.ndarray], probe_keys: list[str]) -> tuple[np.ndarray, dict]:
    """Train linear probes and return weight matrix W and diagnostic stats."""
    D = Z.shape[1]
    M = len(probe_keys)
    W = np.zeros((D, M), dtype=np.float32)
    stats = {}
    
    for m, key in enumerate(probe_keys):
        y = y_dict[key]
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            print(f"Warning: Probe '{key}' has less than 10 valid samples, skipping.")
            stats[key] = {"r2_train": float('nan'), "r2_cv": float('nan'), "n_valid": int(valid.sum())}
            continue
            
        Z_valid = Z[valid]
        y_valid = y[valid]
        
        # Standardize target
        y_mean = y_valid.mean()
        y_std = y_valid.std() + 1e-12
        y_valid_std = (y_valid - y_mean) / y_std
        
        # Fit on full train valid set
        model = LinearRegression(fit_intercept=True)
        model.fit(Z_valid, y_valid_std)
        w = model.coef_
        W[:, m] = w
        
        r2_train = r2_score(y_valid_std, model.predict(Z_valid))
        
        # 5-fold CV
        cv_scores = []
        kf = KFold(n_splits=min(5, len(y_valid)), shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(Z_valid):
            m_cv = LinearRegression(fit_intercept=True)
            m_cv.fit(Z_valid[train_idx], y_valid_std[train_idx])
            pred = m_cv.predict(Z_valid[test_idx])
            cv_scores.append(r2_score(y_valid_std[test_idx], pred))
            
        stats[key] = {
            "r2_train": float(r2_train),
            "r2_cv": float(np.mean(cv_scores)),
            "n_valid": int(valid.sum())
        }
    
    return W, stats


def main():
    parser = argparse.ArgumentParser(description="Physics Probe Subspace mKNN")
    parser.add_argument("--platonic-root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="physics")
    parser.add_argument("--model-a", type=str, default="vit_base")
    parser.add_argument("--model-b", type=str, default="dinov3_vitb16")
    parser.add_argument("--max-n", type=int, default=16384)
    parser.add_argument("--probes", type=str, default="independent",
                        help="Comma-separated probe keys, or 'all', 'independent', 'default11'")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    root = platonic_root(args.platonic_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "outputs" / "probe_basis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve probes
    if args.probes == "all":
        probe_keys = list(ALL_PROBES.keys())
    elif args.probes == "independent":
        probe_keys = INDEPENDENT_PROBES
    elif args.probes == "default11":
        probe_keys = DEFAULT_11_PROBES
    else:
        probe_keys = [p.strip() for p in args.probes.split(",")]
        
    print(f"Using {len(probe_keys)} probes.")
    
    # 1. Load data
    path_a = root / "data_hf" / args.dataset / f"{args.model_a}_test.parquet"
    path_b = root / "data_hf" / args.dataset / f"{args.model_b}_test.parquet"
    
    print("Loading embeddings...")
    Z_A = load_embeddings(path_a, col=f"{args.model_a}_galaxies")[:args.max_n]
    Z_B = load_embeddings(path_b, col=f"{args.model_b}_galaxies")[:args.max_n]
    assert len(Z_A) == len(Z_B), "Models must have matching row counts"
    
    n_samples = min(args.max_n, len(Z_A))
    Z_A = Z_A[:n_samples]
    Z_B = Z_B[:n_samples]
    
    # L2 normalize raw embeddings
    Z_A = l2_normalize(Z_A)
    Z_B = l2_normalize(Z_B)
    
    print(f"Loading physics labels for {n_samples} samples...")
    labels = load_physics_labels(n_samples, split="test")
    
    # 2. Train/Test split
    print("Splitting train/test...")
    idx = np.arange(n_samples)
    idx_train, idx_test = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    
    Z_A_train, Z_A_test = Z_A[idx_train], Z_A[idx_test]
    Z_B_train, Z_B_test = Z_B[idx_train], Z_B[idx_test]
    
    y_train = {k: v[idx_train] for k, v in labels.items()}
    y_test = {k: v[idx_test] for k, v in labels.items()}
    
    # 3. Train Probes
    print("Training probes for Model A...")
    W_A, stats_A = train_probes(Z_A_train, y_train, probe_keys)
    
    print("Training probes for Model B...")
    W_B, stats_B = train_probes(Z_B_train, y_train, probe_keys)
    
    # 4. QR Decompose to find orthogonal subspace
    print("Extracting orthonormal basis...")
    Q_A, R_A = np.linalg.qr(W_A)
    Q_B, R_B = np.linalg.qr(W_B)
    
    r_A = Q_A.shape[1]
    r_B = Q_B.shape[1]
    print(f"Basis A rank: {r_A}, Basis B rank: {r_B}")
    
    # 5. Project Test Data
    print("Projecting test data...")
    Z_A_test_proj = Z_A_test @ Q_A
    Z_B_test_proj = Z_B_test @ Q_B
    
    # 6. mKNN Metrics
    print("Computing mKNN...")
    device = torch.device(args.device)
    
    # Tensors
    ZA_t = torch.from_numpy(Z_A_test).to(device)
    ZB_t = torch.from_numpy(Z_B_test).to(device)
    ZAp_t = torch.from_numpy(Z_A_test_proj).to(device)
    ZBp_t = torch.from_numpy(Z_B_test_proj).to(device)
    
    # Dense cosine mKNN
    knn_A_dense = knn_cosine(ZA_t, args.k)
    knn_B_dense = knn_cosine(ZB_t, args.k)
    mknn_dense = mknn_overlap(knn_A_dense, knn_B_dense)
    
    # Projected Euclidean mKNN (in subspace)
    knn_A_proj = knn_euclidean(ZAp_t, args.k)
    knn_B_proj = knn_euclidean(ZBp_t, args.k)
    mknn_subspace = mknn_overlap(knn_A_proj, knn_B_proj)
    
    # Sweep over subspace dimensions to see how mKNN saturates
    print("Computing mKNN sweep over subspace dimensions...")
    mknn_vs_dim = []
    max_dim = min(r_A, r_B)
    for d in range(1, max_dim + 1):
        ZAp_d = torch.from_numpy(Z_A_test_proj[:, :d]).to(device)
        ZBp_d = torch.from_numpy(Z_B_test_proj[:, :d]).to(device)
        
        kAd = knn_euclidean(ZAp_d, args.k)
        kBd = knn_euclidean(ZBp_d, args.k)
        
        mknn_vs_dim.append({
            "dim": d,
            "mknn": mknn_overlap(kAd, kBd)
        })
        
    # 7. Write results
    results = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "n_samples": n_samples,
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "n_probes": len(probe_keys),
        "rank_a": int(r_A),
        "rank_b": int(r_B),
        "mknn_dense_cosine": mknn_dense,
        "mknn_subspace_euclidean": mknn_subspace,
        "mknn_vs_dim": mknn_vs_dim,
        "stats_a": stats_A,
        "stats_b": stats_B,
    }
    
    res_path = out_dir / "results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
        
    np.savez(
        out_dir / "probe_weights.npz",
        W_A=W_A, W_B=W_B, Q_A=Q_A, Q_B=Q_B, probe_keys=probe_keys
    )
    
    print("\n=== RESULTS ===")
    print(f"mKNN Dense Cosine:       {mknn_dense:.4f}")
    print(f"mKNN Subspace Euclidean: {mknn_subspace:.4f}")
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    main()
