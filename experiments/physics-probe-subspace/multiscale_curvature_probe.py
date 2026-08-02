#!/usr/bin/env python3
"""Multi-scale PCA residual curvature ↔ physics probe error correlation.

Following Little et al. (MIT, 2011), computes for each test point:

    rf(K) = 1 - (sum of top-k_t eigenvalues) / (total variance)

where k_t is the global Two-NN intrinsic dimensionality estimate, and K is
the neighborhood size. Curvature is the growth of the residual fraction across
scales:

    κ_ms(i) = max(0, rf(K_large, i) − rf(K_small, i))

Results are reported per model and aggregated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import (  # noqa: E402
    ALL_PROBES,
    DEFAULT_11_PROBES,
    INDEPENDENT_PROBES,
    compute_probe_residuals,
    correlation_analysis,
    load_embeddings,
    load_physics_labels,
    platonic_root,
    train_probes,
)


# ---------------------------------------------------------------------------
# kNN utilities (FAISS preferred, sklearn fallback)
# ---------------------------------------------------------------------------

def build_knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Return (n, k) integer kNN index array (Euclidean, no self)."""
    try:
        import faiss
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        index = faiss.IndexFlatL2(X32.shape[1])
        index.add(X32)
        _, I = index.search(X32, k + 1)
        return I[:, 1 : k + 1]
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
        I = nn.kneighbors(X, return_distance=False)
        return I[:, 1:]


# ---------------------------------------------------------------------------
# Two-NN global intrinsic dimension (via effdim)
# ---------------------------------------------------------------------------

def estimate_global_id(X: np.ndarray) -> float:
    """Estimate global intrinsic dimensionality using Two-NN (effdim)."""
    try:
        import sys as _sys
        _repo_root = Path(__file__).resolve().parents[2]
        src_path = str(_repo_root / "src")
        if src_path not in _sys.path:
            _sys.path.insert(0, src_path)
        from effdim.geometry import two_nn_dimensionality
        return two_nn_dimensionality(X.astype(np.float32))
    except Exception as exc:
        print(f"  Warning: two_nn_dimensionality failed ({exc}), using k_t=10 fallback.", flush=True)
        return 10.0


# ---------------------------------------------------------------------------
# Multi-scale PCA residual fraction
# ---------------------------------------------------------------------------

def residual_fraction(
    X: np.ndarray,
    knn_idx: np.ndarray,
    k_t: int,
) -> np.ndarray:
    """Compute the PCA residual fraction for each point.

    rf(i) = 1 - (sum of top-k_t eigenvalues of local covariance) / total variance

    Parameters
    ----------
    X       : (n, D) embeddings
    knn_idx : (n, K) precomputed neighbor indices
    k_t     : number of tangent directions to retain

    Returns
    -------
    rf : (n,) float32 residual fraction per point
    """
    n = X.shape[0]
    K = knn_idx.shape[1]
    k_t_eff = min(k_t, K - 1, X.shape[1])  # can't exceed neighborhood size or ambient dim
    rf = np.zeros(n, dtype=np.float64)

    for i in range(n):
        nbrs = X[knn_idx[i]]               # (K, D)
        mu = nbrs.mean(axis=0)
        centred = nbrs - mu                 # (K, D)
        # Economy SVD: singular values of centred / sqrt(K-1)
        # sv^2 = eigenvalues of covariance C = centred^T centred / (K-1)
        sv = np.linalg.svd(centred / max(np.sqrt(K - 1), 1e-9), compute_uv=False)
        ev = sv ** 2                        # descending eigenvalues
        total = ev.sum()
        if total < 1e-15:
            rf[i] = 0.0
            continue
        top_k = ev[:k_t_eff].sum()
        rf[i] = 1.0 - top_k / total

    return rf.astype(np.float32)


def compute_multiscale_curvature(
    X: np.ndarray,
    k_small: int,
    k_large: int,
    k_t: int,
) -> dict[str, np.ndarray]:
    """Compute κ_ms and component residual fractions for all points.

    Parameters
    ----------
    X       : (n, D) embeddings
    k_small : neighborhood size for small scale (e.g. 30)
    k_large : neighborhood size for large scale (e.g. 200)
    k_t     : tangent dimension (from Two-NN)

    Returns
    -------
    dict with keys 'rf_small', 'rf_large', 'kappa_ms'
    """
    print(f"  Building kNN graph (k_large={k_large})...", flush=True)
    knn_large = build_knn_indices(X, k_large)
    knn_small = knn_large[:, :k_small]  # reuse — already sorted by distance

    print(f"  Computing rf(K_small={k_small}, k_t={k_t})...", flush=True)
    rf_small = residual_fraction(X, knn_small, k_t)

    print(f"  Computing rf(K_large={k_large}, k_t={k_t})...", flush=True)
    rf_large = residual_fraction(X, knn_large, k_t)

    kappa = np.maximum(0.0, rf_large - rf_small).astype(np.float32)

    return {
        "rf_small":  rf_small,
        "rf_large":  rf_large,
        "kappa_ms":  kappa,
    }


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def run_one_model(
    model_name: str,
    parquet_path: Path,
    col: str,
    *,
    root: Path,
    args: argparse.Namespace,
    probe_keys: list[str],
    labels_full: dict[str, np.ndarray],
    out_dir: Path,
) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  Model: {model_name}  ({col})", flush=True)
    print(f"{'='*60}", flush=True)

    # --- Load embeddings ---
    Z = load_embeddings(parquet_path, col=col)
    n_total = min(args.max_n, len(Z))
    Z = Z[:n_total]
    labels = {k: v[:n_total] for k, v in labels_full.items()}

    # L2 normalize
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    # --- Train/test split ---
    idx = np.arange(n_total)
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed
    )

    # --- Two-NN intrinsic dimension on train split ---
    print("  Estimating intrinsic dimension (Two-NN)...", flush=True)
    k_t_float = estimate_global_id(Z[idx_train])
    k_t = max(1, int(round(k_t_float)))
    print(f"  Two-NN k_t = {k_t_float:.2f}  →  using k_t={k_t}", flush=True)

    # --- Train probes on train split ---
    Z_train = Z[idx_train]
    y_train = {k: v[idx_train] for k, v in labels.items()}
    print(f"  Training {len(probe_keys)} linear probes...", flush=True)
    W, probe_stats = train_probes(Z_train, y_train, probe_keys)

    # --- Probe residuals on test split ---
    Z_test = Z[idx_test]
    y_test = {k: v[idx_test] for k, v in labels.items()}
    residuals, mean_residual = compute_probe_residuals(Z_test, y_test, W, probe_keys)
    print(f"  Mean probe ε²: {np.nanmean(mean_residual):.4f} ± {np.nanstd(mean_residual):.4f}", flush=True)

    # --- Multi-scale curvature on test split ---
    curv_dict = compute_multiscale_curvature(
        Z_test, args.k_small, args.k_large, k_t
    )
    print(
        f"  κ_ms: mean={curv_dict['kappa_ms'].mean():.4f}  "
        f"std={curv_dict['kappa_ms'].std():.4f}  "
        f"max={curv_dict['kappa_ms'].max():.4f}",
        flush=True,
    )

    # --- Correlation analysis ---
    # Report kappa_ms as primary signal; also include rf_small and rf_large as decomposition
    tag = model_name.replace("/", "_")
    print(f"\n  Correlation analysis ({tag}):", flush=True)
    summary = correlation_analysis(
        curv_dict, mean_residual, residuals, probe_keys,
        output_dir=out_dir, tag=tag,
    )
    summary["model"] = model_name
    summary["col"] = col
    summary["k_t"] = k_t
    summary["k_t_float"] = float(k_t_float)
    summary["n_train"] = int(len(idx_train))
    summary["n_test"] = int(len(idx_test))
    summary["probe_stats"] = probe_stats
    summary["mean_probe_error"] = float(np.nanmean(mean_residual))
    summary["kappa_ms_mean"] = float(curv_dict["kappa_ms"].mean())
    summary["kappa_ms_std"] = float(curv_dict["kappa_ms"].std())

    # Save per-model curvature arrays
    np.savez_compressed(
        out_dir / f"{tag}_curvature.npz",
        **curv_dict,
        mean_residual=mean_residual,
        idx_test=idx_test,
    )

    return summary


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(summaries: list[dict]) -> dict:
    """Merge per-model Spearman rows, compute mean |ρ| across models."""
    all_metrics: set[str] = set()
    for s in summaries:
        for row in s.get("spearman", []):
            all_metrics.add(row["metric"])

    agg_rows = []
    for metric in sorted(all_metrics):
        rhos, aucs = [], []
        for s in summaries:
            for row in s.get("spearman", []):
                if row["metric"] == metric:
                    rhos.append(row["spearman_rho"])
                    aucs.append(row["logistic_auc"])
        agg_rows.append({
            "metric": metric,
            "mean_abs_rho": float(np.mean(np.abs(rhos))),
            "mean_auc": float(np.nanmean(aucs)),
            "rhos_per_model": rhos,
        })
    agg_rows.sort(key=lambda r: r["mean_abs_rho"], reverse=True)
    return {"aggregated_spearman": agg_rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--model-a", default="vit_base")
    p.add_argument("--model-b", default="dinov3_vitb16")
    p.add_argument("--dataset", default="physics")
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument(
        "--probes", default="independent",
        help="Probe set: 'independent', 'all', 'default11', or comma-separated keys"
    )
    p.add_argument("--k-small", type=int, default=30,
                   help="Neighborhood size for small-scale PCA (inner radius)")
    p.add_argument("--k-large", type=int, default=200,
                   help="Neighborhood size for large-scale PCA (outer radius)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    root = platonic_root(args.platonic_root)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "outputs" / "multiscale_curvature_probe"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve probe set
    if args.probes == "independent":
        probe_keys = INDEPENDENT_PROBES
    elif args.probes == "all":
        probe_keys = list(ALL_PROBES.keys())
    elif args.probes == "default11":
        probe_keys = DEFAULT_11_PROBES
    else:
        probe_keys = [pk.strip() for pk in args.probes.split(",")]
    print(f"Using {len(probe_keys)} probes.", flush=True)

    # Load labels once
    n_labels = args.max_n
    print(f"Loading physics labels ({n_labels} samples)...", flush=True)
    labels_full = load_physics_labels(n_labels, split="test")

    parquet_dir = root / "data_hf" / args.dataset
    model_configs = [
        (args.model_a, parquet_dir / f"{args.model_a}_test.parquet", f"{args.model_a}_galaxies"),
        (args.model_b, parquet_dir / f"{args.model_b}_test.parquet", f"{args.model_b}_galaxies"),
    ]

    summaries = []
    for model_name, parquet_path, col in model_configs:
        summary = run_one_model(
            model_name, parquet_path, col,
            root=root,
            args=args,
            probe_keys=probe_keys,
            labels_full=labels_full,
            out_dir=out_dir,
        )
        summaries.append(summary)

    agg = aggregate_results(summaries)

    # --- Write results ---
    payload = {
        "experiment": "multiscale_curvature_probe",
        "args": {
            "max_n": args.max_n,
            "k_small": args.k_small,
            "k_large": args.k_large,
            "probe_set": args.probes,
            "n_probes": len(probe_keys),
            "seed": args.seed,
        },
        "per_model": summaries,
        **agg,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    # --- Markdown summary ---
    lines = [
        "# Multi-Scale PCA Residual Curvature ↔ Physics Probe Error",
        "",
        f"- n_max={args.max_n}, probes={args.probes} ({len(probe_keys)}), "
        f"K_small={args.k_small}, K_large={args.k_large}",
        "",
        "## Aggregated Spearman |ρ| (mean across models)",
        "",
        "| Metric | Mean |ρ| | Mean AUC |",
        "|---|---:|---:|",
    ]
    for row in agg["aggregated_spearman"]:
        lines.append(f"| {row['metric']} | {row['mean_abs_rho']:.3f} | {row['mean_auc']:.3f} |")

    for s in summaries:
        lines += [
            "",
            f"## {s['model']} (n_test={s['n_test']}, k_t={s['k_t']})",
            f"Two-NN ID = {s['k_t_float']:.2f}, κ_ms mean={s['kappa_ms_mean']:.4f} ± {s['kappa_ms_std']:.4f}",
            "",
            "| Metric | ρ | p-val | AUC |",
            "|---|---:|---:|---:|",
        ]
        for row in s.get("spearman", []):
            lines.append(
                f"| {row['metric']} | {row['spearman_rho']:+.3f} "
                f"| {row['spearman_pval']:.2e} | {row['logistic_auc']:.3f} |"
            )

    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {out_dir}", flush=True)
    print((out_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
