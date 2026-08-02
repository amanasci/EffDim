#!/usr/bin/env python3
"""SAE-based curvature ↔ physics probe error correlation experiment.

For each model independently (Model A and Model B):
  1. Train or load a cached TopK SAE on the dense embeddings.
  2. Train linear probes for physics properties on the train split.
  3. Compute per-point probe residuals on the test split.
  4. Build a kNN graph on raw embeddings and compute 5 SAE curvature metrics.
  5. Correlate curvature metrics with probe error (Spearman ρ, binned box-plots,
     logistic AUC).

Results are reported per-model and aggregated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import (  # noqa: E402
    ALL_PROBES,
    DEFAULT_11_PROBES,
    INDEPENDENT_PROBES,
    compute_probe_residuals,
    correlation_analysis,
    ensure_sae_import,
    load_embeddings,
    load_physics_labels,
    platonic_root,
    train_probes,
)

ensure_sae_import()
from sae_model import TopKSAE  # noqa: E402


# ---------------------------------------------------------------------------
# SAE training + caching
# ---------------------------------------------------------------------------

def _sae_checkpoint_dir(
    root: Path,
    parquet_stem: str,
    col: str,
    feature_dim: int,
    topk: int,
    seed: int,
) -> Path:
    tag = f"F{feature_dim}_k{topk}_seed{seed}"
    return root / "outputs" / "sae" / parquet_stem / col / tag


def train_sae(
    X_train: np.ndarray,
    *,
    feature_dim: int,
    topk: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> tuple[TopKSAE, np.ndarray, np.ndarray]:
    """Train a TopK SAE. Returns (model, scaler_mean, scaler_scale)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train).astype(np.float32)

    input_dim = X_sc.shape[1]
    model = TopKSAE(input_dim, feature_dim, topk).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(X_sc)
    steps_per_epoch = max(1, n // batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = torch.as_tensor(X_sc[idx], device=device)
            x_hat, _ = model(xb)
            loss = nn.functional.mse_loss(x_hat, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(
                f"  SAE epoch {epoch+1:3d}/{epochs}  loss={epoch_loss/max(n_batches,1):.5f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)",
                flush=True,
            )
    model.eval()
    return model, scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)


def train_or_load_sae(
    X: np.ndarray,
    train_idx: np.ndarray,
    *,
    sae_dir: Path,
    feature_dim: int,
    topk: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> dict:
    """Load SAE from checkpoint if it exists, otherwise train and save."""
    model_pt = sae_dir / "model.pt"
    config_json = sae_dir / "config.json"
    scaler_npz = sae_dir / "scaler_stats.npz"

    if model_pt.is_file() and config_json.is_file() and scaler_npz.is_file():
        print(f"  Loading cached SAE from {sae_dir}", flush=True)
        cfg = json.loads(config_json.read_text())
        sc = np.load(scaler_npz)
        model = TopKSAE(cfg["dim"], cfg["feature_dim"], cfg["k"]).to(device)
        model.load_state_dict(
            torch.load(model_pt, map_location=device, weights_only=True)
        )
        model.eval()
        return {
            "model": model,
            "mean": sc["mean"].astype(np.float32),
            "scale": sc["scale"].astype(np.float32),
            "k": int(cfg["k"]),
            "feature_dim": int(cfg["feature_dim"]),
        }

    print(f"  Training SAE (F={feature_dim}, k={topk}, epochs={epochs}) → {sae_dir}", flush=True)
    X_train = X[train_idx]
    model, mean, scale = train_sae(
        X_train,
        feature_dim=feature_dim,
        topk=topk,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )

    sae_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_pt)
    config_json.write_text(
        json.dumps({"dim": X.shape[1], "feature_dim": feature_dim, "k": topk, "seed": seed})
    )
    np.savez_compressed(scaler_npz, mean=mean, scale=scale)
    print(f"  SAE saved → {sae_dir}", flush=True)

    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "k": topk,
        "feature_dim": feature_dim,
    }


@torch.inference_mode()
def encode_sae(bundle: dict, X: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    """Encode dense embeddings to SAE sparse codes."""
    xs = (X - bundle["mean"]) / bundle["scale"]
    outs = []
    for i in range(0, len(xs), bs):
        _, z = bundle["model"](torch.as_tensor(xs[i : i + bs], device=device))
        outs.append(z.cpu().numpy())
    return np.vstack(outs).astype(np.float32)


@torch.inference_mode()
def reconstruct_sae(bundle: dict, X: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    """Decode reconstructed embeddings from SAE (in original embedding space)."""
    xs = (X - bundle["mean"]) / bundle["scale"]
    recs = []
    for i in range(0, len(xs), bs):
        x_hat, _ = bundle["model"](torch.as_tensor(xs[i : i + bs], device=device))
        recs.append(x_hat.cpu().numpy())
    recs_sc = np.vstack(recs).astype(np.float32)
    # Inverse-standardize
    return recs_sc * bundle["scale"] + bundle["mean"]


# ---------------------------------------------------------------------------
# kNN graph (batched, no GPU required for pure numpy)
# ---------------------------------------------------------------------------

def build_knn_with_dists(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) of kNN in X (Euclidean, no self)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    D, I = nn.kneighbors(X, return_distance=True)
    return D[:, 1:], I[:, 1:]


# ---------------------------------------------------------------------------
# SAE curvature metrics
# ---------------------------------------------------------------------------

def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Effective rank via participation ratio of eigenvalue spectrum."""
    ev = eigenvalues[eigenvalues > 0]
    if len(ev) == 0:
        return 0.0
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def compute_sae_curvature(
    Z_raw: np.ndarray,
    C: np.ndarray,
    knn_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute 5 SAE-based curvature metrics for every point.

    Parameters
    ----------
    Z_raw  : (n, D) dense embeddings (L2-normalised)
    C      : (n, F) SAE sparse codes
    knn_idx: (n, k) kNN indices in Z_raw

    Returns
    -------
    dict mapping metric_name -> (n,) float32 array
    """
    n, F = C.shape
    k = knn_idx.shape[1]

    # Binary support masks (active atoms)
    active = (C > 0).astype(np.float32)  # (n, F)

    jaccard_var   = np.zeros(n, dtype=np.float64)
    code_grad     = np.zeros(n, dtype=np.float64)
    local_rank    = np.zeros(n, dtype=np.float64)
    atom_turnover = np.zeros(n, dtype=np.float64)

    for i in range(n):
        nbrs = knn_idx[i]  # (k,)

        # --- Active-set Jaccard variance ---
        a_i = active[i]                 # (F,)
        a_nbrs = active[nbrs]           # (k, F)
        inter = (a_i[None, :] * a_nbrs).sum(axis=1)
        union = (np.maximum(a_i[None, :], a_nbrs)).sum(axis=1).clip(min=1e-9)
        jac = inter / union             # (k,)
        jaccard_var[i] = jac.var()

        # --- Code gradient norm ---
        dz = Z_raw[nbrs] - Z_raw[i]    # (k, D)
        dc = C[nbrs] - C[i]            # (k, F)
        dz_norm = np.linalg.norm(dz, axis=1).clip(min=1e-9)
        dc_norm = np.linalg.norm(dc, axis=1)
        code_grad[i] = (dc_norm / dz_norm).mean()

        # --- Local code rank (participation ratio of code covariance) ---
        local_codes = C[nbrs]           # (k, F)
        centered = local_codes - local_codes.mean(axis=0)
        # Use fast economy SVD: eigenvalues of C^T C / k
        if k >= 2:
            _, sv, _ = np.linalg.svd(centered / np.sqrt(max(k - 1, 1)), full_matrices=False)
            ev = sv ** 2
            local_rank[i] = participation_ratio(ev)
        else:
            local_rank[i] = 0.0

        # --- Atom turnover rate ---
        mean_active = a_nbrs.mean(axis=0)   # consensus activation probability
        # Turnover: fraction of atoms where point i disagrees with neighborhood consensus
        turnover = np.abs(a_i - mean_active).mean()
        atom_turnover[i] = turnover

    # Reconstruction error is computed separately (requires SAE forward pass)
    return {
        "active_set_jaccard_var": jaccard_var.astype(np.float32),
        "code_gradient_norm":     code_grad.astype(np.float32),
        "local_code_rank":        local_rank.astype(np.float32),
        "atom_turnover_rate":     atom_turnover.astype(np.float32),
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
    device: torch.device,
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

    # --- Train or load SAE ---
    sae_dir = _sae_checkpoint_dir(
        root, parquet_path.stem, col,
        args.sae_feature_dim, args.sae_topk, args.seed
    )
    bundle = train_or_load_sae(
        Z, idx_train,
        sae_dir=sae_dir,
        feature_dim=args.sae_feature_dim,
        topk=args.sae_topk,
        epochs=args.sae_epochs,
        batch_size=args.sae_batch_size,
        lr=args.sae_lr,
        device=device,
        seed=args.seed,
    )

    # --- Encode all points ---
    print("  Encoding SAE codes...", flush=True)
    C = encode_sae(bundle, Z, device)   # (n, F)

    # --- Reconstruction error ---
    print("  Computing reconstruction error...", flush=True)
    Z_hat = reconstruct_sae(bundle, Z, device)
    rec_err = np.mean((Z - Z_hat) ** 2, axis=1).astype(np.float32)

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

    # --- kNN graph on test embeddings ---
    print(f"  Building kNN graph (k={args.k_curv}) on test split...", flush=True)
    knn_dists, knn_idx = build_knn_with_dists(Z_test, args.k_curv)
    d_k = knn_dists[:, -1]

    # --- SAE curvature metrics on test split ---
    C_test = C[idx_test]
    rec_err_test = rec_err[idx_test]
    print("  Computing SAE curvature metrics...", flush=True)
    curv_dict = compute_sae_curvature(Z_test, C_test, knn_idx)
    curv_dict["reconstruction_error"] = rec_err_test

    # --- Correlation analysis ---
    tag = model_name.replace("/", "_")
    print(f"\n  Correlation analysis ({tag}):", flush=True)
    summary = correlation_analysis(
        curv_dict, mean_residual, residuals, probe_keys,
        output_dir=out_dir, tag=tag, density_metric=d_k
    )
    summary["model"] = model_name
    summary["col"] = col
    summary["n_train"] = int(len(idx_train))
    summary["n_test"] = int(len(idx_test))
    summary["probe_stats"] = probe_stats
    summary["mean_probe_error"] = float(np.nanmean(mean_residual))
    summary["sae_dir"] = str(sae_dir)
    summary["sae_feature_dim"] = args.sae_feature_dim
    summary["sae_topk"] = args.sae_topk

    # Save per-model curvature arrays
    np.savez_compressed(
        out_dir / f"{tag}_curvature.npz",
        **{k: v for k, v in curv_dict.items()},
        mean_residual=mean_residual,
        idx_test=idx_test,
    )

    return summary


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(summaries: list[dict]) -> dict:
    """Merge per-model Spearman rows and compute mean |ρ| across models."""
    all_metrics = set()
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
        dense_rhos = [r["spearman_rho_dense"] for s in summaries for r in s.get("spearman", []) if r["metric"] == metric and "spearman_rho_dense" in r and not np.isnan(r["spearman_rho_dense"])]
        agg_rows.append({
            "metric": metric,
            "mean_abs_rho": float(np.mean(np.abs(rhos))),
            "mean_auc": float(np.nanmean(aucs)),
            "rhos_per_model": rhos,
            "mean_abs_rho_dense": float(np.mean(np.abs(dense_rhos))) if dense_rhos else float('nan')
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
    p.add_argument("--k-curv", type=int, default=50,
                   help="Number of neighbors for SAE curvature metrics")
    # SAE hyperparameters
    p.add_argument("--sae-feature-dim", type=int, default=2048)
    p.add_argument("--sae-topk", type=int, default=64)
    p.add_argument("--sae-epochs", type=int, default=50)
    p.add_argument("--sae-batch-size", type=int, default=512)
    p.add_argument("--sae-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    root = platonic_root(args.platonic_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "outputs" / "sae_curvature_probe"
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
        probe_keys = [p.strip() for p in args.probes.split(",")]
    print(f"Using {len(probe_keys)} probes.", flush=True)

    # Load labels once (shared across models)
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
            device=device,
            out_dir=out_dir,
        )
        summaries.append(summary)

    agg = aggregate_results(summaries)

    # --- Write results ---
    payload = {
        "experiment": "sae_curvature_probe",
        "args": {
            "max_n": args.max_n,
            "k_curv": args.k_curv,
            "sae_feature_dim": args.sae_feature_dim,
            "sae_topk": args.sae_topk,
            "sae_epochs": args.sae_epochs,
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
        "# SAE Curvature ↔ Physics Probe Error",
        "",
        f"- n_max={args.max_n}, probes={args.probes} ({len(probe_keys)}), "
        f"k_curv={args.k_curv}, SAE F={args.sae_feature_dim} k={args.sae_topk}",
        "",
        "## Aggregated Spearman |ρ| (mean across models)",
        "",
        "| Metric | Mean |ρ| | Mean |ρ| (Dense Q1) | Mean AUC |",
        "|---|---:|---:|---:|",
    ]
    for row in agg["aggregated_spearman"]:
        dense_val = row.get("mean_abs_rho_dense", float('nan'))
        dense_str = f"{dense_val:.3f}" if not np.isnan(dense_val) else "N/A"
        lines.append(f"| {row['metric']} | {row['mean_abs_rho']:.3f} | {dense_str} | {row['mean_auc']:.3f} |")

    for s in summaries:
        lines += [
            "",
            f"## {s['model']} (n_test={s['n_test']})",
            "",
            "| Metric | ρ | ρ (Dense Q1) | p-val | AUC |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in s.get("spearman", []):
            dense_rho = row.get("spearman_rho_dense", float('nan'))
            dense_str = f"{dense_rho:+.3f}" if not np.isnan(dense_rho) else "N/A"
            lines.append(
                f"| {row['metric']} | {row['spearman_rho']:+.3f} | {dense_str} "
                f"| {row['spearman_pval']:.2e} | {row['logistic_auc']:.3f} |"
            )

    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {out_dir}", flush=True)
    print((out_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
