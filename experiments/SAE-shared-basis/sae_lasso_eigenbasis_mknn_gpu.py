#!/usr/bin/env python3
"""Lasso affine eigen/singular-basis experiments A + B + C.

Fit global SAE FISTA Lasso  basis ≈ other @ W + b, save W, then:

  A) Project *standardized* codes into paired singular charts of W_std
     (basis_std @ V_r vs other_std @ U_r); mKNN on test + shuffle/randn controls.
  B) Low-rank transfer ŷ = other @ W_r + b; mKNN vs full W / Ridge.
  C) Local-ball Ridge W + SVD variants with matched vs random-ball controls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import (  # noqa: E402
    ensure_sae_import,
    load_aligned_pair,
    load_col,
    platonic_root,
    resolve_path,
    singular_chart_coords,
)

ensure_sae_import()
from sae_model import TopKSAE  # noqa: E402


def l2n(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(eps)


@torch.inference_mode()
def knn_cos(Z: torch.Tensor, k: int, row_batch: int = 256) -> torch.Tensor:
    Z = l2n(Z)
    n = Z.shape[0]
    out = torch.empty(n, k, device=Z.device, dtype=torch.long)
    for s in range(0, n, row_batch):
        e = min(n, s + row_batch)
        sim = Z[s:e] @ Z.T
        b = e - s
        sim[torch.arange(b, device=Z.device), torch.arange(s, e, device=Z.device)] = (
            -torch.inf
        )
        out[s:e] = torch.topk(sim, k=k, dim=1).indices
    return out


@torch.inference_mode()
def mknn(nn1: torch.Tensor, nn2: torch.Tensor, k: int) -> float:
    a, b = nn1.cpu().numpy(), nn2.cpu().numpy()
    return float(np.mean([len(set(a[i]) & set(b[i])) for i in range(len(a))]) / k)


def load_sae(sae_dir: Path, device: torch.device) -> dict:
    cfg = json.loads((sae_dir / "config.json").read_text())
    sc = np.load(sae_dir / "scaler_stats.npz")
    model = TopKSAE(cfg["dim"], cfg["feature_dim"], cfg["k"]).to(device)
    model.load_state_dict(
        torch.load(sae_dir / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return {
        "model": model,
        "mean": sc["mean"].astype(np.float32),
        "scale": sc["scale"].astype(np.float32),
        "k": int(cfg["k"]),
        "feature_dim": int(cfg["feature_dim"]),
    }


@torch.inference_mode()
def encode(bundle: dict, X: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    xs = (X - bundle["mean"]) / bundle["scale"]
    outs = []
    for i in range(0, len(xs), bs):
        _, z = bundle["model"](torch.as_tensor(xs[i : i + bs], device=device))
        outs.append(z.cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def cosine_rowwise(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), eps)
    return float((an * bn).sum(axis=1).mean())


def idf_np(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    df = (C > 0).sum(axis=0).astype(np.float64)
    return (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)


def soft_thresh(X: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(X) * torch.clamp(X.abs() - lam, min=0.0)


@torch.inference_mode()
def estimate_lipschitz(X: torch.Tensor, n_iter: int = 20) -> float:
    """Spectral norm of X^T X via power iteration → Lip constant for ||XW-Y||^2 / n."""
    f = X.shape[1]
    v = torch.randn(f, device=X.device)
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(n_iter):
        v = X.T @ (X @ v)
        v = v / v.norm().clamp_min(1e-12)
    # grad of (1/n)||XW-Y||^2_F w.r.t W is (2/n) X^T (XW-Y); Lip ≈ (2/n)||X||_op^2
    op = (X @ v).norm().item()
    n = X.shape[0]
    return float(2.0 * (op**2) / max(n, 1))


def fit_fista_lasso(
    codes_basis: np.ndarray,
    codes_other: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    lam: float,
    steps: int,
    device: torch.device,
    seed: int,
) -> dict:
    """basis ≈ other @ W + b with L1 on W (FISTA on standardized features)."""
    torch.manual_seed(seed)
    x_tr = codes_other[train_idx]
    y_tr = codes_basis[train_idx]
    x_te = codes_other[test_idx]
    y_te = codes_basis[test_idx]

    x_scaler = StandardScaler().fit(x_tr)
    y_scaler = StandardScaler().fit(y_tr)
    xs_tr = torch.as_tensor(x_scaler.transform(x_tr), device=device, dtype=torch.float32)
    ys_tr = torch.as_tensor(y_scaler.transform(y_tr), device=device, dtype=torch.float32)
    xs_te = torch.as_tensor(x_scaler.transform(x_te), device=device, dtype=torch.float32)

    f_in, f_out = xs_tr.shape[1], ys_tr.shape[1]
    W = torch.zeros(f_in, f_out, device=device)
    # FISTA state
    Z = W.clone()
    t = 1.0
    L = estimate_lipschitz(xs_tr)
    lr = 1.0 / max(L, 1e-8)
    n = xs_tr.shape[0]
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        # grad of (1/n)||X Z - Y||^2_F
        resid = xs_tr @ Z - ys_tr
        grad = (2.0 / n) * (xs_tr.T @ resid)
        W_new = soft_thresh(Z - lr * grad, lam * lr)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = W_new + ((t - 1.0) / t_new) * (W_new - W)
        W = W_new
        t = t_new
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            mse = float((resid**2).mean())
            nnz = float((W.abs() > 1e-4).float().mean())
            curve.append({"step": step, "mse": mse, "nnz_frac": nnz, "lr": lr})
            print(
                f"    FISTA {step:4d}/{steps}  mse={mse:.5f}  nnz={nnz:.4f}  lr={lr:.3e}",
                flush=True,
            )

    # bias in standardized space = 0 if both centered; map back via scalers
    b_std = torch.zeros(f_out, device=device)
    with torch.no_grad():
        y_hat_te_std = xs_te @ W + b_std
        y_hat_tr_std = xs_tr @ W + b_std
        y_hat_te = y_scaler.inverse_transform(y_hat_te_std.cpu().numpy())
        y_hat_tr = y_scaler.inverse_transform(y_hat_tr_std.cpu().numpy())
        xs_all = torch.as_tensor(
            x_scaler.transform(codes_other), device=device, dtype=torch.float32
        )
        y_hat_all = y_scaler.inverse_transform((xs_all @ W + b_std).cpu().numpy())
        W_np = W.cpu().numpy().astype(np.float32)
        # ambient b for unstandardized: y ≈ (x - mx)/sx @ W * sy + my
        # = x @ (W * sy / sx[:,None]) + (my - mx @ ...)
        sx = x_scaler.scale_.astype(np.float64)
        sy = y_scaler.scale_.astype(np.float64)
        mx = x_scaler.mean_.astype(np.float64)
        my = y_scaler.mean_.astype(np.float64)
        W_raw = (W_np.astype(np.float64) * sy[None, :]) / sx[:, None]
        b_raw = my - mx @ W_raw

    return {
        "W_std": W_np,
        "b_std": np.zeros(f_out, dtype=np.float32),
        "W_raw": W_raw.astype(np.float32),
        "b_raw": b_raw.astype(np.float32),
        "x_mean": x_scaler.mean_.astype(np.float32),
        "x_scale": x_scaler.scale_.astype(np.float32),
        "y_mean": y_scaler.mean_.astype(np.float32),
        "y_scale": y_scaler.scale_.astype(np.float32),
        "y_hat_all": y_hat_all.astype(np.float32),
        "test_cos": cosine_rowwise(y_te, y_hat_te),
        "train_cos": cosine_rowwise(y_tr, y_hat_tr),
        "test_mse": float(np.mean((y_te - y_hat_te) ** 2)),
        "nnz_frac": float((np.abs(W_np) > 1e-4).mean()),
        "lam": lam,
        "steps": steps,
        "fit_sec": time.time() - t0,
        "curve": curve,
        "method": "fista",
    }


def fit_ridge_affine(
    codes_basis: np.ndarray,
    codes_other: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict:
    x_tr = codes_other[train_idx]
    y_tr = codes_basis[train_idx]
    x_te = codes_other[test_idx]
    y_te = codes_basis[test_idx]
    x_scaler = StandardScaler().fit(x_tr)
    y_scaler = StandardScaler().fit(y_tr)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_scaler.transform(x_tr), y_scaler.transform(y_tr))
    # sklearn Ridge: coef_ (n_targets, n_features) → W = coef_.T
    W_std = model.coef_.T.astype(np.float32)
    b_std = model.intercept_.astype(np.float32)
    y_hat_te = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(x_te))
    )
    y_hat_all = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(codes_other))
    )
    sx = x_scaler.scale_.astype(np.float64)
    sy = y_scaler.scale_.astype(np.float64)
    mx = x_scaler.mean_.astype(np.float64)
    my = y_scaler.mean_.astype(np.float64)
    W_raw = (W_std.astype(np.float64) * sy[None, :]) / sx[:, None]
    b_raw = my - mx @ W_raw + b_std.astype(np.float64) * sy
    return {
        "W_std": W_std,
        "b_std": b_std,
        "W_raw": W_raw.astype(np.float32),
        "b_raw": b_raw.astype(np.float32),
        "x_mean": x_scaler.mean_.astype(np.float32),
        "x_scale": x_scaler.scale_.astype(np.float32),
        "y_mean": y_scaler.mean_.astype(np.float32),
        "y_scale": y_scaler.scale_.astype(np.float32),
        "y_hat_all": y_hat_all.astype(np.float32),
        "test_cos": cosine_rowwise(y_te, y_hat_te),
        "nnz_frac": 1.0,
        "lam": alpha,
        "method": "ridge",
    }


def apply_std_affine(
    codes_other: np.ndarray,
    fit: dict,
    W_std: np.ndarray | None = None,
) -> np.ndarray:
    W = fit["W_std"] if W_std is None else W_std
    xs = (codes_other - fit["x_mean"]) / fit["x_scale"]
    ys = xs @ W + fit["b_std"]
    return (ys * fit["y_scale"] + fit["y_mean"]).astype(np.float32)


def svd_W(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # W: (f_in, f_out), other @ W → basis. SVD: W = U_in Σ V_out^T? 
    # numpy: W = U Σ Vt with U (f_in,f_in), Vt (f_out,f_out)
    # For basis ≈ other @ W: columns of W are output directions in basis? 
    # other (n,f_in) @ W (f_in,f_out) → (n,f_out)
    # Right singular vectors of W in input space: V from W = U Σ V^T where
    # standard: M = U Σ Vh with U left (row space of M = input), Vh right (output).
    # So U spans other-space (input), Vh.T spans basis-space (output).
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return U.astype(np.float32), S.astype(np.float32), Vt.astype(np.float32)


def low_rank_W(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, r: int) -> np.ndarray:
    r = min(r, len(S))
    return (U[:, :r] * S[:r]) @ Vt[:r, :]


def shuffle_W(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W2 = W.copy()
    flat = W2.ravel()
    rng.shuffle(flat)
    return flat.reshape(W.shape)


def randn_matched(W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    G = rng.standard_normal(W.shape).astype(np.float32)
    fro_w = float(np.linalg.norm(W))
    fro_g = float(np.linalg.norm(G))
    return G * (fro_w / max(fro_g, 1e-12))


def ball_mknn_np(A: np.ndarray, B: np.ndarray, k: int, device: torch.device) -> float:
    return mknn(
        knn_cos(torch.as_tensor(A, device=device), k),
        knn_cos(torch.as_tensor(B, device=device), k),
        k,
    )


def fit_local_ridge_ambient(
    Xv: np.ndarray,
    Xd: np.ndarray,
    alpha: float = 1.0,
    *,
    fit_idx: np.ndarray | None = None,
    eval_idx: np.ndarray | None = None,
) -> dict:
    """Fit Ridge on fit_idx (default: all); map/eval on eval_idx (default: all).

    Using the same points for fit and mKNN is interpolation — prefer a split.
    """
    if fit_idx is None:
        fit_idx = np.arange(len(Xv))
    if eval_idx is None:
        eval_idx = np.arange(len(Xv))
    Xd_f, Xv_f = Xd[fit_idx], Xv[fit_idx]
    Xd_e, Xv_e = Xd[eval_idx], Xv[eval_idx]
    x_scaler = StandardScaler().fit(Xd_f)
    y_scaler = StandardScaler().fit(Xv_f)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_scaler.transform(Xd_f), y_scaler.transform(Xv_f))
    W_std = model.coef_.T.astype(np.float32)
    b_std = model.intercept_.astype(np.float32)
    mapped_eval = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(Xd_e))
    ).astype(np.float32)
    return {
        "W_std": W_std,
        "b_std": b_std,
        "x_mean": x_scaler.mean_.astype(np.float32),
        "x_scale": x_scaler.scale_.astype(np.float32),
        "y_mean": y_scaler.mean_.astype(np.float32),
        "y_scale": y_scaler.scale_.astype(np.float32),
        "mapped": mapped_eval,
        "Xv_eval": Xv_e.astype(np.float32),
        "Xd_eval": Xd_e.astype(np.float32),
        "fit_idx": np.asarray(fit_idx),
        "eval_idx": np.asarray(eval_idx),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet1", default="data_hf/physics/vit_base_test.parquet")
    p.add_argument("--col1", default="vit_base_galaxies")
    p.add_argument("--parquet2", default="data_hf/physics/dinov3_vitb16_test.parquet")
    p.add_argument("--col2", default="dinov3_vitb16_galaxies")
    p.add_argument(
        "--sae1",
        default="outputs/sae/vit_base_test/vit_base_galaxies/F2048_k64_seed0",
    )
    p.add_argument(
        "--sae2",
        default="outputs/sae/dinov3_vitb16_test/dinov3_vitb16_galaxies/F2048_k64_seed0",
    )
    p.add_argument(
        "--sphere-json",
        default="outputs/local_ii_metric/local_ii_vit_base_galaxies_knn80_dense_pack0p5_d10.json",
    )
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--lams", type=float, nargs="+", default=[0.001, 0.003, 0.01])
    p.add_argument("--fista-steps", type=int, default=400)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument("--n-spheres", type=int, default=150)
    p.add_argument("--ball-knn", type=int, default=256)
    p.add_argument("--skip-C", action="store_true")
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--allow-truncate", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--output-dir",
        default="outputs/sae_lasso_eigenbasis/physics_vit_dino_n16k_F2048_k64",
    )
    args = p.parse_args()

    root = platonic_root(args.platonic_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    def R(path: str) -> Path:
        return resolve_path(root, path)

    out_dir = R(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load + encode ----
    X1, X2 = load_aligned_pair(
        R(args.parquet1),
        args.col1,
        R(args.parquet2),
        args.col2,
        allow_truncate=args.allow_truncate,
    )
    n = len(X1)
    if args.max_n and n > args.max_n:
        sel = np.sort(rng.choice(n, size=args.max_n, replace=False))
        X1, X2 = X1[sel], X2[sel]
        n = args.max_n
        index_map = sel
    else:
        index_map = np.arange(n)

    b1 = load_sae(R(args.sae1), device)
    b2 = load_sae(R(args.sae2), device)
    print("Encoding SAE...", flush=True)
    C1 = encode(b1, X1, device)
    C2 = encode(b2, X2, device)
    print(f"n={n} codes {C1.shape} / {C2.shape}", flush=True)

    idx = np.arange(n)
    train_val_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=args.val_frac,
        random_state=args.seed + 7,
        shuffle=True,
    )
    train_idx, val_idx, test_idx = (
        np.sort(train_idx),
        np.sort(val_idx),
        np.sort(test_idx),
    )
    te = torch.as_tensor(test_idx, device=device, dtype=torch.long)
    print(
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
        flush=True,
    )

    # ---- fit Ridge + FISTA both dirs, save W ----
    fits: dict[str, dict] = {}
    print("\n=== Ridge ===", flush=True)
    fits["ridge_dino_in_vit"] = fit_ridge_affine(
        C1, C2, train_idx=train_idx, test_idx=test_idx, alpha=args.ridge_alpha
    )
    fits["ridge_vit_in_dino"] = fit_ridge_affine(
        C2, C1, train_idx=train_idx, test_idx=test_idx, alpha=args.ridge_alpha
    )
    print(
        f"  ridge dino→vit cos={fits['ridge_dino_in_vit']['test_cos']:.4f}",
        flush=True,
    )
    print(
        f"  ridge vit→dino cos={fits['ridge_vit_in_dino']['test_cos']:.4f}",
        flush=True,
    )

    for lam in args.lams:
        print(f"\n=== FISTA lam={lam} dino→vit ===", flush=True)
        fits[f"fista_{lam:g}_dino_in_vit"] = fit_fista_lasso(
            C1,
            C2,
            train_idx=train_idx,
            test_idx=test_idx,
            lam=lam,
            steps=args.fista_steps,
            device=device,
            seed=args.seed,
        )
        print(
            f"  cos={fits[f'fista_{lam:g}_dino_in_vit']['test_cos']:.4f} "
            f"nnz={fits[f'fista_{lam:g}_dino_in_vit']['nnz_frac']:.4f}",
            flush=True,
        )
        print(f"=== FISTA lam={lam} vit→dino ===", flush=True)
        fits[f"fista_{lam:g}_vit_in_dino"] = fit_fista_lasso(
            C2,
            C1,
            train_idx=train_idx,
            test_idx=test_idx,
            lam=lam,
            steps=args.fista_steps,
            device=device,
            seed=args.seed + 1,
        )
        print(
            f"  cos={fits[f'fista_{lam:g}_vit_in_dino']['test_cos']:.4f} "
            f"nnz={fits[f'fista_{lam:g}_vit_in_dino']['nnz_frac']:.4f}",
            flush=True,
        )

    # Select primary λ per direction by validation cosine (not test).
    primary_keys = []
    primary_lams = {}
    for direction, basis, other in [
        ("dino_in_vit", C1, C2),
        ("vit_in_dino", C2, C1),
    ]:
        best_lam, best_score = None, -1e9
        for lam in args.lams:
            key = f"fista_{lam:g}_{direction}"
            fit = fits[key]
            yhat = apply_std_affine(other, fit)
            bs = basis[val_idx]
            ys = yhat[val_idx]
            bn = bs / np.maximum(np.linalg.norm(bs, axis=1, keepdims=True), 1e-12)
            yn = ys / np.maximum(np.linalg.norm(ys, axis=1, keepdims=True), 1e-12)
            score = float((bn * yn).sum(axis=1).mean())
            fit["val_cos"] = score
            if score > best_score:
                best_score, best_lam = score, lam
        primary_lams[direction] = best_lam
        primary_keys.append(f"fista_{best_lam:g}_{direction}")
        print(
            f"Val-selected {direction}: lam={best_lam:g} (val cos={best_score:.4f})",
            flush=True,
        )
    primary_lam = primary_lams.get("vit_in_dino", args.lams[0])

    # save W packs
    w_dir = out_dir / "weights"
    w_dir.mkdir(exist_ok=True)
    for key, fit in fits.items():
        np.savez_compressed(
            w_dir / f"{key}.npz",
            W_std=fit["W_std"],
            b_std=fit["b_std"],
            W_raw=fit["W_raw"],
            b_raw=fit["b_raw"],
            x_mean=fit["x_mean"],
            x_scale=fit["x_scale"],
            y_mean=fit["y_mean"],
            y_scale=fit["y_scale"],
            lam=np.array([fit.get("lam", 0.0)]),
            test_cos=np.array([fit["test_cos"]]),
            nnz_frac=np.array([fit.get("nnz_frac", 1.0)]),
        )
        U, S, Vt = svd_W(fit["W_std"])
        np.savez_compressed(w_dir / f"{key}_svd.npz", U=U, S=S, Vt=Vt)
        fit["U"], fit["S"], fit["Vt"] = U, S, Vt
        Wstd = fit["W_std"]
        if Wstd.shape[0] == Wstd.shape[1]:
            Ws = 0.5 * (Wstd + Wstd.T)
            evals, evecs = np.linalg.eigh(Ws)
            order = np.argsort(np.abs(evals))[::-1]
            fit["sym_evals"] = evals[order].astype(np.float32)
            fit["sym_evecs"] = evecs[:, order].astype(np.float32)
        else:
            fit["sym_evals"] = None
            fit["sym_evecs"] = None

    # ---- baselines + helpers ----
    X1n = X1 / np.maximum(np.linalg.norm(X1, axis=1, keepdims=True), 1e-12)
    X2n = X2 / np.maximum(np.linalg.norm(X2, axis=1, keepdims=True), 1e-12)
    C1_t = torch.as_tensor(C1, device=device)
    C2_t = torch.as_tensor(C2, device=device)
    Z1 = torch.as_tensor(X1n, device=device)
    Z2 = torch.as_tensor(X2n, device=device)
    idf1 = torch.as_tensor(idf_np(C1[train_idx]), device=device)
    idf2 = torch.as_tensor(idf_np(C2[train_idx]), device=device)

    def add_mknn(rows: list, method: str, A: torch.Tensor, B: torch.Tensor) -> float:
        s = mknn(
            knn_cos(A[te], args.k, args.row_batch),
            knn_cos(B[te], args.k, args.row_batch),
            args.k,
        )
        rows.append({"method": method, "mknn": s})
        print(f"  mknn {method:<55} {s:.4f}", flush=True)
        return s

    baseline_rows: list[dict] = []
    print("\n=== Baselines ===", flush=True)
    add_mknn(baseline_rows, "dense_cosine", Z1, Z2)
    add_mknn(baseline_rows, "sae_codes", C1_t, C2_t)
    add_mknn(baseline_rows, "sae_idf", C1_t * idf1[None], C2_t * idf2[None])
    for key in ["ridge_dino_in_vit", "ridge_vit_in_dino"] + primary_keys:
        fit = fits[key]
        mapped = torch.as_tensor(fit["y_hat_all"], device=device)
        if "dino_in_vit" in key:
            add_mknn(baseline_rows, f"{key}/full_map", C1_t, mapped)
            add_mknn(
                baseline_rows,
                f"{key}/full_map_idf",
                C1_t * idf1[None],
                mapped * idf1[None],
            )
        else:
            add_mknn(baseline_rows, f"{key}/full_map", C2_t, mapped)
            add_mknn(
                baseline_rows,
                f"{key}/full_map_idf",
                C2_t * idf2[None],
                mapped * idf2[None],
            )

    # ===================== A =====================
    print("\n=== A: singular chart (no map) ===", flush=True)
    rows_A: list[dict] = []
    for key in primary_keys:
        fit = fits[key]
        U, S, Vt = fit["U"], fit["S"], fit["Vt"]
        # W: other @ W → basis. U (f_in) spans other, Vt (f_out) spans basis.
        # A: C_basis @ Vt.T[:, :r]  vs  C_other @ U[:, :r]
        if "dino_in_vit" in key:
            C_basis, C_other = C1, C2
            name_basis, name_other = "vit", "dino"
        else:
            C_basis, C_other = C2, C1
            name_basis, name_other = "dino", "vit"

        for kind, W_use, tag in [
            ("true", fit["W_std"], "true"),
            ("shuffle", shuffle_W(fit["W_std"], rng), "shuffle"),
            ("randn", randn_matched(fit["W_std"], rng), "randn"),
        ]:
            Uu, Ss, Vvt = svd_W(W_use)
            for r in args.ranks:
                r = min(r, Uu.shape[1], Vvt.shape[0])
                # W_std SVD must be applied in the *standardized* code spaces.
                Zb, Zo = singular_chart_coords(
                    C_basis, C_other, fit, Uu, Vvt, r
                )
                s = add_mknn(
                    rows_A,
                    f"A/{key}/{tag}/r{r}",
                    torch.as_tensor(Zb, device=device),
                    torch.as_tensor(Zo, device=device),
                )
                zb_all, zo_all = singular_chart_coords(
                    C_basis, C_other, fit, Uu, Vvt, r
                )
                val_s = mknn(
                    knn_cos(
                        torch.as_tensor(zb_all[val_idx], device=device),
                        args.k,
                        args.row_batch,
                    ),
                    knn_cos(
                        torch.as_tensor(zo_all[val_idx], device=device),
                        args.k,
                        args.row_batch,
                    ),
                    args.k,
                )
                rows_A[-1].update(
                    {
                        "fit": key,
                        "control": tag,
                        "r": r,
                        "direction": f"{name_other}_to_{name_basis}",
                        "val_mknn": val_s,
                    }
                )
            # secondary: symmetrized eig projection of both in basis space only makes
            # sense after identifying dims — skip for shuffle/randn of non-square path;
            # for true square W, project both onto sym evecs (invalid across spaces).
            # Report energy of top-r singular values.
            energy = float((Ss[: max(args.ranks)] ** 2).sum() / max((Ss**2).sum(), 1e-12))
            rows_A.append(
                {
                    "method": f"A/{key}/{tag}/singular_energy_top{max(args.ranks)}",
                    "mknn": None,
                    "energy_frac": energy,
                    "control": tag,
                    "fit": key,
                }
            )

    # ===================== B =====================
    print("\n=== B: low-rank W_r transfer ===", flush=True)
    rows_B: list[dict] = []
    for key in primary_keys + ["ridge_dino_in_vit", "ridge_vit_in_dino"]:
        fit = fits[key]
        if "dino_in_vit" in key:
            C_basis_t, C_other = C1_t, C2
            idf = idf1
        else:
            C_basis_t, C_other = C2_t, C1
            idf = idf2

        for kind, W_use, tag in [
            ("true", fit["W_std"], "true"),
            ("shuffle", shuffle_W(fit["W_std"], rng), "shuffle"),
            ("randn", randn_matched(fit["W_std"], rng), "randn"),
        ]:
            Uu, Ss, Vvt = svd_W(W_use)
            # full map with this W
            y_full = apply_std_affine(C_other, fit, W_std=W_use)
            add_mknn(
                rows_B,
                f"B/{key}/{tag}/full",
                C_basis_t,
                torch.as_tensor(y_full, device=device),
            )
            rows_B[-1].update({"fit": key, "control": tag, "r": int(Ss.shape[0])})
            for r in args.ranks:
                Wr = low_rank_W(Uu, Ss, Vvt, r)
                y_r = apply_std_affine(C_other, fit, W_std=Wr)
                add_mknn(
                    rows_B,
                    f"B/{key}/{tag}/r{r}",
                    C_basis_t,
                    torch.as_tensor(y_r, device=device),
                )
                rows_B[-1].update({"fit": key, "control": tag, "r": r})
                add_mknn(
                    rows_B,
                    f"B/{key}/{tag}/r{r}_idf",
                    C_basis_t * idf[None],
                    torch.as_tensor(y_r, device=device) * idf[None],
                )
                rows_B[-1].update({"fit": key, "control": tag, "r": r})

    # ===================== C =====================
    rows_C: list[dict] = []
    if not args.skip_C:
        print("\n=== C: local-ball spectral + random control ===", flush=True)
        sphere = json.loads(R(args.sphere_json).read_text())
        centres_full = np.array([r["point_index"] for r in sphere["rows"]], dtype=np.int64)
        curv = np.array(
            [r["normalized_projector_variance"] for r in sphere["rows"]], float
        )
        # map sphere centres into subsampled index space if needed
        # embeddings X1/X2 are already subsampled; sphere indices refer to full parquet.
        # Rebuild balls on the FULL parquet for correct centres, or remap.
        # Simplest: reload full L2 embeddings for local C (separate from SAE subsample).
        X1_full = load_col(R(args.parquet1), args.col1, l2=True)
        X2_full = load_col(R(args.parquet2), args.col2, l2=True)
        assert X1_full.shape[0] == X2_full.shape[0]

        order = np.arange(len(centres_full))
        q1, q2 = np.quantile(curv, [1 / 3, 2 / 3])
        bands = [
            order[curv <= q1],
            order[(curv > q1) & (curv <= q2)],
            order[curv > q2],
        ]
        per = max(1, args.n_spheres // 3)
        pick = np.sort(
            np.concatenate(
                [rng.choice(b, size=min(per, len(b)), replace=False) for b in bands]
            )
        )
        vit_centres = centres_full[pick]
        rand_dino_centres = rng.choice(
            X2_full.shape[0], size=len(vit_centres), replace=False
        )

        knn_ball = args.ball_knn
        nn1 = NearestNeighbors(n_neighbors=knn_ball + 1, metric="euclidean").fit(X1_full)
        nn2 = NearestNeighbors(n_neighbors=knn_ball + 1, metric="euclidean").fit(X2_full)

        def neigh(nn, X, c):
            idx = nn.kneighbors(X[c : c + 1], return_distance=False)[0]
            return idx[idx != c][:knn_ball]

        local_ranks = [8, 16, 32, 64]
        acc = {
            "native_matched": [],
            "native_random": [],
            "full_affine_matched": [],
            "full_affine_random": [],
            "A_local_matched": {r: [] for r in local_ranks},
            "A_local_random": {r: [] for r in local_ranks},
            "B_local_matched": {r: [] for r in local_ranks},
            "B_local_random": {r: [] for r in local_ranks},
        }

        for i, c in enumerate(vit_centres):
            c = int(c)
            idx_v = neigh(nn1, X1_full, c)
            Xv = X1_full[idx_v]

            # matched — fit on half the ball, evaluate mKNN on held-out half
            Xd_m = X2_full[idx_v]
            m_ball = len(Xv)
            if m_ball < 40:
                continue
            perm = rng.permutation(m_ball)
            n_fit = max(20, int(0.7 * m_ball))
            fit_ix, eval_ix = perm[:n_fit], perm[n_fit:]
            if len(eval_ix) < 10:
                continue
            fit_m = fit_local_ridge_ambient(
                Xv, Xd_m, alpha=args.ridge_alpha, fit_idx=fit_ix, eval_idx=eval_ix
            )
            Um, Sm, Vtm = svd_W(fit_m["W_std"])
            Xv_e, Xd_e = fit_m["Xv_eval"], fit_m["Xd_eval"]
            acc["native_matched"].append(ball_mknn_np(Xv_e, Xd_e, args.k, device))
            acc["full_affine_matched"].append(
                ball_mknn_np(Xv_e, fit_m["mapped"], args.k, device)
            )
            for r in local_ranks:
                rr = min(r, Um.shape[1], Vtm.shape[0])
                Zb, Zo = singular_chart_coords(Xv_e, Xd_e, fit_m, Um, Vtm, rr)
                acc["A_local_matched"][r].append(ball_mknn_np(Zb, Zo, args.k, device))
                Wr = low_rank_W(Um, Sm, Vtm, rr)
                xs = (Xd_e - fit_m["x_mean"]) / fit_m["x_scale"]
                ys = xs @ Wr + fit_m["b_std"]
                mapped_r = (ys * fit_m["y_scale"] + fit_m["y_mean"]).astype(np.float32)
                acc["B_local_matched"][r].append(
                    ball_mknn_np(Xv_e, mapped_r, args.k, device)
                )

            # random DINO centre, rank-paired
            c2 = int(rand_dino_centres[i])
            idx_d = neigh(nn2, X2_full, c2)
            Xd_r = X2_full[idx_d]
            m = min(len(Xv), len(Xd_r))
            Xv_r, Xd_rr = Xv[:m], Xd_r[:m]
            if m < 40:
                continue
            perm = rng.permutation(m)
            n_fit = max(20, int(0.7 * m))
            fit_ix, eval_ix = perm[:n_fit], perm[n_fit:]
            if len(eval_ix) < 10:
                continue
            fit_r = fit_local_ridge_ambient(
                Xv_r,
                Xd_rr,
                alpha=args.ridge_alpha,
                fit_idx=fit_ix,
                eval_idx=eval_ix,
            )
            Ur, Sr, Vtr = svd_W(fit_r["W_std"])
            Xv_e, Xd_e = fit_r["Xv_eval"], fit_r["Xd_eval"]
            acc["native_random"].append(ball_mknn_np(Xv_e, Xd_e, args.k, device))
            acc["full_affine_random"].append(
                ball_mknn_np(Xv_e, fit_r["mapped"], args.k, device)
            )
            for r in local_ranks:
                rr = min(r, Ur.shape[1], Vtr.shape[0])
                Zb, Zo = singular_chart_coords(Xv_e, Xd_e, fit_r, Ur, Vtr, rr)
                acc["A_local_random"][r].append(ball_mknn_np(Zb, Zo, args.k, device))
                Wr = low_rank_W(Ur, Sr, Vtr, rr)
                xs = (Xd_e - fit_r["x_mean"]) / fit_r["x_scale"]
                ys = xs @ Wr + fit_r["b_std"]
                mapped_r = (ys * fit_r["y_scale"] + fit_r["y_mean"]).astype(np.float32)
                acc["B_local_random"][r].append(
                    ball_mknn_np(Xv_e, mapped_r, args.k, device)
                )

            if (i + 1) % 25 == 0 or i + 1 == len(vit_centres):
                print(f"  C spheres {i+1}/{len(vit_centres)}", flush=True)

        def summ(vals):
            a = np.asarray(vals, float)
            return {
                "mean": float(a.mean()),
                "std": float(a.std()),
                "median": float(np.median(a)),
                "n": int(len(a)),
            }

        rows_C.append({"method": "native_matched", **summ(acc["native_matched"])})
        rows_C.append({"method": "native_random", **summ(acc["native_random"])})
        rows_C.append(
            {"method": "full_affine_matched", **summ(acc["full_affine_matched"])}
        )
        rows_C.append(
            {"method": "full_affine_random", **summ(acc["full_affine_random"])}
        )
        for r in local_ranks:
            rows_C.append(
                {"method": f"A_local_matched_r{r}", **summ(acc["A_local_matched"][r])}
            )
            rows_C.append(
                {"method": f"A_local_random_r{r}", **summ(acc["A_local_random"][r])}
            )
            rows_C.append(
                {"method": f"B_local_matched_r{r}", **summ(acc["B_local_matched"][r])}
            )
            rows_C.append(
                {"method": f"B_local_random_r{r}", **summ(acc["B_local_random"][r])}
            )
        for row in rows_C:
            print(
                f"  C {row['method']:<28} mean={row['mean']:.4f} ± {row['std']:.4f}",
                flush=True,
            )

    # ---- write results ----
    def fit_meta(fit: dict) -> dict:
        return {
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in fit.items()
            if k
            not in (
                "W_std",
                "b_std",
                "W_raw",
                "b_raw",
                "y_hat_all",
                "U",
                "S",
                "Vt",
                "sym_evecs",
                "sym_evals",
                "x_mean",
                "x_scale",
                "y_mean",
                "y_scale",
                "curve",
            )
        } | {
            "curve_tail": (fit.get("curve") or [])[-3:],
            "top10_singular": [float(x) for x in fit["S"][:10]],
            "sym_evals_top10": [float(x) for x in fit["sym_evals"][:10]],
        }

    payload = {
        "meta": {
            "n": n,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "lams": args.lams,
            "primary_lam": primary_lam,
            "ranks": args.ranks,
            "k": args.k,
            "seed": args.seed,
            "ball_knn": args.ball_knn,
            "n_spheres": args.n_spheres,
            "protocol": {
                "A": "C_basis @ V_r vs C_other @ U_r from SVD(W); no map applied",
                "B": "ŷ = other @ W_r + b; mKNN vs basis codes",
                "C": "local ambient Ridge W; A/B-local + random-ball control",
            },
        },
        "fits": {k: fit_meta(v) for k, v in fits.items()},
        "baselines": baseline_rows,
        "A": [r for r in rows_A if r.get("mknn") is not None],
        "A_energy": [r for r in rows_A if r.get("mknn") is None],
        "B": rows_B,
        "C": rows_C,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    # markdown
    lines = [
        "# Lasso eigenbasis experiments (A + B + C)",
        "",
        f"- n={n}, train/test={len(train_idx)}/{len(test_idx)}, TopK SAE F={C1.shape[1]} k={b1['k']}",
        f"- FISTA λ∈{args.lams}, steps={args.fista_steps}; primary λ={primary_lam}",
        f"- Ranks r∈{args.ranks}; mKNN k={args.k}",
        "",
        "## Baselines (test mKNN)",
        "",
        "| method | mknn |",
        "|---|---:|",
    ]
    for r in baseline_rows:
        lines.append(f"| {r['method']} | {r['mknn']:.4f} |")

    lines += ["", "## A — Singular chart (no map)", ""]
    lines.append(
        "Compare `C_basis @ V_r` vs `C_other @ U_r` from SVD(`W`) where "
        "`basis ≈ other @ W + b`. Controls: shuffle / randn `W`."
    )
    lines += ["", "| fit | control | r | mknn |", "|---|---|---:|---:|"]
    for r in rows_A:
        if r.get("mknn") is None:
            continue
        lines.append(
            f"| {r.get('fit','')} | {r.get('control','')} | {r.get('r','')} | {r['mknn']:.4f} |"
        )

    # verdict A
    def best_true_A(fit_key):
        # Select rank on validation; report that config's *test* mKNN.
        cands = [
            r
            for r in rows_A
            if r.get("fit") == fit_key
            and r.get("control") == "true"
            and r.get("mknn") is not None
            and r.get("val_mknn") is not None
        ]
        if not cands:
            return float("nan"), None
        best = max(cands, key=lambda r: r["val_mknn"])
        return float(best["mknn"]), best.get("r")

    def best_ctrl_A(fit_key, ctrl, r_star):
        xs = [
            r["mknn"]
            for r in rows_A
            if r.get("fit") == fit_key
            and r.get("control") == ctrl
            and r.get("r") == r_star
            and r.get("mknn") is not None
        ]
        return float(xs[0]) if xs else float("nan")

    lines.append("")
    lines.append(
        "Rank for verdict selected by **validation** mKNN; scores below are test."
    )
    for key in primary_keys:
        bt, r_star = best_true_A(key)
        bs = best_ctrl_A(key, "shuffle", r_star)
        br = best_ctrl_A(key, "randn", r_star)
        if bt > max(bs, br) + 0.02:
            verd = (
                f"**A/{key}:** true singular chart beats controls at val-selected r={r_star} "
                f"(test {bt:.3f} vs shuffle {bs:.3f} / randn {br:.3f})."
            )
        else:
            verd = (
                f"**A/{key}:** no clear edge over controls at r={r_star} "
                f"(true {bt:.3f}, shuffle {bs:.3f}, randn {br:.3f}) — likely artifact."
            )
        lines.append(verd)

    lines += ["", "## B — Low-rank transfer `W_r`", ""]
    lines += ["| fit | control | r | mknn |", "|---|---|---:|---:|"]
    for r in rows_B:
        if "_idf" in r["method"]:
            continue
        lines.append(
            f"| {r.get('fit','')} | {r.get('control','')} | {r.get('r','')} | {r['mknn']:.4f} |"
        )

    lines.append("")
    for key in primary_keys:
        true_full = next(
            (
                r["mknn"]
                for r in rows_B
                if r["method"] == f"B/{key}/true/full"
            ),
            float("nan"),
        )
        sh_full = next(
            (r["mknn"] for r in rows_B if r["method"] == f"B/{key}/shuffle/full"),
            float("nan"),
        )
        # best true low-rank
        lines.append(
            f"**B/{key}:** full={true_full:.3f}; shuffle-full={sh_full:.3f}. "
            f"(Low-rank grid on test is diagnostic only — do not select r on test.)"
        )

    if rows_C:
        lines += ["", "## C — Local ball + random control", ""]
        lines.append(
            f"knn-ball={args.ball_knn}, n_spheres={args.n_spheres}, local ambient Ridge + SVD "
            f"(fit 70% of ball, evaluate mKNN on held-out 30%; singular charts in standardized coords)."
        )
        lines += ["", "| method | mean mknn | std |", "|---|---:|---:|"]
        for r in rows_C:
            lines.append(f"| {r['method']} | {r['mean']:.4f} | {r['std']:.4f} |")
        fa_m = next(r["mean"] for r in rows_C if r["method"] == "full_affine_matched")
        fa_r = next(r["mean"] for r in rows_C if r["method"] == "full_affine_random")
        lines.append("")
        lines.append(
            f"**C full-affine:** matched={fa_m:.3f} vs random={fa_r:.3f} "
            + (
                "(illusory — matched≈random)."
                if abs(fa_m - fa_r) < 0.05
                else "(matched differs from random)."
            )
        )
        # A-local gap
        for r in [8, 16, 32, 64]:
            m = next(
                (x["mean"] for x in rows_C if x["method"] == f"A_local_matched_r{r}"),
                None,
            )
            rr = next(
                (x["mean"] for x in rows_C if x["method"] == f"A_local_random_r{r}"),
                None,
            )
            if m is not None and rr is not None:
                gap = m - rr
                lines.append(
                    f"**C A-local r={r}:** matched={m:.3f} random={rr:.3f} gap={gap:+.3f}."
                )

    lines.append("")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_dir}", flush=True)
    print((out_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
