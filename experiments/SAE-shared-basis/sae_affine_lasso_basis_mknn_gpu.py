#!/usr/bin/env python3
"""L1-regularized (Lasso) affine map between TopK SAE codes → shared basis mKNN.

Same protocol as sae_affine_basis_mknn_gpu.py, but fits
  basis ≈ other @ W + b
with L1 on W (GPU proximal / Adam+L1), plus optional sklearn MultiTaskLasso.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import (  # noqa: E402
    binary_metrics_topk,
    ensure_sae_import,
    load_aligned_pair,
    platonic_root,
    resolve_path,
    ridge_ref_compatible,
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


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    return binary_metrics_topk(y_true, y_pred, k)


def pack_metrics(y: np.ndarray, yhat: np.ndarray, split: str) -> dict:
    return {
        "split": split,
        "mse": float(mean_squared_error(y, yhat)),
        "r2": float(r2_score(y, yhat, multioutput="uniform_average")),
        "cosine": cosine_rowwise(y, yhat),
        "binary": binary_metrics(y, yhat, k=max(1, int((y > 0).sum(axis=1).mean()))),
    }


def fit_lasso_affine_gpu(
    codes_basis: np.ndarray,
    codes_other: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    l1: float,
    steps: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
    val_idx: np.ndarray | None = None,
) -> dict:
    """basis ≈ other @ W + b with L1 on W (Adam on MSE + l1*|W|)."""
    torch.manual_seed(seed)
    x_tr = codes_other[train_idx]
    y_tr = codes_basis[train_idx]
    x_te = codes_other[test_idx]
    y_te = codes_basis[test_idx]

    x_scaler = StandardScaler().fit(x_tr)
    y_scaler = StandardScaler().fit(y_tr)
    xs_tr = torch.as_tensor(x_scaler.transform(x_tr), device=device)
    ys_tr = torch.as_tensor(y_scaler.transform(y_tr), device=device)
    xs_te = torch.as_tensor(x_scaler.transform(x_te), device=device)

    f_in, f_out = xs_tr.shape[1], ys_tr.shape[1]
    W = torch.zeros(f_in, f_out, device=device, requires_grad=True)
    b = torch.zeros(f_out, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    n = xs_tr.shape[0]
    curve = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        pred = xs_tr[idx] @ W + b
        mse = torch.mean((pred - ys_tr[idx]) ** 2)
        loss = mse + l1 * W.abs().mean()
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            with torch.no_grad():
                nnz = float((W.abs() > 1e-4).float().mean())
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "mse": float(mse.detach()),
                    "nnz_frac": nnz,
                }
            )
            print(
                f"    step {step:4d}  mse={mse.item():.5f}  loss={loss.item():.5f}  "
                f"nnz_frac={nnz:.4f}",
                flush=True,
            )

    with torch.no_grad():
        W_final = W.detach().clone()
        b_final = b.detach().clone()
        ys_hat_te = xs_te @ W_final + b_final
        ys_hat_tr = xs_tr @ W_final + b_final
        y_hat_te = y_scaler.inverse_transform(ys_hat_te.cpu().numpy())
        y_hat_tr = y_scaler.inverse_transform(ys_hat_tr.cpu().numpy())
        xs_all = torch.as_tensor(x_scaler.transform(codes_other), device=device)
        y_hat_all = y_scaler.inverse_transform((xs_all @ W_final + b_final).cpu().numpy())

        W_np = W_final.cpu().numpy()
        nnz_frac = float((np.abs(W_np) > 1e-4).mean())
        row_nnz = float((np.abs(W_np).max(axis=1) > 1e-4).mean())
        col_nnz = float((np.abs(W_np).max(axis=0) > 1e-4).mean())
        v = np.random.default_rng(0).standard_normal(W_np.shape[1])
        for _ in range(8):
            v = W_np.T @ (W_np @ v)
            v = v / (np.linalg.norm(v) + 1e-12)
        op_norm = float(np.linalg.norm(W_np @ v))
        fro = float(np.linalg.norm(W_np))
        eff_rank_proxy = float((fro / max(op_norm, 1e-12)) ** 2)

    out = {
        "y_hat_all": y_hat_all.astype(np.float32),
        "train": pack_metrics(y_tr, y_hat_tr, "train"),
        "test": pack_metrics(y_te, y_hat_te, "test"),
        "l1": l1,
        "nnz_frac": nnz_frac,
        "row_active_frac": row_nnz,
        "col_active_frac": col_nnz,
        "coef_fro": fro,
        "coef_op_norm_est": op_norm,
        "effective_rank_proxy": eff_rank_proxy,
        "curve": curve,
        "method": "gpu_l1_adam",
    }
    if val_idx is not None and len(val_idx):
        x_va = codes_other[val_idx]
        y_va = codes_basis[val_idx]
        y_hat_va = y_scaler.inverse_transform(
            (torch.as_tensor(x_scaler.transform(x_va), device=device) @ W_final + b_final)
            .cpu()
            .numpy()
        )
        out["val"] = pack_metrics(y_va, y_hat_va, "val")
    return out


def fit_multitask_lasso(
    codes_basis: np.ndarray,
    codes_other: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict:
    from sklearn.linear_model import MultiTaskLasso

    x_tr = codes_other[train_idx]
    y_tr = codes_basis[train_idx]
    x_te = codes_other[test_idx]
    y_te = codes_basis[test_idx]
    x_scaler = StandardScaler().fit(x_tr)
    y_scaler = StandardScaler().fit(y_tr)
    xs_tr = x_scaler.transform(x_tr)
    ys_tr = y_scaler.transform(y_tr)
    xs_te = x_scaler.transform(x_te)

    # MultiTaskLasso: y = X @ coef_ + intercept_; coef_ shape (n_features, n_targets) in recent sklearn?
    # Actually coef_ is (n_targets, n_features) for MultiTaskLasso
    model = MultiTaskLasso(
        alpha=alpha, fit_intercept=True, max_iter=2000, tol=1e-3, selection="random"
    )
    t0 = time.time()
    model.fit(xs_tr, ys_tr)
    fit_sec = time.time() - t0
    print(f"    MultiTaskLasso fit {fit_sec:.1f}s", flush=True)

    y_hat_te = y_scaler.inverse_transform(model.predict(xs_te))
    y_hat_tr = y_scaler.inverse_transform(model.predict(xs_tr))
    y_hat_all = y_scaler.inverse_transform(model.predict(x_scaler.transform(codes_other)))

    coef = model.coef_  # (n_targets, n_features)
    W = coef.T
    nnz_frac = float((np.abs(W) > 1e-6).mean())
    return {
        "y_hat_all": y_hat_all.astype(np.float32),
        "train": pack_metrics(y_tr, y_hat_tr, "train"),
        "test": pack_metrics(y_te, y_hat_te, "test"),
        "l1": alpha,
        "nnz_frac": nnz_frac,
        "row_active_frac": float((np.abs(W).max(axis=1) > 1e-6).mean()),
        "col_active_frac": float((np.abs(W).max(axis=0) > 1e-6).mean()),
        "coef_fro": float(np.linalg.norm(W)),
        "fit_sec": fit_sec,
        "method": "sklearn_multitask_lasso",
    }


def idf_np(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    df = (C > 0).sum(axis=0).astype(np.float64)
    return (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)


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
    p.add_argument("--max-n", type=int, default=16384)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument(
        "--l1-coefs",
        type=float,
        nargs="+",
        default=[1e-3, 3e-3, 1e-2, 3e-2],
        help="L1 strengths for GPU Adam-Lasso",
    )
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--mtl-alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.05],
        help="sklearn MultiTaskLasso alphas (can be slow)",
    )
    p.add_argument("--skip-mtl", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--allow-truncate", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of train used as val for λ selection")
    p.add_argument(
        "--ridge-ref",
        default=None,
        help="Optional prior Ridge results.json; only used if meta matches this run",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/sae_affine_lasso_basis/physics_vit_dino_n16k_F2048_k64",
    )
    args = p.parse_args()

    root = platonic_root(args.platonic_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    def R(p: str) -> Path:
        return resolve_path(root, p)

    X1, X2 = load_aligned_pair(
        R(args.parquet1),
        args.col1,
        R(args.parquet2),
        args.col2,
        allow_truncate=args.allow_truncate,
    )
    n = len(X1)
    rng = np.random.default_rng(args.seed)
    if args.max_n and n > args.max_n:
        sel = np.sort(rng.choice(n, size=args.max_n, replace=False))
        X1, X2 = X1[sel], X2[sel]
        n = args.max_n

    b1 = load_sae(R(args.sae1), device)
    b2 = load_sae(R(args.sae2), device)
    print("Encoding...", flush=True)
    C1 = encode(b1, X1, device)
    C2 = encode(b2, X2, device)
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
    print(
        f"n={n} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
        flush=True,
    )

    ridge_ref = None
    if args.ridge_ref:
        ridge_path = R(args.ridge_ref)
        if ridge_path.is_file():
            cand = json.loads(ridge_path.read_text())
            if ridge_ref_compatible(
                cand, n=n, col1=args.col1, col2=args.col2, seed=args.seed
            ):
                ridge_ref = cand
                print(f"Using compatible Ridge ref: {ridge_path}", flush=True)
            else:
                print(
                    f"Ignoring Ridge ref {ridge_path}: meta does not match "
                    f"(n/col1/col2/seed).",
                    flush=True,
                )

    fits = []  # list of (tag, direction, block)

    for l1 in args.l1_coefs:
        print(f"\n=== GPU L1 Adam  l1={l1}  DINO→ViT ===", flush=True)
        d2v = fit_lasso_affine_gpu(
            C1,
            C2,
            train_idx=train_idx,
            test_idx=test_idx,
            val_idx=val_idx,
            l1=l1,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        print(
            f"  val cos={d2v['val']['cosine']:.4f} test cos={d2v['test']['cosine']:.4f} "
            f"jacc={d2v['test']['binary']['jaccard_at_k']:.4f} nnz={d2v['nnz_frac']:.4f}",
            flush=True,
        )
        fits.append((f"gpu_l1_{l1:g}", "dino_in_vit", d2v))

        print(f"=== GPU L1 Adam  l1={l1}  ViT→DINO ===", flush=True)
        v2d = fit_lasso_affine_gpu(
            C2,
            C1,
            train_idx=train_idx,
            test_idx=test_idx,
            val_idx=val_idx,
            l1=l1,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + 1,
        )
        print(
            f"  val cos={v2d['val']['cosine']:.4f} test cos={v2d['test']['cosine']:.4f} "
            f"jacc={v2d['test']['binary']['jaccard_at_k']:.4f} nnz={v2d['nnz_frac']:.4f}",
            flush=True,
        )
        fits.append((f"gpu_l1_{l1:g}", "vit_in_dino", v2d))

    if not args.skip_mtl:
        for alpha in args.mtl_alphas:
            print(f"\n=== MultiTaskLasso α={alpha} DINO→ViT ===", flush=True)
            try:
                d2v = fit_multitask_lasso(
                    C1, C2, train_idx=train_idx, test_idx=test_idx, alpha=alpha
                )
                print(
                    f"  test cos={d2v['test']['cosine']:.4f} nnz={d2v['nnz_frac']:.4f}",
                    flush=True,
                )
                fits.append((f"mtl_{alpha:g}", "dino_in_vit", d2v))
            except Exception as exc:  # noqa: BLE001
                print(f"  MTL failed: {exc}", flush=True)

    # Select best L1 per direction on validation cosine (GPU fits only)
    selected = {}
    for direction in ("dino_in_vit", "vit_in_dino"):
        cands = [
            (tag, block)
            for tag, d, block in fits
            if d == direction and block.get("val") is not None
        ]
        if cands:
            tag, block = max(cands, key=lambda tb: tb[1]["val"]["cosine"])
            selected[direction] = tag
            print(
                f"Val-selected {direction}: {tag} "
                f"(val cos={block['val']['cosine']:.4f})",
                flush=True,
            )

    Z1 = torch.as_tensor(X1, device=device)
    Z2 = torch.as_tensor(X2, device=device)
    Z1n = Z1 / Z1.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Z2n = Z2 / Z2.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C1_t = torch.as_tensor(C1, device=device)
    C2_t = torch.as_tensor(C2, device=device)
    te = torch.as_tensor(test_idx, device=device, dtype=torch.long)
    idf1 = torch.as_tensor(idf_np(C1[train_idx]), device=device)
    idf2 = torch.as_tensor(idf_np(C2[train_idx]), device=device)

    mknn_rows = []

    def add(method: str, A: torch.Tensor, B: torch.Tensor, *, selected_flag: bool = False) -> None:
        s = mknn(
            knn_cos(A[te], args.k, args.row_batch),
            knn_cos(B[te], args.k, args.row_batch),
            args.k,
        )
        mknn_rows.append({"method": method, "mknn": s, "val_selected": selected_flag})
        print(f"  mknn {method:<48} {s:.4f}", flush=True)

    print("\n=== mKNN (test; λ selected on val) ===", flush=True)
    add("dense_cosine", Z1n, Z2n)
    add("sae_codes_cosine", C1_t, C2_t)
    add("sae_idf_cosine", C1_t * idf1[None], C2_t * idf2[None])

    for tag, direction, block in fits:
        mapped = torch.as_tensor(block["y_hat_all"], device=device)
        is_sel = selected.get(direction) == tag
        if direction == "dino_in_vit":
            add(
                f"{tag}/shared_vit_basis",
                C1_t,
                mapped,
                selected_flag=is_sel,
            )
            add(
                f"{tag}/shared_vit_basis_idf",
                C1_t * idf1[None],
                mapped * idf1[None],
                selected_flag=is_sel,
            )
        else:
            add(
                f"{tag}/shared_dino_basis",
                C2_t,
                mapped,
                selected_flag=is_sel,
            )
            add(
                f"{tag}/shared_dino_basis_idf",
                C2_t * idf2[None],
                mapped * idf2[None],
                selected_flag=is_sel,
            )

    out_dir = R(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "n": n,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "l1_coefs": args.l1_coefs,
            "mtl_alphas": args.mtl_alphas if not args.skip_mtl else [],
            "steps": args.steps,
            "lr": args.lr,
            "k": args.k,
            "seed": args.seed,
            "col1": args.col1,
            "col2": args.col2,
            "sae_k": b1["k"],
            "val_selected": selected,
            "selection_rule": "max val cosine per direction",
        },
        "ridge_ref_mknn": (ridge_ref.get("mknn_rows") if ridge_ref else None),
        "fits": [
            {
                "tag": tag,
                "direction": direction,
                **{k: v for k, v in block.items() if k not in ("y_hat_all", "curve")},
                "curve_tail": (block.get("curve") or [])[-3:],
            }
            for tag, direction, block in fits
        ],
        "mknn_rows": mknn_rows,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Affine Lasso SAE-code basis transfer",
        "",
        f"- n={n}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}, TopK k={b1['k']}",
        f"- GPU L1 Adam l1∈{args.l1_coefs}, steps={args.steps}",
        "- Hyperparameter selection: **max validation cosine** per direction",
        "",
        "## Code prediction",
        "",
        "| fit | direction | val cos | test cos | Jaccard | nnz(W) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for tag, direction, block in fits:
        vc = block.get("val", {}).get("cosine", float("nan"))
        teb = block["test"]
        mark = " ★" if selected.get(direction) == tag else ""
        lines.append(
            f"| {tag}{mark} | {direction} | {vc:.4f} | {teb['cosine']:.4f} | "
            f"{teb['binary']['jaccard_at_k']:.4f} | {block['nnz_frac']:.4f} |"
        )
    if ridge_ref:
        lines += [
            "",
            "Compatible Ridge ref: "
            f"DINO→ViT cos={ridge_ref['dino_in_vit']['test']['cosine']:.4f}, "
            f"ViT→DINO cos={ridge_ref['vit_in_dino']['test']['cosine']:.4f}",
        ]
    lines += [
        "",
        "## mKNN (test)",
        "",
        "| method | mknn | val_selected |",
        "|---|---:|:---:|",
    ]
    for r in mknn_rows:
        star = "yes" if r.get("val_selected") else ""
        lines.append(f"| {r['method']} | {r['mknn']:.4f} | {star} |")
    sel_rows = [r for r in mknn_rows if r.get("val_selected")]
    if sel_rows:
        best = max(sel_rows, key=lambda r: r["mknn"])
        lines += [
            "",
            f"Best **val-selected** shared-basis: `{best['method']}` mknn={best['mknn']:.4f}.",
            "(Full table includes non-selected λ for diagnostics; do not pick Best from it.)",
            "",
        ]
    if ridge_ref:
        rb = max(ridge_ref["mknn_rows"], key=lambda r: r["mknn"])
        lines.append(
            f"Compatible Ridge ref best: `{rb['method']}` mknn={rb['mknn']:.4f}."
        )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_dir}", flush=True)
    print((out_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
