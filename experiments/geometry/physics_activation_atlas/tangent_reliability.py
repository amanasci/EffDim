"""Tangent-reliability falsification for Physics sphere-normal mean curvature.

Question: is negative K_mean ↔ local OOF global-probe R² genuine second-order
sphere-normal curvature, or first-order tangent leakage from misestimated PCA?
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .confirmatory_object_curvature import decompose_BS
from .curvature_probe_alignment import traceless_B0
from .curvature_probe_screen import partial_spearman, spearman_dict
from .global_probe_curvature_alignment import local_r2_fixed_predictions
from .multimodel_graph_prior_quadratic import knn_torch_ip, load_model_X, l2_normalize
from .paths import platonic_root, resolve_path
from .quadratic import quadratic_features
from .sphere_normal_quadratic import (
    NestedChart,
    _ridge_solve,
    chart_errors,
    normal_projector_apply,
    normalize_rows,
    sphere_project_basis,
)

EPS = 1e-12
SOURCE_MM = "outputs/geometry/physics_multimodel_graph_prior_quadratic"


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def projector(J: np.ndarray) -> np.ndarray:
    return J @ J.T


def grassmann_dist(P1: np.ndarray, P2: np.ndarray, d: int) -> float:
    return float(np.linalg.norm(P1 - P2, "fro") / np.sqrt(max(2 * d, 1)))


def principal_angles(J1: np.ndarray, J2: np.ndarray) -> np.ndarray:
    s = np.clip(np.linalg.svd(J1.T @ J2, compute_uv=False), -1.0, 1.0)
    return np.arccos(s)


@dataclass
class TangentReliabilityConfig:
    output_dir: str = "outputs/geometry/physics_tangent_reliability"
    multimodel_dir: str = SOURCE_MM
    model: str = "vit_base"
    target: str = "mag_r_desi"
    dims: list[int] = field(default_factory=lambda: [8, 12, 16])
    primary_d: int = 16
    k_fit: list[int] = field(default_factory=lambda: [512, 1024, 2048, 3072])
    k_fit_curvature: list[int] = field(default_factory=lambda: [1024, 2048, 3072])
    k_tan: list[int] = field(default_factory=lambda: [128, 256, 512])
    estimators: list[str] = field(
        default_factory=lambda: ["same_patch_pca", "inner_pca", "kernel_pca"]
    )
    n_boot: int = 8
    seed: int = 0
    device: str = "cuda"
    force: bool = False
    stage: str = "all"
    smoke_n_anchors: int = 0  # 0 = all; >0 = first N for smoke
    max_seconds: float = 14400.0
    kernel_bandwidth_frac: float = 0.5  # of knn radius
    grassmann_curvature_thresh: float = 0.15
    assoc_min_n: int = 12

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)

    def mm(self, root: Path) -> Path:
        return resolve_path(root, self.multimodel_dir)


# -------------------- tangent estimators --------------------


def _center_sphere(X: np.ndarray, x0: np.ndarray) -> np.ndarray:
    dx = X - x0
    x0u = x0 / max(np.linalg.norm(x0), EPS)
    return dx - np.outer(dx @ x0u, x0u)


def pca_tangent(
    Xn: np.ndarray,
    x0: np.ndarray,
    d: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return J (D,d), evals descending, diagnostics."""
    dx = _center_sphere(Xn, x0).astype(np.float64)
    if weights is None:
        w = np.ones(len(Xn), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
    w = w / max(w.sum(), EPS)
    Y = np.sqrt(w)[:, None] * dx
    # Gram on feature side when D small relative to n
    n, D = Y.shape
    if n >= D:
        C = Y.T @ Y
        evals, evecs = np.linalg.eigh(C)
        evals = np.maximum(evals[::-1], 0.0)
        V = evecs[:, ::-1]
    else:
        G = Y @ Y.T
        evals, evecs = np.linalg.eigh(G)
        evals = np.maximum(evals[::-1], 0.0)
        evecs = evecs[:, ::-1]
        V = Y.T @ evecs
        V = V / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), EPS)
    J = sphere_project_basis(x0, V[:, :d])
    d_eff = J.shape[1]
    ev = evals / max(n, 1)
    gap = float(ev[d_eff - 1] - ev[d_eff]) if len(ev) > d_eff else float("nan")
    rel_gap = float(gap / max(ev[d_eff - 1], EPS)) if np.isfinite(gap) else float("nan")
    return J, ev, {
        "lambda_d": float(ev[d_eff - 1]) if d_eff else float("nan"),
        "lambda_d1": float(ev[d_eff]) if len(ev) > d_eff else float("nan"),
        "eigengap": gap,
        "rel_eigengap": rel_gap,
        "d_eff": d_eff,
    }


def bootstrap_grassmann_tangent(
    Xn: np.ndarray,
    x0: np.ndarray,
    d: int,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n = len(Xn)
    Ps = []
    Js = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Jb, _, _ = pca_tangent(Xn[idx], x0, d)
        if Jb.shape[1] < max(2, d // 2):
            continue
        # pad/truncate to d columns for projector of rank d_eff
        Ps.append(projector(Jb))
        Js.append(Jb)
    if not Ps:
        J, ev, diag = pca_tangent(Xn, x0, d)
        return J, float("nan"), np.full(d, np.nan), diag
    Pbar = np.mean(Ps, axis=0)
    evals, evecs = np.linalg.eigh(Pbar)
    J = sphere_project_basis(x0, evecs[:, ::-1][:, :d])
    dists = [grassmann_dist(P, Pbar, d) for P in Ps]
    T_boot = float(np.median(dists))
    # angles vs mean
    angs = []
    for Jb in Js:
        if Jb.shape[1] == J.shape[1]:
            angs.append(principal_angles(J, Jb))
    ang_q = np.quantile(np.vstack(angs), [0.5, 0.9], axis=0) if angs else np.full((2, d), np.nan)
    _, ev, diag = pca_tangent(Xn, x0, d)
    diag = {**diag, "T_boot": T_boot, "n_boot_ok": len(Ps)}
    return J, T_boot, ang_q, diag


def kernel_weights(dists: np.ndarray, bandwidth: float) -> np.ndarray:
    """Tricube kernel on distances."""
    u = np.asarray(dists, dtype=np.float64) / max(bandwidth, EPS)
    w = np.where(u < 1.0, (1 - u**3) ** 3, 0.0)
    if w.sum() < EPS:
        w = np.exp(-0.5 * (np.asarray(dists) / max(bandwidth, EPS)) ** 2)
    return w


# -------------------- residual scaling --------------------


def normal_residual_scaling(
    X: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    idx_eval: np.ndarray,
    n_bins: int = 6,
) -> dict:
    if len(idx_eval) < 20:
        return {
            "slope_log": float("nan"),
            "leakage_frac": float("nan"),
            "quad_frac": float("nan"),
            "a1": float("nan"),
            "a2": float("nan"),
            "sigma2": float("nan"),
            "n_eval": int(len(idx_eval)),
        }
    dx = X[idx_eval] - x0
    r = np.linalg.norm(dx, axis=1)
    rn = np.linalg.norm(normal_projector_apply(dx.T, x0, J).T, axis=1)
    m = (r > 1e-6) & (rn > 1e-12) & np.isfinite(r) & np.isfinite(rn)
    r, rn = r[m], rn[m]
    if len(r) < 16:
        return {
            "slope_log": float("nan"),
            "leakage_frac": float("nan"),
            "quad_frac": float("nan"),
            "a1": float("nan"),
            "a2": float("nan"),
            "sigma2": float("nan"),
            "n_eval": int(m.sum()),
        }
    # robust log-log via binned medians
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(r, qs)
    edges = np.unique(edges)
    xs, ys = [], []
    for i in range(len(edges) - 1):
        sel = (r >= edges[i]) & (r <= edges[i + 1] if i == len(edges) - 2 else r < edges[i + 1])
        if sel.sum() < 3:
            continue
        xs.append(np.log(np.median(r[sel])))
        ys.append(np.log(np.median(rn[sel])))
    if len(xs) >= 2:
        slope = float(np.polyfit(xs, ys, 1)[0])
    else:
        slope = float("nan")
    # r_N^2 ≈ σ² + a1 r^2 + a2 r^4, a>=0 via NNLS
    r2 = r**2
    r4 = r**4
    y = rn**2
    A = np.column_stack([np.ones(len(r)), r2, r4])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    sigma2 = max(float(coef[0]), 0.0)
    a1 = max(float(coef[1]), 0.0)
    a2 = max(float(coef[2]), 0.0)
    r_obs = float(np.median(r))
    lin = a1 * r_obs**2
    quad = a2 * r_obs**4
    tot = sigma2 + lin + quad
    return {
        "slope_log": slope,
        "leakage_frac": float(lin / max(tot, EPS)),
        "quad_frac": float(quad / max(tot, EPS)),
        "a1": a1,
        "a2": a2,
        "sigma2": sigma2,
        "n_eval": int(len(r)),
        "r_obs_median": r_obs,
    }


# -------------------- quadratic with fixed tangent --------------------


def fit_nested_fixed_tangent(
    Xloc: np.ndarray,
    x0: np.ndarray,
    J: np.ndarray,
    idx_fit: np.ndarray,
    idx_val: np.ndarray,
    idx_te: np.ndarray,
    ridges: list[float] | None = None,
) -> tuple[NestedChart | None, NestedChart | None, dict]:
    """Fit A/B^S with fixed (x0,J); held-out Δ_S on idx_te."""
    ridges = ridges or [1e-3, 1e-2, 1e-1, 1.0]
    if len(idx_fit) < 20 or len(idx_val) < 8 or J.shape[1] < 2:
        return None, None, {"ok": False, "reason": "too_few"}
    x0 = x0 / max(np.linalg.norm(x0), EPS)
    J = sphere_project_basis(x0, J)
    d = J.shape[1]
    U = (Xloc - x0) @ J
    w = np.ones(len(Xloc), dtype=np.float64)
    sc = np.sqrt(np.maximum((U[idx_fit] ** 2).mean(0), 1e-12))

    def eval_decode(decode, idx):
        if len(idx) == 0:
            return float("inf")
        pred = decode(U[idx])
        return float(np.mean(np.sum((pred - Xloc[idx]) ** 2, axis=1)))

    Phi_f = quadratic_features(U[idx_fit])
    L_f = x0[None, :] + U[idx_fit] @ J.T
    scale = np.linalg.norm(L_f, axis=1, keepdims=True)
    target_un = Xloc[idx_fit] * np.maximum(scale, 1e-8)
    tang_res = (target_un - L_f) @ J
    best_A, lam_A, best_tr = None, ridges[0], float("inf")
    for lam in ridges:
        A = _ridge_solve(Phi_f, tang_res, w[idx_fit], lam)

        def decode_TR(Uloc, _A=A):
            Phi = quadratic_features(Uloc)
            return normalize_rows(x0[None, :] + (Uloc + Phi @ _A.T) @ J.T)

        loss = eval_decode(decode_TR, idx_val)
        if loss < best_tr:
            best_tr, best_A, lam_A = loss, A, lam
    A_flat = best_A
    Phi_f = quadratic_features(U[idx_fit])
    L_tr = x0[None, :] + (U[idx_fit] + Phi_f @ A_flat.T) @ J.T
    scale_tr = np.linalg.norm(L_tr, axis=1, keepdims=True)
    target_tr = Xloc[idx_fit] * np.maximum(scale_tr, 1e-8)
    resid_n = normal_projector_apply((target_tr - L_tr).T, x0, J).T
    best_BS, lam_BS, best_trs = None, ridges[0], float("inf")
    for lam in ridges:
        BS = normal_projector_apply(_ridge_solve(Phi_f, resid_n, w[idx_fit], lam), x0, J)

        def decode_TRS(Uloc, _A=A_flat, _BS=BS):
            Phi = quadratic_features(Uloc)
            Uw = Uloc + Phi @ _A.T
            return normalize_rows(x0[None, :] + Uw @ J.T + Phi @ _BS.T)

        loss = eval_decode(decode_TRS, idx_val)
        if loss < best_trs:
            best_trs, best_BS, lam_BS = loss, BS, lam
    resid_r = normal_projector_apply((target_un - L_f).T, x0, J).T
    best_BSR, lam_BSR = None, ridges[0]
    best_rs = float("inf")
    for lam in ridges:
        BS = normal_projector_apply(_ridge_solve(Phi_f, resid_r, w[idx_fit], lam), x0, J)

        def decode_RS(Uloc, _BS=BS):
            Phi = quadratic_features(Uloc)
            return normalize_rows(x0[None, :] + Uloc @ J.T + Phi @ _BS.T)

        loss = eval_decode(decode_RS, idx_val)
        if loss < best_rs:
            best_rs, best_BSR, lam_BSR = loss, BS, lam
    chart = NestedChart(x0, J, A_flat, best_BS, float(lam_A), float(lam_BS), sc)
    chart_RS = NestedChart(x0, J, np.zeros_like(A_flat), best_BSR, float(lam_A), float(lam_BSR), sc)
    err = chart_errors(chart, chart_RS, Xloc, U, w, idx_te)
    B0, H = traceless_B0(chart.BS_flat, d)
    info = {
        "ok": True,
        "d_eff": d,
        "dS": err["dS"],
        "dT": err["dT"],
        "E_TRS": err["E_TRS"],
        "E_TR": err["E_TR"],
        "K_mean": float(np.linalg.norm(H)),
        "K_traceless": float(np.linalg.norm(B0)),
        "H": H,
        "recon_error": err["E_TRS"],
        **decompose_BS(chart.BS_flat, d),
    }
    return chart, chart_RS, info


def deterministic_splits(
    n: int, seed: int, frac_tan: float = 0.35, frac_fit: float = 0.35, frac_val: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split neighbour indices 0..n-1 into tan/geom, fit, val, te."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tan = max(8, int(frac_tan * n))
    n_fit = max(12, int(frac_fit * n))
    n_val = max(8, int(frac_val * n))
    i0 = 0
    tan = idx[i0 : i0 + n_tan]
    i0 += n_tan
    fit = idx[i0 : i0 + n_fit]
    i0 += n_fit
    val = idx[i0 : i0 + n_val]
    te = idx[i0 + n_val :]
    if len(te) < 8:
        te = val.copy()
    return tan, fit, val, te


# -------------------- synthetics --------------------


def run_synthetic_controls(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    checks = {}

    def make_sphere_base(n, d, D):
        J, _ = np.linalg.qr(rng.normal(size=(D, d)))
        x0 = rng.normal(size=D)
        x0 = x0 - J @ (J.T @ x0)
        x0 /= np.linalg.norm(x0)
        U = rng.normal(size=(n, d)) * 0.25
        return x0, J, U

    # 1) affine + noise: no sphere-normal curvature
    x0, J, U = make_sphere_base(600, 4, 32)
    X = normalize_rows(x0 + U @ J.T + 0.01 * rng.normal(size=(600, 32)))
    Je, _, _ = pca_tangent(X, x0, 4)
    sc = normal_residual_scaling(X, x0, Je, np.arange(200, 600))
    chart, _, info = fit_nested_fixed_tangent(
        X, x0, Je, np.arange(200), np.arange(200, 350), np.arange(350, 600)
    )
    rows.append({"synth": "affine_noise", **sc, "K_mean": info.get("K_mean", np.nan), "dS": info.get("dS", np.nan)})
    checks["affine_near_zero_K"] = bool(info.get("K_mean", 1) < 0.2)

    # 2) affine with rotated tangent → first-order leakage
    R, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    Jbad = sphere_project_basis(x0, R[:, :4])
    Xa = normalize_rows(x0 + U @ J.T)
    sc2 = normal_residual_scaling(Xa, x0, Jbad, np.arange(200, 600))
    rows.append({"synth": "rotated_tangent", **sc2})
    checks["rotated_first_order"] = bool(np.isfinite(sc2["slope_log"]) and sc2["slope_log"] < 1.5)

    # 3) known quadratic curved
    nvec = normal_projector_apply(np.ones(32), x0, J)
    nvec /= max(np.linalg.norm(nvec), EPS)
    Phi = quadratic_features(U)
    q = Phi.shape[1]
    BS = np.zeros((32, q))
    idx = 0
    for a in range(4):
        for b in range(a, 4):
            if a == b:
                BS[:, idx] = 0.6 * nvec
            idx += 1
    Xc = normalize_rows(x0 + U @ J.T + Phi @ BS.T)
    Je3, _, _ = pca_tangent(Xc, Xc.mean(0) / np.linalg.norm(Xc.mean(0)), 4)
    x0c = Xc.mean(0)
    x0c /= np.linalg.norm(x0c)
    sc3 = normal_residual_scaling(Xc, x0c, Je3, np.arange(200, 600))
    _, _, info3 = fit_nested_fixed_tangent(
        Xc, x0c, Je3, np.arange(200), np.arange(200, 350), np.arange(350, 600)
    )
    rows.append({"synth": "quadratic_curved", **sc3, "K_mean": info3.get("K_mean", np.nan)})
    checks["curved_second_order_or_K"] = bool(
        (np.isfinite(sc3["slope_log"]) and sc3["slope_log"] > 1.3) or info3.get("K_mean", 0) > 0.05
    )

    # 4) L2 sphere with only forced radial: B^S ~ 0
    Xr = normalize_rows(x0 + U @ J.T)
    Je4, _, _ = pca_tangent(Xr, x0, 4)
    _, _, info4 = fit_nested_fixed_tangent(
        Xr, x0, Je4, np.arange(200), np.arange(200, 350), np.arange(350, 600)
    )
    rows.append({"synth": "forced_radial_only", "K_mean": info4.get("K_mean", np.nan), "dS": info4.get("dS", np.nan)})
    checks["radial_only_small_BS"] = bool(info4.get("K_mean", 1) < 0.25)

    return {"rows": rows, "checks": checks, "pass": all(checks.values())}


# -------------------- pipeline stages --------------------


def ensure_knn(
    root: Path, cfg: TangentReliabilityConfig, X: np.ndarray, anchors_local: np.ndarray
) -> Path:
    out = cfg.resolved_out(root)
    k_max = max(cfg.k_fit)
    path = out / "cache" / f"{cfg.model}_kmax{k_max}.npz"
    if _done(path, cfg.force):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    # try reuse multimodel 2048 and extend
    mm = cfg.mm(root) / "model_neighbourhoods" / f"{cfg.model}_kmax2048.npz"
    if mm.exists() and k_max > 2048:
        base = np.load(mm)
        # recompute full k_max (fast approx preference noted; exact torch OK for 512 queries)
        idx = knn_torch_ip(X, X[anchors_local], k_max, device, batch=64)
        clean = np.zeros((len(anchors_local), k_max), dtype=np.int64)
        for i, a in enumerate(anchors_local):
            row = [int(j) for j in idx[i] if int(j) != int(a)]
            if len(row) < k_max:
                pad = [j for j in range(len(X)) if j != a and j not in row]
                row.extend(pad[: k_max - len(row)])
            clean[i] = row[:k_max]
        dists = np.linalg.norm(X[clean] - X[anchors_local][:, None, :], axis=2).astype(np.float32)
    else:
        idx = knn_torch_ip(X, X[anchors_local], k_max, device, batch=64)
        clean = np.zeros((len(anchors_local), k_max), dtype=np.int64)
        for i, a in enumerate(anchors_local):
            row = [int(j) for j in idx[i] if int(j) != int(a)]
            if len(row) < k_max:
                pad = [j for j in range(len(X)) if j != a and j not in row]
                row.extend(pad[: k_max - len(row)])
            clean[i] = row[:k_max]
        dists = np.linalg.norm(X[clean] - X[anchors_local][:, None, :], axis=2).astype(np.float32)
    np.savez_compressed(path, anchors_local=anchors_local, neigh=clean, dists=dists)
    print(f"[tanrel] cached knn k_max={k_max}", flush=True)
    return path


def load_context(root: Path, cfg: TangentReliabilityConfig) -> dict:
    mm = cfg.mm(root)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    # prefer all512 list
    all512 = mm / "d_replication_check_all512" / "anchor_ids.json"
    if all512.exists():
        use_sids = json.loads(all512.read_text())["sample_ids"]
    else:
        use_sids = anchors_sid.tolist()
    if cfg.smoke_n_anchors > 0:
        use_sids = use_sids[: cfg.smoke_n_anchors]
    sid_to_ai = {int(s): i for i, s in enumerate(anchors_sid)}
    X = load_model_X(mm, cfg.model)
    folds = pd.read_parquet(mm / "sample_folds.parquet")
    oof = np.load(mm / "global_probes" / "oof_predictions" / f"{cfg.model}_all_targets.npz")
    targets = list(oof["targets"])
    tj = targets.index(cfg.target)
    y = folds[f"y_{cfg.target}"].to_numpy(float)
    pred = oof["oof"][:, tj]
    geo = pd.read_parquet(mm / "local_probe_fields.parquet")
    geo = geo[
        (geo.model == cfg.model)
        & (geo.target == cfg.target)
        & (geo.neighbourhood == "model")
    ]
    return {
        "X": X,
        "anchors_sid": anchors_sid,
        "anchors_local": anchors_local,
        "use_sids": [int(s) for s in use_sids],
        "sid_to_ai": sid_to_ai,
        "y": y,
        "pred": pred,
        "geo": geo,
        "mm": mm,
    }


def local_oof_score(y: np.ndarray, pred: np.ndarray, N: np.ndarray) -> dict:
    yy, pp = y[N], pred[N]
    m = np.isfinite(yy) & np.isfinite(pp)
    if m.sum() < 4:
        return {
            "local_r2": float("nan"),
            "local_label_variance": float("nan"),
            "local_evaluation_count": int(m.sum()),
        }
    return {
        "local_r2": local_r2_fixed_predictions(yy, pp),
        "local_label_variance": float(np.var(yy[m])),
        "local_evaluation_count": int(m.sum()),
    }


def stage_prepare(root: Path, cfg: TangentReliabilityConfig) -> dict:
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "prepare" / "ready.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    ctx = load_context(root, cfg)
    knn_path = ensure_knn(root, cfg, ctx["X"], ctx["anchors_local"])
    meta = {
        "n_anchors": len(ctx["use_sids"]),
        "model": cfg.model,
        "target": cfg.target,
        "knn_path": str(knn_path),
        "multimodel_dir": str(ctx["mm"]),
        "dims": cfg.dims,
        "k_fit": cfg.k_fit,
        "protocol": "oof_global_ridge_mag_r_desi_v1",
        "config_hash": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    }
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps(meta, indent=2))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    return meta


def _estimate_for_cell(
    X: np.ndarray,
    pack: dict,
    ai: int,
    a_local: int,
    k_fit: int,
    d: int,
    estimator: str,
    k_tan: int | None,
    cfg: TangentReliabilityConfig,
    seed: int,
    *,
    do_boot: bool = False,
    do_scale: bool = True,
) -> dict:
    N = pack["neigh"][ai, :k_fit]
    dists = pack["dists"][ai, :k_fit]
    x0 = X[a_local]
    x0 = x0 / max(np.linalg.norm(x0), EPS)
    tan_idx, fit_idx, val_idx, te_idx = deterministic_splits(k_fit, seed)
    T_boot = float("nan")
    ang_q = np.full((2, max(d, 1)), np.nan)
    if estimator == "same_patch_pca":
        Xtan = X[N[tan_idx]]
        J, ev, diag = pca_tangent(Xtan, x0, d)
        if do_boot:
            _, T_boot, ang_q, dboot = bootstrap_grassmann_tangent(
                Xtan, x0, d, cfg.n_boot, seed + 7
            )
            diag = {**diag, "T_boot": T_boot}
    elif estimator == "inner_pca":
        assert k_tan is not None and k_tan < k_fit
        Xtan = X[N[:k_tan]]
        J, ev, diag = pca_tangent(Xtan, x0, d)
        if do_boot:
            _, T_boot, ang_q, dboot = bootstrap_grassmann_tangent(
                Xtan, x0, d, cfg.n_boot, seed + 11
            )
            diag = {**diag, "T_boot": T_boot}
    elif estimator == "kernel_pca":
        bw = float(dists[k_fit - 1] * cfg.kernel_bandwidth_frac)
        w = kernel_weights(dists, bw)
        J, ev, diag = pca_tangent(X[N], x0, d, weights=w)
        diag = {**diag, "bandwidth": bw}
        if do_boot:
            _, T_boot, ang_q, dboot = bootstrap_grassmann_tangent(
                X[N[tan_idx]], x0, d, cfg.n_boot, seed + 13
            )
            diag = {**diag, "T_boot": T_boot}
    elif estimator == "bootstrap_grassmann":
        assert k_tan is not None
        Xtan = X[N[:k_tan]]
        J, T_boot, ang_q, diag = bootstrap_grassmann_tangent(
            Xtan, x0, d, cfg.n_boot, seed + 17
        )
        ev = np.zeros(max(d, 1))
    else:
        raise ValueError(estimator)
    k_half = max(k_tan or (k_fit // 2), d + 5)
    k_half = min(k_half, k_fit - 1)
    J2, _, _ = pca_tangent(X[N[:k_half]], x0, d)
    T_scale = (
        grassmann_dist(projector(J), projector(J2), d)
        if J.shape[1] == J2.shape[1]
        else float("nan")
    )
    scale = (
        normal_residual_scaling(X[N], x0, J, te_idx)
        if do_scale
        else {
            "slope_log": float("nan"),
            "leakage_frac": float("nan"),
            "quad_frac": float("nan"),
            "a1": float("nan"),
            "a2": float("nan"),
            "sigma2": float("nan"),
            "n_eval": 0,
        }
    )
    return {
        "J": J,
        "x0": x0,
        "ev": ev,
        "diag": diag,
        "T_boot": float(diag.get("T_boot", T_boot)),
        "T_scale": T_scale,
        "ang_q50": (
            float(np.nanmedian(ang_q[0]))
            if np.ndim(ang_q) == 2 and np.isfinite(ang_q[0]).any()
            else float("nan")
        ),
        "scale": scale,
        "fit_idx": fit_idx,
        "val_idx": val_idx,
        "te_idx": te_idx,
        "N": N,
        "dists": dists,
    }


def stage_diagnostics(root: Path, cfg: TangentReliabilityConfig) -> None:
    out = cfg.resolved_out(root)
    path = out / "tangent_diagnostics.parquet"
    if _done(path, cfg.force):
        return
    ctx = load_context(root, cfg)
    pack = dict(np.load(out / "cache" / f"{cfg.model}_kmax{max(cfg.k_fit)}.npz"))
    rows = []
    t0 = time.time()
    for si, sid in enumerate(ctx["use_sids"]):
        ai = ctx["sid_to_ai"][int(sid)]
        a_local = int(ctx["anchors_local"][ai])
        if si % 64 == 0:
            print(f"[tanrel][diag] {si}/{len(ctx['use_sids'])}", flush=True)
        for d in cfg.dims:
            for k_fit in cfg.k_fit:
                # Restrict bootstrap_grassmann diagnostics to primary d and large k_fit.
                est_list = ["same_patch_pca", "inner_pca", "kernel_pca"]
                if d == cfg.primary_d and k_fit in (1024, 2048, 3072):
                    est_list.append("bootstrap_grassmann")
                for est in est_list:
                    if est in ("same_patch_pca", "kernel_pca"):
                        ktans: list[int | None] = [None]
                    elif est == "bootstrap_grassmann":
                        ktans = [kt for kt in (256, 512) if kt < k_fit]
                    else:
                        ktans = [kt for kt in cfg.k_tan if kt < k_fit]
                    for k_tan in ktans:
                        seed = cfg.seed + 1009 * ai + 17 * k_fit + d + (k_tan or 0)
                        do_boot = est == "bootstrap_grassmann" or (
                            est == "inner_pca"
                            and d == cfg.primary_d
                            and k_tan == 256
                            and k_fit in (1024, 2048, 3072)
                        )
                        try:
                            cell = _estimate_for_cell(
                                ctx["X"],
                                pack,
                                ai,
                                a_local,
                                k_fit,
                                d,
                                est,
                                k_tan,
                                cfg,
                                seed,
                                do_boot=do_boot,
                                do_scale=True,
                            )
                        except Exception as e:  # noqa: BLE001
                            rows.append(
                                {
                                    "sample_id": sid,
                                    "d": d,
                                    "k_fit": k_fit,
                                    "k_tan": k_tan if k_tan else -1,
                                    "estimator": est,
                                    "error": type(e).__name__,
                                }
                            )
                            continue
                        rows.append(
                            {
                                "sample_id": int(sid),
                                "d": d,
                                "k_fit": k_fit,
                                "k_tan": int(k_tan) if k_tan else -1,
                                "estimator": est,
                                "T_boot": cell["T_boot"],
                                "T_scale": cell["T_scale"],
                                "ang_q50": cell["ang_q50"],
                                "lambda_d": cell["diag"].get("lambda_d", np.nan),
                                "lambda_d1": cell["diag"].get("lambda_d1", np.nan),
                                "eigengap": cell["diag"].get("eigengap", np.nan),
                                "rel_eigengap": cell["diag"].get("rel_eigengap", np.nan),
                                "slope_log": cell["scale"]["slope_log"],
                                "leakage_frac": cell["scale"]["leakage_frac"],
                                "quad_frac": cell["scale"]["quad_frac"],
                                "a1": cell["scale"]["a1"],
                                "a2": cell["scale"]["a2"],
                                "d_eff": cell["diag"].get("d_eff", d),
                            }
                        )
        if time.time() - t0 > cfg.max_seconds * 0.45:
            print("[tanrel][diag] time budget partial save", flush=True)
            break
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[tanrel] diagnostics n={len(rows)} rss={_rss_mb():.0f}", flush=True)


def stage_curvature(root: Path, cfg: TangentReliabilityConfig) -> None:
    out = cfg.resolved_out(root)
    path = out / "curvature_by_tangent.parquet"
    if _done(path, cfg.force):
        return
    ctx = load_context(root, cfg)
    pack = dict(np.load(out / "cache" / f"{cfg.model}_kmax{max(cfg.k_fit)}.npz"))
    diag = pd.read_parquet(out / "tangent_diagnostics.parquet")
    rows = []
    t0 = time.time()
    # bootstrap_grassmann curvature only if its T_scale differs materially from inner_pca
    use_boot_curv = False
    sub = diag[
        (diag.estimator == "bootstrap_grassmann")
        & (diag.d == cfg.primary_d)
        & (diag.k_fit == 2048)
        & (diag.k_tan == 256)
    ]
    sub_in = diag[
        (diag.estimator == "inner_pca")
        & (diag.d == cfg.primary_d)
        & (diag.k_fit == 2048)
        & (diag.k_tan == 256)
    ]
    if len(sub) and len(sub_in):
        mrg = sub[["sample_id", "T_scale"]].merge(
            sub_in[["sample_id", "T_scale"]], on="sample_id", suffixes=("_b", "_i")
        )
        if len(mrg):
            use_boot_curv = bool(
                np.nanmedian(np.abs(mrg.T_scale_b - mrg.T_scale_i)) > cfg.grassmann_curvature_thresh
            )
    estimators = list(cfg.estimators)
    if use_boot_curv and "bootstrap_grassmann" not in estimators:
        estimators.append("bootstrap_grassmann")
    print(f"[tanrel][curv] estimators={estimators} boot_curv={use_boot_curv}", flush=True)

    for si, sid in enumerate(ctx["use_sids"]):
        ai = ctx["sid_to_ai"][int(sid)]
        a_local = int(ctx["anchors_local"][ai])
        if si % 32 == 0:
            print(f"[tanrel][curv] {si}/{len(ctx['use_sids'])}", flush=True)
        for d in cfg.dims:
            for k_fit in cfg.k_fit_curvature:
                for est in estimators:
                    ktans = [None] if est in ("same_patch_pca", "kernel_pca") else [kt for kt in (256, 512) if kt < k_fit]
                    for k_tan in ktans:
                        seed = cfg.seed + 1009 * ai + 17 * k_fit + d + (k_tan or 0) + 99
                        cell = _estimate_for_cell(
                            ctx["X"],
                            pack,
                            ai,
                            a_local,
                            k_fit,
                            d,
                            est,
                            k_tan,
                            cfg,
                            seed,
                            do_boot=False,
                            do_scale=True,
                        )
                        Xloc = ctx["X"][cell["N"]]
                        chart, _, info = fit_nested_fixed_tangent(
                            Xloc,
                            cell["x0"],
                            cell["J"],
                            cell["fit_idx"],
                            cell["val_idx"],
                            cell["te_idx"],
                        )
                        if chart is None or not info.get("ok"):
                            continue
                        score = local_oof_score(ctx["y"], ctx["pred"], cell["N"])
                        rho = float(cell["dists"][k_fit - 1])
                        H = info["H"]
                        rows.append(
                            {
                                "sample_id": int(sid),
                                "d": d,
                                "k_fit": k_fit,
                                "k_tan": int(k_tan) if k_tan else -1,
                                "estimator": est,
                                "K_mean": info["K_mean"],
                                "K_traceless": info["K_traceless"],
                                "dS": info["dS"],
                                "dT": info["dT"],
                                "recon_error": info["recon_error"],
                                "H_norm": float(np.linalg.norm(H)),
                                "mean_fraction": info.get("mean_frac", np.nan),
                                "T_boot": cell["T_boot"],
                                "T_scale": cell["T_scale"],
                                "eigengap": cell["diag"].get("eigengap", np.nan),
                                "rel_eigengap": cell["diag"].get("rel_eigengap", np.nan),
                                "slope_log": cell["scale"]["slope_log"],
                                "leakage_frac": cell["scale"]["leakage_frac"],
                                "quad_frac": cell["scale"]["quad_frac"],
                                "local_r2": score["local_r2"],
                                "local_label_variance": score["local_label_variance"],
                                "local_evaluation_count": score["local_evaluation_count"],
                                "knn_radius": rho,
                                "log_knn_radius": float(np.log(max(rho, EPS))),
                                "H0": float(H[0]) if len(H) else np.nan,
                                "H1": float(H[1]) if len(H) > 1 else np.nan,
                                "H2": float(H[2]) if len(H) > 2 else np.nan,
                            }
                        )
        if time.time() - t0 > cfg.max_seconds * 0.9:
            print("[tanrel][curv] time budget hit", flush=True)
            break
    cdf = pd.DataFrame(rows)
    # attach bootstrap stability from diagnostics where measured
    if len(cdf) and (out / "tangent_diagnostics.parquet").exists():
        ddf = pd.read_parquet(out / "tangent_diagnostics.parquet")
        keys = ["sample_id", "d", "k_fit", "k_tan", "estimator"]
        ddf = ddf[keys + ["T_boot"]].rename(columns={"T_boot": "T_boot_diag"})
        cdf = cdf.merge(ddf, on=keys, how="left")
        miss = ~np.isfinite(cdf["T_boot"].to_numpy(float))
        cdf.loc[miss, "T_boot"] = cdf.loc[miss, "T_boot_diag"]
        cdf = cdf.drop(columns=["T_boot_diag"])
    cdf.to_parquet(path, index=False)
    print(f"[tanrel] curvature n={len(cdf)}", flush=True)


_ASSOC_SCHEMA = [
    "estimator",
    "d",
    "k_fit",
    "k_tan",
    "n",
    "raw_rho",
    "partial_C0",
    "p_partial_C0",
    "median_K_mean",
    "median_dS",
    "frac_dS_pos",
    "median_slope_log",
    "median_leakage",
    "median_T_boot",
]


def stage_associations(root: Path, cfg: TangentReliabilityConfig) -> None:
    out = cfg.resolved_out(root)
    path = out / "probe_associations.parquet"
    if _done(path, cfg.force):
        return
    curv = pd.read_parquet(out / "curvature_by_tangent.parquet")
    n_anch = int(curv.sample_id.nunique()) if len(curv) else 0
    min_n = min(cfg.assoc_min_n, max(5, n_anch))
    rows = []
    for keys, g in curv.groupby(["estimator", "d", "k_fit", "k_tan"]):
        est, d, k_fit, k_tan = keys
        g = g[np.isfinite(g.K_mean) & np.isfinite(g.local_r2)]
        if len(g) < min_n:
            continue
        Km = g.K_mean.to_numpy(float)
        r2 = g.local_r2.to_numpy(float)
        C0 = np.column_stack(
            [
                g.log_knn_radius.to_numpy(float),
                g.local_label_variance.to_numpy(float),
                g.recon_error.to_numpy(float),
                g.local_evaluation_count.to_numpy(float),
            ]
        )
        raw = spearman_dict(Km, r2)
        p0 = partial_spearman(Km, r2, C0)
        extras = {
            "eigengap": g.eigengap.to_numpy(float),
            "T_boot": g.T_boot.to_numpy(float),
            "T_scale": g.T_scale.to_numpy(float),
            "leakage_frac": g.leakage_frac.to_numpy(float),
        }
        path_coefs = {"C0": p0["rho"]}
        for name, col in extras.items():
            # skip all-nan / constant controls
            if not np.isfinite(col).any() or np.nanstd(col) < 1e-12:
                path_coefs[f"C0+{name}"] = float("nan")
                continue
            path_coefs[f"C0+{name}"] = partial_spearman(Km, r2, np.column_stack([C0, col]))["rho"]
        finite_extra = [
            extras[k] for k in extras if np.isfinite(extras[k]).any() and np.nanstd(extras[k]) >= 1e-12
        ]
        if finite_extra:
            path_coefs["C0+all"] = partial_spearman(Km, r2, np.column_stack([C0] + finite_extra))["rho"]
        else:
            path_coefs["C0+all"] = p0["rho"]
        diag_vs_r2 = {f"rho_{name}_r2": spearman_dict(col, r2)["rho"] for name, col in extras.items()}
        rows.append(
            {
                "estimator": est,
                "d": int(d),
                "k_fit": int(k_fit),
                "k_tan": int(k_tan),
                "n": int(len(g)),
                "raw_rho": raw["rho"],
                "partial_C0": p0["rho"],
                "p_partial_C0": p0["pvalue"],
                "median_K_mean": float(np.nanmedian(Km)),
                "median_dS": float(np.nanmedian(g.dS)),
                "frac_dS_pos": float(np.mean(g.dS > 0)),
                "median_slope_log": float(np.nanmedian(g.slope_log)),
                "median_leakage": float(np.nanmedian(g.leakage_frac)),
                "median_T_boot": float(np.nanmedian(g.T_boot)),
                **{f"path_{k}": v for k, v in path_coefs.items()},
                **diag_vs_r2,
            }
        )
    agree_rows = []
    prim = curv[(curv.d == cfg.primary_d) & (curv.k_fit == 2048)] if len(curv) else curv
    if len(prim):
        # collapse k_tan by taking median K_mean per estimator/sample for agreement
        piv = (
            prim.groupby(["sample_id", "estimator"]).K_mean.median().unstack("estimator")
        )
        cols = list(piv.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                st = spearman_dict(piv[a].to_numpy(float), piv[b].to_numpy(float))
                agree_rows.append(
                    {
                        "a": a,
                        "b": b,
                        "spearman_K_mean": st["rho"],
                        "n": st["n"],
                        "d": cfg.primary_d,
                        "k_fit": 2048,
                    }
                )
    adf = pd.DataFrame(rows)
    if adf.empty:
        adf = pd.DataFrame(columns=_ASSOC_SCHEMA)
    adf.to_parquet(path, index=False)
    pd.DataFrame(agree_rows).to_parquet(out / "cross_estimator_agreement.parquet", index=False)
    print(f"[tanrel] associations n={len(rows)} min_n={min_n}", flush=True)


def stage_synth(root: Path, cfg: TangentReliabilityConfig) -> None:
    out = cfg.resolved_out(root)
    path = out / "synthetic_controls.json"
    if _done(path, cfg.force):
        return
    res = run_synthetic_controls(cfg.seed)
    path.write_text(json.dumps(res, indent=2, default=float))
    print(f"[tanrel] synth pass={res['pass']} checks={res['checks']}", flush=True)


def stage_analyze(root: Path, cfg: TangentReliabilityConfig) -> None:
    out = cfg.resolved_out(root)
    assoc = pd.read_parquet(out / "probe_associations.parquet")
    synth = json.loads((out / "synthetic_controls.json").read_text())
    diag = pd.read_parquet(out / "tangent_diagnostics.parquet")
    curv = pd.read_parquet(out / "curvature_by_tangent.parquet")
    agree = (
        pd.read_parquet(out / "cross_estimator_agreement.parquet")
        if (out / "cross_estimator_agreement.parquet").exists()
        else pd.DataFrame()
    )

    label = "tangent_reliability_underpowered"
    reason = "too few completed estimator cells"
    prim = pd.DataFrame()
    same = pd.DataFrame()
    inner = pd.DataFrame()
    med_slope = float("nan")
    med_leak = float("nan")
    if len(assoc) and "d" in assoc.columns:
        prim = assoc[(assoc.d == cfg.primary_d) & (assoc.k_fit == 2048)]
        same = prim[prim.estimator == "same_patch_pca"]
        inner = prim[prim.estimator == "inner_pca"]
        med_slope = float(
            np.nanmedian(diag[(diag.d == cfg.primary_d) & (diag.k_fit == 2048)].slope_log)
        )
        med_leak = float(
            np.nanmedian(diag[(diag.d == cfg.primary_d) & (diag.k_fit == 2048)].leakage_frac)
        )

        def _neg(df):
            if df.empty:
                return False
            return bool(np.nanmedian(df.partial_C0) < -0.08)

        if med_leak > 0.45 and med_slope < 1.4:
            if not same.empty and abs(
                float(same.iloc[0].get("path_C0+leakage_frac", same.iloc[0].partial_C0))
            ) < 0.08:
                label = "pca_tangent_leakage"
                reason = "first-order residual scaling / high leakage; association shrinks after leakage control"
            else:
                label = "mixed_tangent_and_curvature"
                reason = "substantial leakage fraction with residual first-order scaling"
        elif _neg(same) and not _neg(inner):
            label = "broad_secant_geometry"
            reason = "same-patch PCA shows association; independent inner tangents do not"
        elif _neg(same) and _neg(inner):
            cprim = curv[(curv.d == cfg.primary_d) & (curv.k_fit == 2048)]
            dS_ok = float(np.nanmedian(cprim.dS)) > 0 if len(cprim) else False
            slope_ok = med_slope > 1.4
            if dS_ok and (slope_ok or med_leak < 0.35):
                label = "curvature_robust_to_tangent_estimation"
                reason = "negative probe association and positive ΔS recur across estimators"
            else:
                label = "mixed_tangent_and_curvature"
                reason = "association across estimators but residual scaling not cleanly second-order"
        elif len(prim) >= 3:
            label = "tangent_reliability_underpowered"
            reason = "primary cells present but association criteria not met"

    # plots
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    if len(assoc) and "d" in assoc.columns:
        fig1, ax = plt.subplots(figsize=(7, 4))
        for est, g in assoc[assoc.d == cfg.primary_d].groupby("estimator"):
            ax.plot(g.k_fit, g.partial_C0, marker="o", label=est)
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend(fontsize=8)
        ax.set_xlabel("k_fit")
        ax.set_ylabel("partial_C0(K_mean, local_r2)")
        ax.set_title(f"d={cfg.primary_d}")
        fig1.tight_layout()
        fig1.savefig(fig / "partial_C0_by_estimator.png", dpi=140)
        plt.close(fig1)

    kcmp = (
        assoc[(assoc.d == cfg.primary_d) & (assoc.estimator == "same_patch_pca")]
        if len(assoc) and "d" in assoc.columns and "estimator" in assoc.columns
        else pd.DataFrame()
    )
    kcmp_cols = [c for c in ["k_fit", "partial_C0", "raw_rho", "median_dS", "median_leakage"] if c in kcmp.columns]
    n_anch = "n/a"
    aid = cfg.mm(root) / "d_replication_check_all512" / "anchor_ids.json"
    if aid.exists():
        n_anch = str(len(json.loads(aid.read_text())["sample_ids"]))
    diag_sum = "n/a"
    if len(diag):
        diag_sum = (
            diag.groupby(["estimator", "d", "k_fit"])
            .agg(
                {
                    "T_boot": "median",
                    "T_scale": "median",
                    "slope_log": "median",
                    "leakage_frac": "median",
                    "rel_eigengap": "median",
                }
            )
            .reset_index()
            .to_string(index=False)
        )
    report = f"""# Tangent reliability falsification

## Question

Is negative K_mean ↔ local OOF global-probe R² genuine sphere-normal curvature, or PCA tangent leakage?

## Protocol

- Model `{cfg.model}`, target `{cfg.target}`, global OOF ridge only (no local probe refits).
- Anchors: {n_anch} from multimodel `d_replication_check_all512`.
- Primary d={cfg.primary_d}; sensitivity {cfg.dims}.
- k_fit curvature grid {cfg.k_fit_curvature}; diagnostics also at {cfg.k_fit}.
- Sphere-normal projector: P_{{N,S}}=I-P_{{span(x0,J)}}; J⊥x0, J^T J=I.

## Decision label

- `{label}`
- {reason}

## Synthetic controls

pass={synth['pass']}
{json.dumps(synth['checks'], indent=2)}

## Associations (partial_C0 path)

{assoc.to_string(index=False) if len(assoc) else 'empty'}

## Cross-estimator K_mean agreement (d={cfg.primary_d}, k=2048)

{agree.to_string(index=False) if len(agree) else 'n/a'}

## Diagnostics summary (median)

{diag_sum}

## k=3072 vs train-only scale

{kcmp[kcmp_cols].to_string(index=False) if len(kcmp) and kcmp_cols else 'n/a'}

## Strongest defensible interpretation

{reason}
Do not interpret tangent dispersion as extrinsic mean curvature.
Overlap-aware inference: effect sizes above; uncorrected grid p-values are not confirmatory.

## Runtime

See runtime_profile.json.
"""
    (out / "REPORT.md").write_text(report)
    pd.DataFrame([{"label": label, "reason": reason}]).to_csv(out / "decision_labels.csv", index=False)
    print(f"[tanrel] label={label}", flush=True)


def run(cfg: TangentReliabilityConfig, root: Path | None = None) -> dict:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    profile: dict[str, Any] = {"stages": {}}
    stages = {
        "prepare": stage_prepare,
        "synth": stage_synth,
        "diagnostics": stage_diagnostics,
        "curvature": stage_curvature,
        "associations": stage_associations,
        "analyze": stage_analyze,
    }
    order = ["prepare", "synth", "diagnostics", "curvature", "associations", "analyze"]
    want = order if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    deps = {
        "diagnostics": ["prepare"],
        "curvature": ["prepare", "diagnostics"],
        "associations": ["curvature"],
        "analyze": ["associations", "synth", "diagnostics"],
    }
    run_set = set(want)
    for s in want:
        for d in deps.get(s, []):
            run_set.add(d)
    for s in order:
        if s not in run_set:
            continue
        t1 = time.time()
        print(f"[tanrel] stage={s}", flush=True)
        stages[s](root, cfg)
        profile["stages"][f"{s}_s"] = time.time() - t1
        if time.time() - t0 > cfg.max_seconds:
            print("[tanrel] global time budget", flush=True)
            break
    profile.update(
        {
            "total_seconds": time.time() - t0,
            "peak_rss_mb": _rss_mb(),
            "peak_vram_mb": float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else 0.0,
        }
    )
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[tanrel] done in {profile['total_seconds']:.1f}s", flush=True)
    return profile
