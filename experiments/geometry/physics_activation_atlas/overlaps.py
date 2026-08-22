"""Overlap transition maps and consistency diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def soft_intersection(w_a: np.ndarray, w_b: np.ndarray) -> float:
    return float(np.minimum(w_a, w_b).sum())


def fit_weighted_affine(U: np.ndarray, V: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V ≈ U A + b (affine). Returns A (d,d), b (d,)."""
    ww = w / max(w.sum(), 1e-12)
    mu_u = (ww[:, None] * U).sum(axis=0)
    mu_v = (ww[:, None] * V).sum(axis=0)
    Uc = U - mu_u
    Vc = V - mu_v
    # weighted least squares: A = (U^T W U)^+ U^T W V
    Wu = Uc * np.sqrt(ww)[:, None]
    Wv = Vc * np.sqrt(ww)[:, None]
    G = Wu.T @ Wu + 1e-4 * np.eye(U.shape[1])
    A = np.linalg.solve(G, Wu.T @ Wv)
    b = mu_v - mu_u @ A
    return A.astype(np.float64), b.astype(np.float64)


def tangent_disagreement(Qc: np.ndarray, Qcp: np.ndarray) -> float:
    """D_tan = d - ||Qc^T Qcp||_F^2 using orthonormal bases (d ambient? no: (amb,d))."""
    # Qc: (amb, d)
    M = Qc.T @ Qcp
    return float(Qc.shape[1] - np.sum(M**2))


def evaluate_overlaps(
    membership_idx: np.ndarray,
    membership_w: np.ndarray,
    coords: dict[int, np.ndarray],
    bases: dict[int, np.ndarray],
    recon: dict[int, np.ndarray],
    *,
    min_overlap_mass: float = 5.0,
    max_pairs: int = 200,
) -> dict:
    """
    membership_idx: (N, r) chart ids; membership_w: (N, r)
    coords/bases/recon keyed by chart id; arrays aligned to full N (NaN if absent).
    """
    n, r = membership_idx.shape
    # build per-chart weight vectors
    charts = sorted(set(membership_idx.ravel().tolist()))
    W = {c: np.zeros(n, dtype=np.float64) for c in charts}
    for i in range(n):
        for j in range(r):
            c = int(membership_idx[i, j])
            if c < 0:
                continue
            W[c][i] = membership_w[i, j]

    pairs = []
    for ia, ca in enumerate(charts):
        for cb in charts[ia + 1 :]:
            mass = soft_intersection(W[ca], W[cb])
            if mass >= min_overlap_mass:
                pairs.append((ca, cb, mass))
    pairs.sort(key=lambda t: -t[2])
    pairs = pairs[:max_pairs]

    rows = []
    for ca, cb, mass in pairs:
        mask = np.minimum(W[ca], W[cb]) > 1e-6
        if mask.sum() < 8:
            continue
        w = np.minimum(W[ca], W[cb])[mask]
        Ua = coords[ca][mask]
        Ub = coords[cb][mask]
        ok = np.isfinite(Ua).all(axis=1) & np.isfinite(Ub).all(axis=1)
        if ok.sum() < 8:
            continue
        Ua, Ub, w = Ua[ok], Ub[ok], w[ok]
        # train/val split within overlap
        n_ov = len(w)
        n_tr = max(4, int(0.7 * n_ov))
        A, b = fit_weighted_affine(Ua[:n_tr], Ub[:n_tr], w[:n_tr])
        pred = Ua[n_tr:] @ A + b
        mse = float(np.average(np.sum((pred - Ub[n_tr:]) ** 2, axis=1), weights=w[n_tr:])) if n_ov > n_tr else float("nan")
        # cycle
        Ap, bp = fit_weighted_affine(Ub[:n_tr], Ua[:n_tr], w[:n_tr])
        cyc = Ua[:n_tr] @ A + b
        cyc2 = cyc @ Ap + bp
        cycle = float(np.average(np.sum((cyc2 - Ua[:n_tr]) ** 2, axis=1), weights=w[:n_tr]))
        # recon disagreement
        ra = recon[ca][mask][ok]
        rb = recon[cb][mask][ok]
        recon_dis = float(np.average(np.linalg.norm(ra - rb, axis=1), weights=w))
        # tangent
        d_tan = tangent_disagreement(bases[ca], bases[cb])
        cond = float(np.linalg.cond(A))
        rows.append(
            {
                "chart_a": int(ca),
                "chart_b": int(cb),
                "overlap_mass": float(mass),
                "n_overlap": int(ok.sum()),
                "transition_mse": mse,
                "cycle_mse": cycle,
                "recon_disagreement": recon_dis,
                "tangent_disagreement": d_tan,
                "affine_cond": cond,
                "valid": bool(
                    np.isfinite(mse)
                    and mse < 2.0
                    and recon_dis < 0.5
                    and d_tan < 0.5 * bases[ca].shape[1]
                    and cond < 1e4
                ),
            }
        )
    return {"n_pairs": len(rows), "pairs": rows, "min_overlap_mass": min_overlap_mass}


def save_overlaps(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlaps.json").write_text(json.dumps(result, indent=2))
