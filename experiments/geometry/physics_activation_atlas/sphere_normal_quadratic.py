"""Nested sphere-normal quadratic models: separate warping vs sphere-normal bending."""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .coordinates import encode_chart
from .data import load_prepare
from .metrics import weighted_mse
from .paths import platonic_root, resolve_path
from .quadratic import n_quad_features, quadratic_features
from .quadratic_structure import matrix_rank_stats, principal_angle_overlap, truncate_B


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


def normalize_rows(Y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    return (Y / np.maximum(n, eps)).astype(np.float64)


def sphere_project_basis(x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    """J <- orth((I - x0 x0^T) J)."""
    x0 = x0 / max(np.linalg.norm(x0), 1e-12)
    Jp = J - np.outer(x0, x0 @ J)
    # thin QR for orthonormalization
    Q, R = np.linalg.qr(Jp, mode="reduced")
    # drop near-zero columns
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.abs(R)
    keep = diag > 1e-8 * (diag.max() if diag.size else 1.0)
    if not np.any(keep):
        return Q
    return Q[:, keep]


def normal_projector_apply(V: np.ndarray, x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    """P_{N,S} V = V - Proj_{span(x0,J)} V. V shape (D,) or (D,k)."""
    x0 = (x0 / max(np.linalg.norm(x0), 1e-12)).astype(np.float64)
    Q, _ = np.linalg.qr(np.column_stack([x0, J]), mode="reduced")
    if V.ndim == 1:
        return V - Q @ (Q.T @ V)
    return V - Q @ (Q.T @ V)


def pack_symmetric_weights(d: int) -> np.ndarray:
    """Weights for flattening B^S: 1 on diagonal, sqrt(2) off-diagonal."""
    w = []
    for a in range(d):
        for b in range(a, d):
            w.append(1.0 if a == b else np.sqrt(2.0))
    return np.asarray(w, dtype=np.float64)


def flatten_BS_for_svd(BS: np.ndarray, d: int) -> np.ndarray:
    """
    BS: (D, q) with q=d(d+1)/2 in order a<=b.
    Return metric-aware matrix (D, q) with off-diagonal columns scaled by sqrt(2).
    """
    w = pack_symmetric_weights(d)
    return BS * w[None, :]


@dataclass
class NestedChart:
    x0: np.ndarray  # (D,)
    J: np.ndarray  # (D, d) orthonormal, sphere-tangent
    A_flat: np.ndarray  # (d, q) tangential warping
    BS_flat: np.ndarray  # (D, q) sphere-normal bending (in im P_NS)
    ridge_A: float
    ridge_BS: float
    coord_scale: np.ndarray  # (d,) whitening scales for rank analysis

    def phi(self, U: np.ndarray) -> np.ndarray:
        return quadratic_features(U)

    def warp(self, U: np.ndarray) -> np.ndarray:
        return U + self.phi(U) @ self.A_flat.T

    def decode_R(self, U: np.ndarray) -> np.ndarray:
        return normalize_rows(self.x0[None, :] + U @ self.J.T)

    def decode_TR(self, U: np.ndarray) -> np.ndarray:
        Uw = self.warp(U)
        return normalize_rows(self.x0[None, :] + Uw @ self.J.T)

    def decode_RS(self, U: np.ndarray) -> np.ndarray:
        Y = self.x0[None, :] + U @ self.J.T + self.phi(U) @ self.BS_flat.T
        return normalize_rows(Y)

    def decode_TRS(self, U: np.ndarray) -> np.ndarray:
        Uw = self.warp(U)
        Y = self.x0[None, :] + Uw @ self.J.T + self.phi(U) @ self.BS_flat.T
        return normalize_rows(Y)


def _ridge_solve(Phi: np.ndarray, Target: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Solve min ||sqrt(w)(Phi B^T - Target)|| + lam||B||; Target (N,k), return B (k,q)."""
    sw = np.sqrt(np.maximum(w, 0.0))
    sw = sw / max(np.linalg.norm(sw), 1e-12) * np.sqrt(len(sw))
    Pw = Phi * sw[:, None]
    Tw = Target * sw[:, None]
    G = Pw.T @ Pw + lam * np.eye(Pw.shape[1])
    Rhs = Pw.T @ Tw
    try:
        return np.linalg.solve(G, Rhs).T
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, Rhs, rcond=None)[0].T


def _ridge_multi(Phi: np.ndarray, Target: np.ndarray, w: np.ndarray, ridges: list[float]):
    """Train-loss ridge pick (fallback for nulls). Prefer validation decode pick in fit_nested_chart."""
    best_B, best_lam, best_loss = None, ridges[0], float("inf")
    for lam in ridges:
        B = _ridge_solve(Phi, Target, w, lam)
        pred = Phi @ B.T
        loss = float(np.sum(w * ((pred - Target) ** 2).sum(1)) / max(w.sum(), 1e-12))
        if loss < best_loss:
            best_loss, best_B, best_lam = loss, B, lam
    return best_B, float(best_lam), best_loss


def fit_nested_chart(
    X: np.ndarray,
    U: np.ndarray,
    w: np.ndarray,
    idx_geom: np.ndarray,
    idx_fit: np.ndarray,
    idx_val: np.ndarray,
    *,
    ridges: list[float] | None = None,
) -> tuple[NestedChart, NestedChart, dict, np.ndarray]:
    """
    Cross-fit style:
      idx_geom: estimate x0, J
      idx_fit: fit A, B^S
      idx_val: choose ridges via held-out nested decode MSE
    """
    ridges = ridges or [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0]
    wg = w[idx_geom]
    wg = wg / max(wg.sum(), 1e-12)
    x0 = (wg[:, None] * X[idx_geom]).sum(axis=0)
    x0 = x0 / max(np.linalg.norm(x0), 1e-12)
    Xc = X[idx_geom] - x0
    Xc = Xc - np.outer(Xc @ x0, x0)
    Y = np.sqrt(wg)[:, None] * Xc
    _, _, vt = np.linalg.svd(Y, full_matrices=False)
    d = U.shape[1]
    J = sphere_project_basis(x0, vt[:d].T)
    d = J.shape[1]
    U_all = ((X - x0) @ J).astype(np.float64)
    sc = np.sqrt(np.maximum((wg[:, None] * (U_all[idx_geom] ** 2)).sum(axis=0), 1e-12))

    def eval_decode(decode, idx):
        ww = w[idx]
        if ww.sum() <= 0 or len(idx) == 0:
            return float("inf")
        pred = decode(U_all[idx])
        return weighted_mse(pred.astype(np.float32), X[idx].astype(np.float32), ww)

    Phi_f = quadratic_features(U_all[idx_fit])
    L_f = x0[None, :] + U_all[idx_fit] @ J.T
    scale = np.linalg.norm(L_f, axis=1, keepdims=True)
    target_un = X[idx_fit] * np.maximum(scale, 1e-8)
    tang_res = (target_un - L_f) @ J

    # Select A by validation E_TR
    A_cands = {lam: _ridge_solve(Phi_f, tang_res, w[idx_fit], lam) for lam in ridges}
    best_A, lam_A, best_tr = None, ridges[0], float("inf")
    for lam, A in A_cands.items():

        def decode_TR(Uloc, _A=A):
            Phi = quadratic_features(Uloc)
            return normalize_rows(x0[None, :] + (Uloc + Phi @ _A.T) @ J.T)

        loss = eval_decode(decode_TR, idx_val)
        if loss < best_tr:
            best_tr, best_A, lam_A = loss, A, lam
    A_flat = best_A

    L_tr = x0[None, :] + (U_all[idx_fit] + Phi_f @ A_flat.T) @ J.T
    scale_tr = np.linalg.norm(L_tr, axis=1, keepdims=True)
    target_tr = X[idx_fit] * np.maximum(scale_tr, 1e-8)
    resid_n = normal_projector_apply((target_tr - L_tr).T, x0, J).T

    # Select B^S by validation E_TRS (given A)
    BS_cands = {
        lam: normal_projector_apply(_ridge_solve(Phi_f, resid_n, w[idx_fit], lam), x0, J)
        for lam in ridges
    }
    best_BS, lam_BS, best_trs = None, ridges[0], float("inf")
    for lam, BS in BS_cands.items():

        def decode_TRS(Uloc, _A=A_flat, _BS=BS):
            Phi = quadratic_features(Uloc)
            Uw = Uloc + Phi @ _A.T
            return normalize_rows(x0[None, :] + Uw @ J.T + Phi @ _BS.T)

        loss = eval_decode(decode_TRS, idx_val)
        if loss < best_trs:
            best_trs, best_BS, lam_BS = loss, BS, lam
    BS_flat = best_BS

    resid_r = normal_projector_apply((target_un - L_f).T, x0, J).T
    BSR_cands = {
        lam: normal_projector_apply(_ridge_solve(Phi_f, resid_r, w[idx_fit], lam), x0, J)
        for lam in ridges
    }
    best_BSR, lam_BSR, best_rs = None, ridges[0], float("inf")
    for lam, BS in BSR_cands.items():

        def decode_RS(Uloc, _BS=BS):
            Phi = quadratic_features(Uloc)
            return normalize_rows(x0[None, :] + Uloc @ J.T + Phi @ _BS.T)

        loss = eval_decode(decode_RS, idx_val)
        if loss < best_rs:
            best_rs, best_BSR, lam_BSR = loss, BS, lam

    chart = NestedChart(
        x0=x0, J=J, A_flat=A_flat, BS_flat=BS_flat, ridge_A=float(lam_A), ridge_BS=float(lam_BS), coord_scale=sc
    )
    chart_RS = NestedChart(
        x0=x0,
        J=J,
        A_flat=np.zeros_like(A_flat),
        BS_flat=best_BSR,
        ridge_A=float(lam_A),
        ridge_BS=float(lam_BSR),
        coord_scale=sc,
    )
    info = {
        "ridge_A": float(lam_A),
        "ridge_BS": float(lam_BS),
        "ridge_BS_R": float(lam_BSR),
        "d": d,
        "n_geom": int(len(idx_geom)),
        "n_fit": int(len(idx_fit)),
        "n_val": int(len(idx_val)),
        "val_E_R": eval_decode(chart.decode_R, idx_val),
        "val_E_TR": eval_decode(chart.decode_TR, idx_val),
        "val_E_RS": eval_decode(chart_RS.decode_RS, idx_val),
        "val_E_TRS": eval_decode(chart.decode_TRS, idx_val),
    }
    return chart, chart_RS, info, U_all


def chart_errors(
    chart: NestedChart,
    chart_RS: NestedChart,
    X: np.ndarray,
    U: np.ndarray,
    w: np.ndarray,
    idx: np.ndarray,
) -> dict:
    ww = w[idx]
    if len(idx) == 0 or ww.sum() <= 0:
        return {k: float("nan") for k in ["E_R", "E_TR", "E_RS", "E_TRS", "dT", "dSR", "dS"]}
    XR = X[idx]
    E_R = weighted_mse(chart.decode_R(U[idx]).astype(np.float32), XR.astype(np.float32), ww)
    E_TR = weighted_mse(chart.decode_TR(U[idx]).astype(np.float32), XR.astype(np.float32), ww)
    E_RS = weighted_mse(chart_RS.decode_RS(U[idx]).astype(np.float32), XR.astype(np.float32), ww)
    E_TRS = weighted_mse(chart.decode_TRS(U[idx]).astype(np.float32), XR.astype(np.float32), ww)
    return {
        "E_R": E_R,
        "E_TR": E_TR,
        "E_RS": E_RS,
        "E_TRS": E_TRS,
        "dT": float(E_R - E_TR),
        "dSR": float(E_R - E_RS),
        "dS": float(E_TR - E_TRS),
        "n": int(len(idx)),
        "w_sum": float(ww.sum()),
    }


def sample_deltas_S(
    chart: NestedChart, X: np.ndarray, U: np.ndarray, w: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """Per-sample E_TR - E_TRS (squared error)."""
    if len(idx) == 0:
        return np.zeros(0)
    pred_tr = chart.decode_TR(U[idx])
    pred_trs = chart.decode_TRS(U[idx])
    e_tr = ((pred_tr - X[idx]) ** 2).sum(1)
    e_trs = ((pred_trs - X[idx]) ** 2).sum(1)
    return e_tr - e_trs


def mc_pvalue(real: float, nulls: np.ndarray) -> tuple[float, int]:
    nulls = np.asarray(nulls, dtype=np.float64)
    nulls = nulls[np.isfinite(nulls)]
    B = len(nulls)
    if B == 0 or not np.isfinite(real):
        return float("nan"), 0
    return float((1 + np.sum(nulls >= real)) / (B + 1)), B


def predictive_rank_curve(
    chart: NestedChart,
    X: np.ndarray,
    U: np.ndarray,
    w: np.ndarray,
    idx: np.ndarray,
) -> list[dict]:
    """Truncate B^S and measure held-out Δ_S vs full."""
    ranks = [0, 1, 2, 4, 8, 16, "full"]
    full = chart_errors(chart, chart, X, U, w, idx)  # RS unused
    # recompute with truncations
    d = chart.J.shape[1]
    BS = chart.BS_flat
    E_TR = weighted_mse(
        chart.decode_TR(U[idx]).astype(np.float32), X[idx].astype(np.float32), w[idx]
    ) if len(idx) else float("nan")
    rows = []
    dS_full = None
    for r in ranks:
        ch = NestedChart(
            x0=chart.x0,
            J=chart.J,
            A_flat=chart.A_flat,
            BS_flat=np.zeros_like(BS) if r == 0 else (BS if r == "full" else truncate_B(BS, int(r))),
            ridge_A=chart.ridge_A,
            ridge_BS=chart.ridge_BS,
            coord_scale=chart.coord_scale,
        )
        E_TRS = weighted_mse(
            ch.decode_TRS(U[idx]).astype(np.float32), X[idx].astype(np.float32), w[idx]
        ) if len(idx) else float("nan")
        dS = float(E_TR - E_TRS)
        if r == "full":
            dS_full = dS
        rows.append({"rank": str(r), "E_TRS": E_TRS, "dS": dS, "E_TR": E_TR})
    for row in rows:
        row["frac_of_full_dS"] = float(row["dS"] / (dS_full + 1e-12)) if dS_full not in (None, 0) else float("nan")
    return rows


def one_se_rank(rows: list[dict], se: float) -> str:
    """Smallest rank within 1 SE of best (lowest E_TRS / highest dS)."""
    best = max(r["dS"] for r in rows if np.isfinite(r["dS"]))
    thresh = best - se
    for r in rows:
        if r["rank"] == "full":
            continue
        if r["dS"] >= thresh:
            return r["rank"]
    return "full"


# -------------------- synthetics --------------------


def _embed_sphere_normalize(Y: np.ndarray) -> np.ndarray:
    return normalize_rows(Y)


def synth_normalized_affine(n: int, d: int, D: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(D, d))
    J, _ = np.linalg.qr(J)
    x0 = rng.normal(size=D)
    x0 /= np.linalg.norm(x0)
    # make x0 not in span J
    x0 = x0 - J @ (J.T @ x0)
    x0 /= np.linalg.norm(x0)
    U = rng.normal(size=(n, d)) * 0.3
    Y = x0[None, :] + U @ J.T
    return _embed_sphere_normalize(Y), U.astype(np.float64)


def synth_warped_patch(n: int, d: int, D: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(D, d))
    J, _ = np.linalg.qr(J)
    x0 = rng.normal(size=D)
    x0 = x0 - J @ (J.T @ x0)
    x0 /= np.linalg.norm(x0)
    U = rng.normal(size=(n, d)) * 0.3
    # tangential warp A
    A = 0.8 * rng.normal(size=(d, n_quad_features(d)))
    Phi = quadratic_features(U)
    Uw = U + Phi @ A.T
    Y = x0[None, :] + Uw @ J.T
    return _embed_sphere_normalize(Y), U.astype(np.float64)


def synth_sphere_normal_curve(n: int, d: int, D: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(D, d))
    J, _ = np.linalg.qr(J)
    x0 = rng.normal(size=D)
    x0 = x0 - J @ (J.T @ x0)
    x0 /= np.linalg.norm(x0)
    # one normal direction
    nvec = rng.normal(size=D)
    nvec = normal_projector_apply(nvec, x0, J)
    nvec /= max(np.linalg.norm(nvec), 1e-12)
    U = rng.normal(size=(n, d)) * 0.35
    Phi = quadratic_features(U)
    # B^S: only first quadratic mode (u_0^2) along nvec
    BS = np.zeros((D, n_quad_features(d)))
    BS[:, 0] = 1.2 * nvec
    Y = x0[None, :] + U @ J.T + Phi @ BS.T
    return _embed_sphere_normalize(Y), U.astype(np.float64), BS


def run_synthetic_controls(seed: int = 0) -> dict:
    rows = []
    for name, maker in [
        ("normalized_affine", synth_normalized_affine),
        ("coordinate_warped", synth_warped_patch),
    ]:
        X, Utrue = maker(1200, 4, 32, seed)
        w = np.ones(len(X))
        idx = np.arange(len(X))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        g, f, v, te = np.split(idx, [300, 750, 975])
        chart, chart_RS, info, U = fit_nested_chart(X, Utrue, w, g, f, v)
        err = chart_errors(chart, chart_RS, X, U, w, te)
        rows.append({"synth": name, **err, **{k: info[k] for k in ("ridge_A", "ridge_BS")}})
    X, Utrue, BS_true = synth_sphere_normal_curve(1200, 4, 32, seed + 3)
    w = np.ones(len(X))
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed + 3)
    rng.shuffle(idx)
    g, f, v, te = np.split(idx, [300, 750, 975])
    chart, chart_RS, info, U = fit_nested_chart(X, Utrue, w, g, f, v)
    err = chart_errors(chart, chart_RS, X, U, w, te)
    Uq, _, _ = np.linalg.svd(flatten_BS_for_svd(chart.BS_flat, chart.J.shape[1]), full_matrices=False)
    n_true = BS_true[:, 0]
    n_true /= max(np.linalg.norm(n_true), 1e-12)
    align = float(abs(np.dot(Uq[:, 0], n_true))) if Uq.size else float("nan")
    rows.append(
        {
            "synth": "sphere_normal_curved",
            **err,
            "normal_mode_alignment": align,
            **{k: info[k] for k in ("ridge_A", "ridge_BS")},
        }
    )
    ok = {
        "normalized_affine_dS_near0": abs(rows[0]["dS"]) <= max(1e-4, 0.25 * abs(rows[0]["dT"]) + 1e-4),
        "warped_dT_positive_dS_small": rows[1]["dT"] > 0.001 and abs(rows[1]["dS"]) < 0.5 * rows[1]["dT"],
        "curved_dS_positive": rows[2]["dS"] > 0.001,
        "curved_mode_recovered": align > 0.5,
    }
    return {"rows": rows, "checks": ok, "pass": all(ok.values())}


# -------------------- pipeline --------------------


@dataclass
class SphereNormalConfig:
    stage: str = "all"
    output_dir: str = "outputs/geometry/physics_quadratic_atlas_sphere_normal"
    structure_dir: str = "outputs/geometry/physics_quadratic_atlas_structure"
    ablation_prepare: str = "outputs/geometry/physics_activation_atlas_geometry_ablation/prepare"
    primary: str = "n6_d8"
    configs: list[str] = field(
        default_factory=lambda: ["n6_d8", "n4_d8", "n8_d8", "n6_d6", "n6_d10", "n6_d12"]
    )
    n_bootstrap: int = 40
    n_null: int = 40
    seed: int = 0
    force: bool = False
    max_seconds: float = 7200.0

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


def _budget(t0: float, cfg: SphereNormalConfig, where: str) -> None:
    if time.time() - t0 > cfg.max_seconds:
        raise RuntimeError(f"Hard stop at {where}: {time.time()-t0:.1f}s")


def load_cached_config(root: Path, cfg: SphereNormalConfig, cfg_id: str):
    gdir = resolve_path(root, cfg.structure_dir) / "grid" / cfg_id
    if not gdir.exists():
        raise FileNotFoundError(gdir)
    W = sparse.load_npz(gdir / "memberships_csr.npz")
    data = load_prepare(resolve_path(root, cfg.ablation_prepare))
    return W, data, gdir


def fit_config_nested(
    root: Path,
    cfg: SphereNormalConfig,
    cfg_id: str,
    *,
    bootstrap_seed: int | None = None,
) -> dict:
    W, data, gdir = load_cached_config(root, cfg, cfg_id)
    X = data["X"]
    tr_all = data["train_local"]
    te = data["test_local"]
    seed = cfg.seed if bootstrap_seed is None else bootstrap_seed
    rng = np.random.default_rng(seed)
    if bootstrap_seed is not None:
        tr_all = rng.choice(tr_all, size=len(tr_all), replace=True)
        tr_all = np.unique(tr_all)
    # three-way split of train for cross-fitting
    tr = np.asarray(tr_all, dtype=np.int64).copy()
    rng.shuffle(tr)
    n = len(tr)
    n_g = max(20, int(0.4 * n))
    n_f = max(20, int(0.4 * n))
    idx_geom, idx_fit, idx_val = tr[:n_g], tr[n_g : n_g + n_f], tr[n_g + n_f :]
    if len(idx_val) < 10:
        idx_val = idx_fit.copy()

    d = int(cfg_id.split("_")[1][1:])
    chart_rows = []
    sample_dS = []
    rank_rows = []
    BS_list = []

    for c in range(W.shape[1]):
        w = np.asarray(W[:, c].todense()).ravel()
        # initial U placeholder (d) — fit_nested re-encodes
        U_dummy = np.zeros((len(X), d), dtype=np.float64)
        # restrict geom/fit/val to chart members
        def memb(idx):
            return idx[w[idx] > 1e-6]

        g, f, v = memb(idx_geom), memb(idx_fit), memb(idx_val)
        te_c = memb(te)
        if len(g) < d + 5 or len(f) < 20:
            continue
        chart, chart_RS, info, U = fit_nested_chart(X, U_dummy, w, g, f, v)
        err = chart_errors(chart, chart_RS, X, U, w, te_c)
        dS_samp = sample_deltas_S(chart, X, U, w, te_c)
        sample_dS.extend(
            [{"chart": c, "index": int(i), "dS": float(dv), "w": float(w[i])} for i, dv in zip(te_c, dS_samp)]
        )
        # rank stats on metric-whitened coords: scale columns of BS by coord scales in features
        # Whiten: use U' = U / sc, then B' features change — approximate by scaling BS columns
        sc = chart.coord_scale
        # feature scales: for u_a u_b, scale by sc_a * sc_b
        feat_scale = []
        dd = chart.J.shape[1]
        for a in range(dd):
            for b in range(a, dd):
                feat_scale.append(sc[a] * sc[b])
        feat_scale = np.asarray(feat_scale)
        BS_w = chart.BS_flat * feat_scale[None, :]
        BS_flat_svd = flatten_BS_for_svd(BS_w, dd)
        stats = matrix_rank_stats(BS_flat_svd)
        svals = np.linalg.svd(BS_flat_svd, compute_uv=False)
        curve = predictive_rank_curve(chart, X, U, w, te_c)
        se = float(np.std(dS_samp) / np.sqrt(max(len(dS_samp), 1)))
        r_1se = one_se_rank(curve, se)
        r90 = "full"
        for row in curve:
            if row["rank"] != "full" and row["frac_of_full_dS"] >= 0.9:
                r90 = row["rank"]
                break
        n_feat = n_quad_features(dd)
        n_eff = float((w[f].sum() ** 2) / max((w[f] ** 2).sum(), 1e-12))
        chart_rows.append(
            {
                "config_id": cfg_id,
                "chart": c,
                **err,
                **info,
                "frac_samples_dS_pos": float(np.mean(dS_samp > 0)) if len(dS_samp) else float("nan"),
                "BS_stable_rank": stats["stable_rank"],
                "BS_entropy_rank": stats["entropy_rank"],
                "BS_participation_ratio": stats["participation_ratio"],
                "BS_rank90": stats["rank90"],
                "BS_rank95": stats["rank95"],
                "pred_rank_1se": r_1se,
                "pred_rank_90pct_dS": r90,
                "n_eff_fit": n_eff,
                "n_eff_per_quad_feature": n_eff / max(n_feat, 1),
                "n_quad_features": n_feat,
                "BS_svals": svals.tolist(),
            }
        )
        for row in curve:
            rank_rows.append({**row, "config_id": cfg_id, "chart": c})
        BS_list.append(chart.BS_flat)

    dS_vals = np.array([r["dS"] for r in chart_rows], dtype=np.float64)
    return {
        "config_id": cfg_id,
        "charts": chart_rows,
        "rank_curve": rank_rows,
        "sample_dS": sample_dS,
        "mean_dS": float(np.nanmean(dS_vals)) if len(dS_vals) else float("nan"),
        "median_dS": float(np.nanmedian(dS_vals)) if len(dS_vals) else float("nan"),
        "frac_charts_dS_pos": float(np.mean(dS_vals > 0)) if len(dS_vals) else float("nan"),
        "mean_dT": float(np.nanmean([r["dT"] for r in chart_rows])) if chart_rows else float("nan"),
        "mean_dSR": float(np.nanmean([r["dSR"] for r in chart_rows])) if chart_rows else float("nan"),
        "BS_list": BS_list,
    }


def stage_synthetic(root: Path, cfg: SphereNormalConfig) -> dict:
    out = cfg.resolved_out(root) / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    if _done(out / "synthetic.json", cfg.force):
        return json.loads((out / "synthetic.json").read_text())
    res = run_synthetic_controls(cfg.seed)
    (out / "synthetic.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(res["rows"]).to_csv(out / "synthetic_controls.csv", index=False)
    return res


def stage_grid(root: Path, cfg: SphereNormalConfig, t0: float) -> dict:
    out = cfg.resolved_out(root) / "grid"
    marker = out / "grid_summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_charts = []
    all_rank = []
    for cfg_id in cfg.configs:
        _budget(t0, cfg, cfg_id)
        print(f"[sphere-normal] fit {cfg_id}", flush=True)
        res = fit_config_nested(root, cfg, cfg_id)
        # drop BS_list from json
        s = {k: v for k, v in res.items() if k not in ("BS_list", "sample_dS", "rank_curve", "charts")}
        s["n_charts"] = len(res["charts"])
        summaries.append(s)
        all_rank.extend(res["rank_curve"])
        charts_safe = []
        spec = []
        for row in res["charts"]:
            row = dict(row)
            svals = row.get("BS_svals") or []
            for i, sv in enumerate(svals):
                spec.append(
                    {
                        "config_id": cfg_id,
                        "chart": int(row["chart"]),
                        "mode": i,
                        "singular_value": float(sv),
                    }
                )
            row["BS_svals"] = json.dumps(svals)
            charts_safe.append(row)
        if spec:
            pd.DataFrame(spec).to_csv(out / f"{cfg_id}_BS_spectra.csv", index=False)
        pd.DataFrame(charts_safe).to_parquet(out / f"{cfg_id}_charts.parquet", index=False)
        all_charts.extend(charts_safe)
        pd.DataFrame(res["rank_curve"]).to_parquet(out / f"{cfg_id}_rank_curve.parquet", index=False)
        pd.DataFrame(res["sample_dS"]).to_parquet(out / f"{cfg_id}_sample_dS.parquet", index=False)
        # save BS
        if res["BS_list"]:
            np.savez_compressed(out / f"{cfg_id}_BS.npz", **{f"BS_{i}": B for i, B in enumerate(res["BS_list"])})
        print(
            f"[sphere-normal] {cfg_id} dS={s['mean_dS']:.5f} dT={s['mean_dT']:.5f} "
            f"charts+={s['frac_charts_dS_pos']:.2f}",
            flush=True,
        )
    pd.DataFrame(all_charts).to_parquet(out / "nested_model_errors.parquet", index=False)
    pd.DataFrame(all_rank).to_parquet(out / "predictive_rank.parquet", index=False)
    summary = {"configs": summaries}
    (out / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_bootstrap_nulls(root: Path, cfg: SphereNormalConfig, t0: float) -> dict:
    out = cfg.resolved_out(root) / "bootstrap_nulls"
    marker = out / "summary.json"
    if _done(marker, cfg.force):
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    cfg_id = cfg.primary
    W, data, gdir = load_cached_config(root, cfg, cfg_id)
    X = data["X"]
    tr_all = data["train_local"]
    te = data["test_local"]
    d = int(cfg_id.split("_")[1][1:])

    boot_rows = []
    # reference BS for principal angles
    ref = fit_config_nested(root, cfg, cfg_id)
    BS_ref = ref["BS_list"]

    for b in range(cfg.n_bootstrap):
        _budget(t0, cfg, f"boot{b}")
        res = fit_config_nested(root, cfg, cfg_id, bootstrap_seed=cfg.seed + 1000 + b)
        overlaps = []
        for c, BS in enumerate(res["BS_list"]):
            if c < len(BS_ref):
                for k in (1, 2, 4):
                    overlaps.append(
                        principal_angle_overlap(
                            flatten_BS_for_svd(BS, d),
                            flatten_BS_for_svd(BS_ref[c], d),
                            k,
                        )
                    )
        boot_rows.append(
            {
                "bootstrap": b,
                "mean_dS": res["mean_dS"],
                "mean_dT": res["mean_dT"],
                "frac_charts_dS_pos": res["frac_charts_dS_pos"],
                "top1_overlap": float(np.nanmean(overlaps[0::3])) if overlaps else float("nan"),
                "top2_overlap": float(np.nanmean(overlaps[1::3])) if overlaps else float("nan"),
                "top4_overlap": float(np.nanmean(overlaps[2::3])) if overlaps else float("nan"),
            }
        )
        if (b + 1) % 5 == 0:
            print(f"[sphere-normal] bootstrap {b+1}/{cfg.n_bootstrap}", flush=True)

    rng0 = np.random.default_rng(cfg.seed + 7)
    tr = np.asarray(data["train_local"]).copy()
    rng0.shuffle(tr)
    n = len(tr)
    idx_geom, idx_fit, idx_val = tr[: int(0.4 * n)], tr[int(0.4 * n) : int(0.8 * n)], tr[int(0.8 * n) :]

    # Precompute per-chart geometry once for nulls.
    chart_cache = []
    for c in range(W.shape[1]):
        w = np.asarray(W[:, c].todense()).ravel()

        def memb(idx, _w=w):
            return idx[_w[idx] > 1e-6]

        g, f, v = memb(idx_geom), memb(idx_fit), memb(idx_val)
        te_c = memb(te)
        if len(f) < 30 or len(te_c) < 10:
            continue
        chart0, _, _, U = fit_nested_chart(X, np.zeros((len(X), d)), w, g, f, v)
        x0, J = chart0.x0, chart0.J
        Phi = quadratic_features(U[f])
        L_f = x0[None, :] + U[f] @ J.T
        scale = np.linalg.norm(L_f, axis=1, keepdims=True)
        target_un = X[f] * np.maximum(scale, 1e-8)
        tang_res = (target_un - L_f) @ J
        chart_cache.append(
            {
                "c": c,
                "w": w,
                "f": f,
                "te_c": te_c,
                "chart0": chart0,
                "U": U,
                "Phi": Phi,
                "tang_res": tang_res,
                "target_un": target_un,
                "L_f": L_f,
            }
        )

    null_shuffle, null_random = [], []
    for nrep in range(cfg.n_null):
        _budget(t0, cfg, f"null_rep{nrep}")
        rng = np.random.default_rng(cfg.seed + 9000 + nrep)
        dS_s_charts, dS_r_charts = [], []
        for cc in chart_cache:
            x0, J = cc["chart0"].x0, cc["chart0"].J
            f, te_c, w, U = cc["f"], cc["te_c"], cc["w"], cc["U"]
            Phi_s = cc["Phi"][rng.permutation(len(cc["Phi"]))]
            A_s, lamA, _ = _ridge_multi(Phi_s, cc["tang_res"], w[f], [1e-2, 1e-1, 1.0])
            L_tr = x0[None, :] + (U[f] + Phi_s @ A_s.T) @ J.T
            resid_n = normal_projector_apply(
                (X[f] * np.maximum(np.linalg.norm(L_tr, axis=1, keepdims=True), 1e-8) - L_tr).T,
                x0,
                J,
            ).T
            BS_s, lamB, _ = _ridge_multi(Phi_s, resid_n, w[f], [1e-2, 1e-1, 1.0])
            BS_s = normal_projector_apply(BS_s, x0, J)
            ch_s = NestedChart(x0, J, A_s, BS_s, lamA, lamB, cc["chart0"].coord_scale)
            dS_s_charts.append(chart_errors(ch_s, ch_s, X, U, w, te_c)["dS"])
            R = rng.normal(size=(d, d))
            Phi_r = quadratic_features((U @ R)[f])
            BS_r, lamB, _ = _ridge_multi(
                Phi_r,
                normal_projector_apply((cc["target_un"] - cc["L_f"]).T, x0, J).T,
                w[f],
                [1e-2, 1e-1, 1.0],
            )
            BS_r = normal_projector_apply(BS_r, x0, J)
            Phi_te = quadratic_features((U @ R)[te_c])
            pred = normalize_rows(x0[None, :] + U[te_c] @ J.T + Phi_te @ BS_r.T)
            E_TRS = weighted_mse(pred.astype(np.float32), X[te_c].astype(np.float32), w[te_c])
            E_TR = weighted_mse(
                cc["chart0"].decode_TR(U[te_c]).astype(np.float32),
                X[te_c].astype(np.float32),
                w[te_c],
            )
            dS_r_charts.append(float(E_TR - E_TRS))
        null_shuffle.append(float(np.nanmean(dS_s_charts)) if dS_s_charts else float("nan"))
        null_random.append(float(np.nanmean(dS_r_charts)) if dS_r_charts else float("nan"))
        if (nrep + 1) % 5 == 0:
            print(f"[sphere-normal] null {nrep+1}/{cfg.n_null}", flush=True)

    real = ref["mean_dS"]
    p_s, B_s = mc_pvalue(real, np.asarray(null_shuffle))
    p_r, B_r = mc_pvalue(real, np.asarray(null_random))
    boot_df = pd.DataFrame(boot_rows)
    null_df = pd.DataFrame({"shuffle_dS": null_shuffle, "random_dS": null_random})
    boot_df.to_parquet(out / "bootstrap.parquet", index=False)
    null_df.to_parquet(out / "nulls.parquet", index=False)
    summary = {
        "real_mean_dS": real,
        "bootstrap_mean": float(boot_df.mean_dS.mean()),
        "bootstrap_ci95": [
            float(boot_df.mean_dS.quantile(0.025)),
            float(boot_df.mean_dS.quantile(0.975)),
        ],
        "frac_boot_dS_pos": float((boot_df.mean_dS > 0).mean()),
        "p_shuffle": p_s,
        "p_random_features": p_r,
        "B_null": int(B_s),
        "B_bootstrap": int(len(boot_df)),
        "top4_BS_overlap_mean": float(boot_df.top4_overlap.mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_report(root: Path, cfg: SphereNormalConfig) -> dict:
    out = cfg.resolved_out(root)
    grid = json.loads((out / "grid" / "grid_summary.json").read_text())
    synth = json.loads((out / "synthetic" / "synthetic.json").read_text())
    boot = json.loads((out / "bootstrap_nulls" / "summary.json").read_text())
    charts = pd.read_parquet(out / "grid" / "nested_model_errors.parquet")
    ranks = pd.read_parquet(out / "grid" / "predictive_rank.parquet")
    primary = next(s for s in grid["configs"] if s["config_id"] == cfg.primary)
    pcharts = charts[charts.config_id == cfg.primary]

    # verdict logic
    dS = primary["mean_dS"]
    frac = primary["frac_charts_dS_pos"]
    boot_pos = boot["frac_boot_dS_pos"]
    p_s, p_r = boot["p_shuffle"], boot["p_random_features"]
    pred90 = pcharts["pred_rank_90pct_dS"].astype(str)
    # low-rank if many charts have pred_rank_90pct in {1,2,4}
    low_rank_frac = float(np.mean(pred90.isin(["1", "2", "4"]))) if len(pred90) else 0.0
    mean_stable = float(pcharts.BS_stable_rank.mean()) if len(pcharts) else float("nan")

    if not synth.get("pass", False):
        # still proceed but note
        pass

    if not np.isfinite(dS) or (frac < 0.6 and boot_pos < 0.6):
        verdict = "sphere_normal_test_underpowered"
    elif dS <= 0 or p_s > 0.05 or boot_pos < 0.8:
        verdict = "second_order_but_not_sphere_normal"
    elif low_rank_frac >= 0.5 and boot["top4_BS_overlap_mean"] > 0.3:
        verdict = "sphere_normal_curvature_stable_low_rank"
    elif dS > 0 and p_s <= 0.05 and boot_pos >= 0.8 and low_rank_frac < 0.5:
        # distributed if predictive rank needs full/high
        if float(np.mean(pred90.isin(["full", "16", "8"]))) > 0.5:
            verdict = "sphere_normal_curvature_distributed"
        else:
            verdict = "sphere_normal_curvature_predictive"
    else:
        verdict = "sphere_normal_curvature_predictive"

    statements = {
        "second_order_but_not_sphere_normal": (
            "Held-out second-order gains are explained by radial normalization and/or "
            "tangential coordinate warping; sphere-normal bending is not predictive."
        ),
        "sphere_normal_curvature_predictive": (
            "After removing tangential warping, a sphere-normal quadratic term still reduces "
            "held-out error (Δ_S>0) across charts/bootstraps and beats pairing nulls."
        ),
        "sphere_normal_curvature_distributed": (
            "Sphere-normal bending is predictive, but predictive mass is spread across many "
            "quadratic normal modes rather than a few dominant ones."
        ),
        "sphere_normal_curvature_stable_low_rank": (
            "Sphere-normal bending is predictive and concentrated in a small number of "
            "stable normal modes under bootstrap principal-angle and truncation tests."
        ),
        "sphere_normal_test_underpowered": (
            "The sphere-normal test is underpowered or unstable at this smoke scale; "
            "do not claim geometric curvature."
        ),
    }

    # tables
    pd.DataFrame(grid["configs"]).to_csv(out / "delta_summary_by_config.csv", index=False)
    pcharts.to_csv(out / "primary_chart_table.csv", index=False)
    ranks[ranks.config_id == cfg.primary].to_csv(out / "primary_predictive_rank.csv", index=False)

    # sample complexity table
    sens = charts.groupby("config_id").agg(
        mean_dS=("dS", "mean"),
        mean_dT=("dT", "mean"),
        mean_n_eff_per_feat=("n_eff_per_quad_feature", "mean"),
        mean_BS_stable_rank=("BS_stable_rank", "mean"),
        frac_dS_pos=("dS", lambda s: float(np.mean(np.asarray(s) > 0))),
    ).reset_index()
    sens.to_csv(out / "sample_complexity_sensitivity.csv", index=False)

    report = f"""# Sphere-normal quadratic atlas

## Verdict

`{verdict}`

{statements[verdict]}

## Synthetic controls

pass={synth.get("pass")} checks={json.dumps(synth.get("checks"), indent=2)}

## Primary configuration `{cfg.primary}`

- mean Δ_S = {dS:.6f}
- mean Δ_T = {primary["mean_dT"]:.6f}
- mean Δ_{{S|R}} = {primary["mean_dSR"]:.6f}
- fraction charts with Δ_S>0 = {frac:.3f}
- bootstrap mean Δ_S = {boot["bootstrap_mean"]:.6f}
- bootstrap 95% CI = {boot["bootstrap_ci95"]}
- fraction bootstrap Δ_S>0 = {boot_pos:.3f}
- corrected MC p(shuffle) = {p_s:.4f} (B={boot["B_null"]})
- corrected MC p(random features) = {p_r:.4f}
- mean BS stable rank = {mean_stable:.3f}
- fraction charts with pred-rank-90% in {{1,2,4}} = {low_rank_frac:.3f}
- mean top-4 BS subspace overlap (bootstrap) = {boot["top4_BS_overlap_mean"]:.3f}

## Nested model note

Models use sphere-tangent orthonormal J and P_{{N,S}}=I-Proj(span(x0,J)).
No explicit radial quadratic term (Normalize induces radial geometry).

## Sensitivities

{sens.to_string(index=False)}

## Exact next command (not run)

```bash
cd ~/platonic-universe && source .venv/bin/activate && \\
PYTHONPATH=experiments python -m geometry.run_sphere_normal_quadratic \\
  --stage all --n-bootstrap 100 --n-null 100 --seed 0
```

Do **not** launch retrieval / Fisher / JS / physics-label analyses from this verdict.
Existing correlations remain exploratory only.
"""
    (out / "REPORT.md").write_text(report)
    # Spectra table (primary)
    spec_rows = []
    for _, row in pcharts.iterrows():
        svals = row.get("BS_svals", None)
        if isinstance(svals, str):
            svals = json.loads(svals)
        if svals is None or (isinstance(svals, float) and not np.isfinite(svals)):
            continue
        for i, s in enumerate(list(svals)):
            spec_rows.append({"chart": int(row["chart"]), "mode": i, "singular_value": float(s)})
    if spec_rows:
        pd.DataFrame(spec_rows).to_csv(out / "primary_BS_spectra.csv", index=False)
    analysis = {
        "verdict": verdict,
        "statement": statements[verdict],
        "primary": primary,
        "bootstrap_nulls": boot,
        "synthetic_pass": synth.get("pass"),
        "next_command": (
            "cd ~/platonic-universe && source .venv/bin/activate && "
            "PYTHONPATH=experiments python -m geometry.run_sphere_normal_quadratic "
            "--stage all --n-bootstrap 100 --n-null 100 --seed 0"
        ),
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    return analysis


STAGES = ["synthetic", "grid", "bootstrap_nulls", "report"]


def run_sphere_normal(cfg: SphereNormalConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    results: dict[str, Any] = {}
    order = STAGES if cfg.stage == "all" else [cfg.stage]
    for s in order:
        print(f"[sphere-normal] stage={s} rss={_rss():.1f}", flush=True)
        _budget(t0, cfg, s)
        if s == "synthetic":
            results[s] = stage_synthetic(root, cfg)
        elif s == "grid":
            results[s] = stage_grid(root, cfg, t0)
        elif s == "bootstrap_nulls":
            results[s] = stage_bootstrap_nulls(root, cfg, t0)
        elif s == "report":
            results[s] = stage_report(root, cfg)
        else:
            raise ValueError(s)
    results["total_seconds"] = time.time() - t0
    results["peak_rss_mb"] = _rss()
    (out / "run_summary.json").write_text(json.dumps(results, indent=2, default=str))
    return results
