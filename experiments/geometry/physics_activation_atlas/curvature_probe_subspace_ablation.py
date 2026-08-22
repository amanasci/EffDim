"""Confirmatory ablation: does the frozen curvature subspace add specific probe information?"""

from __future__ import annotations

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
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .confirmatory_object_curvature import (
    _fit_neighborhood,
    select_anchors,
    unpack_BS_symmetric,
)
from .curvature_probe_alignment import B0_flat_for_svd, traceless_B0
from .curvature_probe_screen import (
    EXPECTED_HASH,
    LOCAL_DIM,
    ScreenConfig,
    load_frozen_curvature,
    load_labels_for_selection,
    spearman_dict,
)
from .data import load_prepare
from .paths import platonic_root, resolve_path
from .quadratic import n_quad_features

EPS = 1e-12
PROBE_ALPHA = 100.0
SCALES = (1024, 2048)


def _rss() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _done(p: Path, force: bool) -> bool:
    return p.exists() and not force


@dataclass
class AblationConfig:
    output_dir: str = "outputs/geometry/physics_curvature_probe_subspace_ablation"
    curvature_path: str = (
        "outputs/geometry/physics_quadratic_atlas_sphere_normal/"
        "object_curvature_features_aggregated.parquet"
    )
    alignment_dir: str = "outputs/geometry/physics_curvature_probe_alignment"
    structure_dir: str = "outputs/geometry/physics_quadratic_atlas_structure"
    prepare_dir: str = "outputs/geometry/physics_activation_atlas_geometry_ablation/prepare"
    labels_path: str = "data_hf/physics/vit_base_test_labels.npz"
    expected_hash: str = EXPECTED_HASH
    scales: list[int] = field(default_factory=lambda: list(SCALES))
    primary_k: int = 2048
    probe_alpha: float = PROBE_ALPHA
    n_folds: int = 5
    n_random_controls: int = 50
    n_random_controls_secondary: int = 20
    ranks: list[Any] = field(default_factory=lambda: ["full", 16, 8])
    n_bootstrap: int = 1000
    n_disjoint_anchors: int = 64
    seed: int = 0
    force: bool = False
    max_seconds: float = 7200.0

    def resolved_out(self, root: Path) -> Path:
        return resolve_path(root, self.output_dir)


# -------------------- geometry helpers --------------------


def orthonormalize_mutually(T: np.ndarray, x0: np.ndarray, UB: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure T, x0, U_B mutually orthogonal; return orthonormal (T, x0_unit, U_B)."""
    x0u = x0 / max(np.linalg.norm(x0), EPS)
    # T already sphere-tangent; re-orth against x0
    T2 = T - np.outer(x0u, x0u @ T)
    Qt, _ = np.linalg.qr(T2, mode="reduced")
    # project UB into complement of span(x0,T)
    Qbasis, _ = np.linalg.qr(np.column_stack([x0u, Qt]), mode="reduced")
    UB2 = UB - Qbasis @ (Qbasis.T @ UB)
    Qu, Ru = np.linalg.qr(UB2, mode="reduced")
    diag = np.abs(np.diag(Ru)) if Ru.ndim == 2 else np.array([abs(Ru)])
    keep = diag > 1e-8 * (diag.max() if diag.size else 1.0)
    if np.any(keep):
        Qu = Qu[:, keep]
    return Qt, x0u, Qu


def phi_weighted(z: np.ndarray) -> np.ndarray:
    """Symmetric quadratic monomials with √2 off-diagonal weights."""
    n, d = z.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            if a == b:
                cols.append(z[:, a] * z[:, b])
            else:
                cols.append(np.sqrt(2.0) * z[:, a] * z[:, b])
    return np.stack(cols, axis=1)


def ambient_quadratic_form(B0: np.ndarray, z: np.ndarray) -> np.ndarray:
    """v_i = B°[z_i, z_i] ∈ R^D for each sample."""
    n, d = z.shape
    D = B0.shape[0]
    V = np.zeros((n, D), dtype=np.float64)
    for a in range(d):
        V += np.outer(z[:, a] * z[:, a], B0[:, a, a])
        for b in range(a + 1, d):
            V += np.outer(2.0 * z[:, a] * z[:, b], B0[:, a, b])
    return V


def truncate_UB(UB: np.ndarray, s: np.ndarray, rank) -> np.ndarray:
    if rank == "full" or rank is None:
        return UB
    r = min(int(rank), UB.shape[1], len(s))
    return UB[:, :r]


def haar_normal_basis(D: int, x0u: np.ndarray, T: np.ndarray, r: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(np.column_stack([x0u, T]), mode="reduced")
    G = rng.normal(size=(D, r + 4))
    G = G - Q @ (Q.T @ G)
    U, _ = np.linalg.qr(G, mode="reduced")
    return U[:, :r]


def variance_matched_basis(
    dx: np.ndarray, x0u: np.ndarray, T: np.ndarray, r: int, target_var: float, rng: np.random.Generator, n_try: int = 40
) -> tuple[np.ndarray, float]:
    best_U, best_gap = None, float("inf")
    for _ in range(n_try):
        U = haar_normal_basis(dx.shape[1], x0u, T, r, rng)
        v = float(np.var(dx @ U))
        gap = abs(v - target_var)
        if gap < best_gap:
            best_gap, best_U = gap, U
            if gap / max(target_var, EPS) < 0.2:
                break
    assert best_U is not None
    return best_U, float(np.var(dx @ best_U) / max(target_var, EPS))


def normal_pca_basis(dx: np.ndarray, x0u: np.ndarray, T: np.ndarray, r: int) -> np.ndarray:
    Q, _ = np.linalg.qr(np.column_stack([x0u, T]), mode="reduced")
    dn = dx - (dx @ Q) @ Q.T
    n, D = dn.shape
    r = min(r, n, D)
    try:
        if n >= D:
            w, V = np.linalg.eigh(dn.T @ dn)
            U = V[:, ::-1][:, :r]
        else:
            w, V = np.linalg.eigh(dn @ dn.T)
            V = V[:, ::-1][:, :r]
            U = dn.T @ V
            U, _ = np.linalg.qr(U, mode="reduced")
            U = U[:, : min(r, U.shape[1])]
        # re-orthogonalize into normal complement
        U = U - Q @ (Q.T @ U)
        U, _ = np.linalg.qr(U, mode="reduced")
        return U[:, :r]
    except np.linalg.LinAlgError:
        return haar_normal_basis(D, x0u, T, r, np.random.default_rng(0))


def reproject_into_normal(U_other: np.ndarray, x0u: np.ndarray, T: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(np.column_stack([x0u, T]), mode="reduced")
    G = U_other - Q @ (Q.T @ U_other)
    U, R = np.linalg.qr(G, mode="reduced")
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.array([abs(float(R))])
    keep = diag > 1e-8 * (diag.max() if diag.size else 1.0)
    return U[:, keep] if np.any(keep) else U


# -------------------- ridge on features --------------------


def ridge_r2_features(Z_tr, y_tr, Z_te, y_te, *, alpha: float) -> float:
    if Z_tr is None or len(Z_tr) < 8 or len(Z_te) < 4:
        return float("nan")
    if Z_tr.ndim == 1:
        Z_tr = Z_tr.reshape(-1, 1)
        Z_te = Z_te.reshape(-1, 1)
    m_tr = np.isfinite(y_tr) & np.all(np.isfinite(Z_tr), axis=1)
    m_te = np.isfinite(y_te) & np.all(np.isfinite(Z_te), axis=1)
    if m_tr.sum() < 8 or m_te.sum() < 4:
        return float("nan")
    if Z_tr.shape[1] == 0:
        # intercept-only
        pred = np.full(m_te.sum(), float(np.mean(y_tr[m_tr])))
        return float(r2_score(y_te[m_te], pred))
    xs = StandardScaler().fit(Z_tr[m_tr])
    ys = StandardScaler().fit(y_tr[m_tr].reshape(-1, 1))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xs.transform(Z_tr[m_tr]), ys.transform(y_tr[m_tr].reshape(-1, 1)).ravel())
    pred = ys.inverse_transform(model.predict(xs.transform(Z_te[m_te])).reshape(-1, 1)).ravel()
    return float(r2_score(y_te[m_te], pred))


def ridge_direction_features(Z_tr, y_tr, *, alpha: float) -> np.ndarray:
    m = np.isfinite(y_tr) & np.all(np.isfinite(Z_tr), axis=1)
    if m.sum() < 8 or Z_tr.shape[1] == 0:
        return np.zeros(Z_tr.shape[1])
    xs = StandardScaler().fit(Z_tr[m])
    ys = StandardScaler().fit(y_tr[m].reshape(-1, 1))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xs.transform(Z_tr[m]), ys.transform(y_tr[m].reshape(-1, 1)).ravel())
    return (model.coef_ / np.maximum(xs.scale_, EPS)).astype(np.float64)


# -------------------- synthetics --------------------


def run_synthetic_ablation(seed: int = 0) -> dict:
    """Three qualitative controls."""
    rng = np.random.default_rng(seed)
    rows = []

    def eval_models(Z_T, Z_B, y, name):
        n = len(y)
        idx = np.arange(n)
        rng.shuffle(idx)
        te, tr = idx[: n // 3], idx[n // 3 :]
        r_T = ridge_r2_features(Z_T[tr], y[tr], Z_T[te], y[te], alpha=1.0)
        Z_TB = np.column_stack([Z_T, Z_B])
        r_TB = ridge_r2_features(Z_TB[tr], y[tr], Z_TB[te], y[te], alpha=1.0)
        return {"synth": name, "R2_MT": r_T, "R2_MTBobs": r_TB, "delta_B": r_TB - r_T}

    # 1) affine + linear label
    n, d, D = 600, 4, 32
    T, _ = np.linalg.qr(rng.normal(size=(D, d)))
    x0 = rng.normal(size=D)
    x0 = x0 - T @ (T.T @ x0)
    x0 /= np.linalg.norm(x0)
    U = rng.normal(size=(n, d))
    X = x0[None, :] + U @ T.T
    y = U[:, 0]
    zT = (X - x0) @ T
    # fake normal coords noise
    nvec = rng.normal(size=D)
    nvec = nvec - T @ (T.T @ nvec) - x0 * np.dot(x0, nvec)
    nvec /= np.linalg.norm(nvec)
    zB = ((X - x0) @ nvec).reshape(-1, 1)
    rows.append(eval_models(zT, zB, y, "affine_linear_label"))

    # 2) parabola y=u^2 — need curvature-normal feature
    u = rng.normal(size=n)
    # ambient: embed (u, u^2) in 2d plane then pad
    zT = np.column_stack([u, np.zeros(n)])
    zB = (u**2).reshape(-1, 1)
    y = u**2
    rows.append(eval_models(zT[:, :1], zB, y, "parabola_y_u2"))

    # 3) curved manifold, tangent-linear label
    u = rng.normal(size=(n, 2)) * 0.5
    # points on parabola in ambient but label = u0
    zT = u
    zB = (u[:, 0] ** 2 + u[:, 1] ** 2).reshape(-1, 1)
    y = u[:, 0]
    rows.append(eval_models(zT, zB, y, "curved_tangent_linear_label"))

    checks = {
        "affine_tangent_ok_curvature_useless": rows[0]["R2_MT"] > 0.5 and rows[0]["delta_B"] < 0.05,
        "parabola_needs_curvature": rows[1]["R2_MT"] < 0.3 and rows[1]["R2_MTBobs"] > 0.5,
        "curved_tangent_linear_ok": rows[2]["R2_MT"] > 0.5 and rows[2]["delta_B"] < 0.1,
    }
    return {"rows": rows, "checks": checks, "pass": all(checks.values())}


# -------------------- core per-anchor ablation --------------------


def build_features(
    Xn: np.ndarray,
    x0u: np.ndarray,
    T: np.ndarray,
    UB: np.ndarray,
    B0: np.ndarray,
) -> dict[str, np.ndarray]:
    dx = Xn - x0u[None, :]
    z_T = dx @ T
    z_B_obs = dx @ UB
    V = ambient_quadratic_form(B0, z_T)
    z_B_pred = V @ UB
    phi = phi_weighted(z_T)
    return {
        "z_T": z_T,
        "z_B_obs": z_B_obs,
        "z_B_pred": z_B_pred,
        "phi": phi,
        "dx": dx,
        "ambient": Xn,
    }


def cv_model_r2(Z: np.ndarray | None, y: np.ndarray, folds: list[np.ndarray], alpha: float) -> tuple[float, list[np.ndarray]]:
    """Mean held-out R²; also return per-fold weight vectors when Z not None."""
    scores, weights = [], []
    n_folds = len(folds)
    for fi, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        if Z is None:
            # intercept
            pred = np.full(len(te), float(np.nanmean(y[tr])))
            m = np.isfinite(y[te])
            scores.append(float(r2_score(y[te][m], pred[m])) if m.sum() >= 4 else float("nan"))
            weights.append(np.zeros(0))
        else:
            scores.append(ridge_r2_features(Z[tr], y[tr], Z[te], y[te], alpha=alpha))
            weights.append(ridge_direction_features(Z[tr], y[tr], alpha=alpha))
    return float(np.nanmean(scores)), weights


def direction_stability(weights: list[np.ndarray]) -> float:
    vecs = [w for w in weights if w.size and np.linalg.norm(w) > EPS]
    if len(vecs) < 2:
        return float("nan")
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            a = vecs[i] / np.linalg.norm(vecs[i])
            b = vecs[j] / np.linalg.norm(vecs[j])
            # pad to same length if needed
            m = min(len(a), len(b))
            sims.append(abs(float(np.dot(a[:m], b[:m]))))
    return float(np.mean(sims)) if sims else float("nan")


def evaluate_anchor(
    X: np.ndarray,
    y: np.ndarray,
    neigh: np.ndarray,
    chart_pack: dict,
    *,
    alpha: float,
    n_folds: int,
    ranks: list,
    n_random: int,
    seed: int,
    UB_pool: list[np.ndarray] | None,
) -> list[dict]:
    """Return rows for each rank setting."""
    x0u, T, B0, UB_full, svals = (
        chart_pack["x0u"],
        chart_pack["T"],
        chart_pack["B0"],
        chart_pack["UB"],
        chart_pack["s"],
    )
    Xn = X[neigh]
    yn = y[neigh]
    n = len(neigh)
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    folds = np.array_split(order, n_folds)
    dx = Xn - x0u[None, :]
    target_var = float(np.var(dx @ UB_full)) if UB_full.size else 1.0

    rows = []
    for rank in ranks:
        UB = truncate_UB(UB_full, svals, rank)
        r = UB.shape[1]
        if r < 1:
            continue
        feats = build_features(Xn, x0u, T, UB, B0)
        models = {
            "M0": None,
            "MT": feats["z_T"],
            "MTQ": np.column_stack([feats["z_T"], feats["phi"]]),
            "MTBpred": np.column_stack([feats["z_T"], feats["z_B_pred"]]),
            "MTBobs": np.column_stack([feats["z_T"], feats["z_B_obs"]]),
            "MBobs": feats["z_B_obs"],
            "Mfull": feats["ambient"],
        }
        r2s, stabs = {}, {}
        for name, Z in models.items():
            r2, wts = cv_model_r2(Z, yn, folds, alpha)
            r2s[name] = r2
            stabs[name] = direction_stability(wts)

        # equivalence MTBpred vs MTQ
        # correlate predictions on held-out via stacked OOF
        oof_pred = {name: np.full(n, np.nan) for name in ("MTQ", "MTBpred")}
        for fi, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
            for name in oof_pred:
                Z = models[name]
                m_tr = np.isfinite(yn[tr]) & np.all(np.isfinite(Z[tr]), axis=1)
                m_te = np.isfinite(yn[te]) & np.all(np.isfinite(Z[te]), axis=1)
                if m_tr.sum() < 8 or m_te.sum() < 1:
                    continue
                xs = StandardScaler().fit(Z[tr][m_tr])
                ys = StandardScaler().fit(yn[tr][m_tr].reshape(-1, 1))
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(xs.transform(Z[tr][m_tr]), ys.transform(yn[tr][m_tr].reshape(-1, 1)).ravel())
                pred = ys.inverse_transform(model.predict(xs.transform(Z[te][m_te])).reshape(-1, 1)).ravel()
                oof_pred[name][te[m_te]] = pred
        m = np.isfinite(oof_pred["MTQ"]) & np.isfinite(oof_pred["MTBpred"])
        if m.sum() >= 8:
            corr_pred = float(np.corrcoef(oof_pred["MTQ"][m], oof_pred["MTBpred"][m])[0, 1])
            eq_delta = abs(r2s["MTQ"] - r2s["MTBpred"])
        else:
            corr_pred, eq_delta = float("nan"), float("nan")

        dB = r2s["MTBobs"] - r2s["MT"]
        dBpred = r2s["MTBpred"] - r2s["MT"]
        dQ = r2s["MTQ"] - r2s["MT"]
        dfull = r2s["Mfull"] - r2s["MT"]

        # controls
        d_rand = []
        d_varmatch = []
        for ci in range(n_random):
            U_r = haar_normal_basis(X.shape[1], x0u, T, r, rng)
            Z = np.column_stack([feats["z_T"], dx @ U_r])
            rr, _ = cv_model_r2(Z, yn, folds, alpha)
            d_rand.append(rr - r2s["MT"])
            U_v, _ = variance_matched_basis(dx, x0u, T, r, target_var, rng, n_try=12)
            Zv = np.column_stack([feats["z_T"], dx @ U_v])
            rv, _ = cv_model_r2(Zv, yn, folds, alpha)
            d_varmatch.append(rv - r2s["MT"])

        U_pca = normal_pca_basis(dx, x0u, T, r)
        r_pca, _ = cv_model_r2(np.column_stack([feats["z_T"], dx @ U_pca]), yn, folds, alpha)
        d_pca = r_pca - r2s["MT"]

        d_shuf = float("nan")
        if UB_pool and len(UB_pool) > 1:
            other = UB_pool[rng.integers(0, len(UB_pool))]
            # truncate/pad
            if other.shape[1] >= r:
                Uo = other[:, :r]
            else:
                pad = haar_normal_basis(X.shape[1], x0u, T, r - other.shape[1], rng)
                Uo = np.column_stack([other, pad])[:, :r]
            Uo = reproject_into_normal(Uo, x0u, T)
            if Uo.shape[1] < r:
                extra = haar_normal_basis(X.shape[1], x0u, T, r - Uo.shape[1], rng)
                Uo = np.column_stack([Uo, extra])[:, :r]
            rs, _ = cv_model_r2(np.column_stack([feats["z_T"], dx @ Uo[:, :r]]), yn, folds, alpha)
            d_shuf = rs - r2s["MT"]

        rows.append(
            {
                "rank": str(rank),
                "rank_dim": int(r),
                "R2_M0": r2s["M0"],
                "R2_MT": r2s["MT"],
                "R2_MTQ": r2s["MTQ"],
                "R2_MTBpred": r2s["MTBpred"],
                "R2_MTBobs": r2s["MTBobs"],
                "R2_MBobs": r2s["MBobs"],
                "R2_Mfull": r2s["Mfull"],
                "delta_B": dB,
                "delta_Bpred": dBpred,
                "delta_Q": dQ,
                "delta_full": dfull,
                "delta_random_median": float(np.median(d_rand)) if d_rand else float("nan"),
                "delta_random_mean": float(np.mean(d_rand)) if d_rand else float("nan"),
                "delta_varmatch_median": float(np.median(d_varmatch)) if d_varmatch else float("nan"),
                "delta_normal_PCA": float(d_pca),
                "delta_shuffled_B": float(d_shuf),
                "spec_vs_random": float(dB - np.median(d_rand)) if d_rand else float("nan"),
                "spec_vs_pca": float(dB - d_pca),
                "spec_vs_shuffled": float(dB - d_shuf) if np.isfinite(d_shuf) else float("nan"),
                "frac_random_beaten": float(np.mean(dB > np.asarray(d_rand))) if d_rand else float("nan"),
                "MTQ_MTBpred_pred_corr": corr_pred,
                "MTQ_MTBpred_abs_deltaR2": eq_delta,
                "stab_MT": stabs["MT"],
                "stab_MTBobs": stabs["MTBobs"],
                "n_random": int(n_random),
                "target_proj_var": target_var,
            }
        )
    return rows


# -------------------- aggregation / stats --------------------


def summarize_deltas(vals: np.ndarray, n_boot: int, seed: int) -> dict:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return {k: float("nan") for k in ("mean", "median", "trimmed_mean", "frac_pos", "ci95_lo", "ci95_hi")}
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(v, size=len(v), replace=True).mean()) for _ in range(n_boot)]
    trim = v[(v >= np.quantile(v, 0.05)) & (v <= np.quantile(v, 0.95))]
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "trimmed_mean": float(np.mean(trim)) if len(trim) else float("nan"),
        "frac_pos": float(np.mean(v > 0)),
        "ci95_lo": float(np.quantile(boots, 0.025)),
        "ci95_hi": float(np.quantile(boots, 0.975)),
        "n": int(len(v)),
    }


def mc_pvalue_greater(real: float, nulls: np.ndarray) -> tuple[float, int]:
    nulls = np.asarray(nulls, dtype=np.float64)
    nulls = nulls[np.isfinite(nulls)]
    B = len(nulls)
    if B == 0 or not np.isfinite(real):
        return float("nan"), 0
    return float((1 + np.sum(nulls >= real)) / (B + 1)), B


def choose_label(primary: dict, disjoint: dict, synth_pass: bool) -> tuple[str, str]:
    dB = primary.get("delta_B_mean", float("nan"))
    vs_rand = primary.get("spec_vs_random_mean", float("nan"))
    vs_pca = primary.get("spec_vs_pca_mean", float("nan"))
    vs_shuf = primary.get("spec_vs_shuffled_mean", float("nan"))
    p_rand = primary.get("p_spec_vs_random", 1.0)
    dBpred = primary.get("delta_Bpred_mean", float("nan"))
    dQ = primary.get("delta_Q_mean", float("nan"))
    disjoint_ok = bool(disjoint.get("survives", False))
    frac_pos = primary.get("delta_B_frac_pos", 0.0)

    if not np.isfinite(dB) or dB <= 0 or frac_pos < 0.55:
        if primary.get("R2_MT_mean", 0) > 0.4 and dB <= 0.01:
            return "tangent_space_sufficient", "Tangent features already capture the probe; curvature coords add nothing."
        return "inconclusive", "ΔB not consistently positive across anchors."

    specific = (
        vs_rand > 0
        and p_rand <= 0.05
        and vs_pca > 0
        and (not np.isfinite(vs_shuf) or vs_shuf > 0)
    )
    pred_ok = (dBpred > 0.5 * dB) or (dQ > 0.5 * dB)

    if specific and pred_ok and disjoint_ok:
        return (
            "curvature_subspace_adds_specific_probe_information",
            "MTBobs beats MT and matched nulls; predicted/quadratic path recovers substantial gain; disjoint-neighbour sensitivity survives.",
        )
    if specific and pred_ok and not disjoint_ok:
        return (
            "shared_estimation_noise_explains_alignment",
            "Apparent curvature specificity weakens under disjoint-neighbour cross-fitting.",
        )
    if (not specific) and pred_ok and dQ >= dBpred - 1e-6:
        return (
            "intrinsic_quadratic_signal_without_curvature_specificity",
            "Quadratic/tangent-polynomial features explain the gain without beating matched normal controls.",
        )
    if (not specific) and dB > 0:
        return (
            "generic_normal_variance_not_curvature_specific",
            "Normal-space coordinates help, but not beyond matched random/PCA normal subspaces.",
        )
    if not disjoint_ok and dB > 0:
        return (
            "shared_estimation_noise_explains_alignment",
            "Gain does not survive disjoint-neighbour sensitivity.",
        )
    return "inconclusive", "Mixed evidence; no single gate set fully satisfied."


# -------------------- pipeline --------------------


def rematerialize_chart(X, neigh, ai, k, seed) -> dict | None:
    seed_fit = seed + 17 * ai + k
    chart, _, info, Uloc, glob, reason = _fit_neighborhood(X, neigh, LOCAL_DIM, seed=seed_fit)
    if chart is None:
        return None
    B0, _ = traceless_B0(chart.BS_flat, chart.J.shape[1])
    Bflat = B0_flat_for_svd(B0, chart.J.shape[1])
    U, s, _ = np.linalg.svd(Bflat, full_matrices=False)
    keep = s > 1e-8 * (s[0] if len(s) else 1.0)
    UB = U[:, keep] if np.any(keep) else U
    T, x0u, UB = orthonormalize_mutually(chart.J, chart.x0, UB)
    return {
        "x0u": x0u,
        "T": T,
        "B0": B0,
        "UB": UB,
        "s": s[keep] if np.any(keep) else s,
        "glob": glob,
        "info": info,
        "B0_fro": float(np.linalg.norm(B0)),
    }


def run_ablation(cfg: AblationConfig, root: Path | None = None) -> dict[str, Any]:
    root = root or platonic_root()
    out = cfg.resolved_out(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t0 = time.time()
    peak = _rss()

    # synthetics + unit checks
    synth = run_synthetic_ablation(cfg.seed)
    (out / "synthetic_ablation.json").write_text(json.dumps(synth, indent=2))
    pd.DataFrame(synth["rows"]).to_csv(out / "synthetic_ablation.csv", index=False)
    print(f"[ablation] synthetics pass={synth['pass']} checks={synth['checks']}", flush=True)

    scfg = ScreenConfig(curvature_path=cfg.curvature_path, expected_hash=cfg.expected_hash)
    curv = load_frozen_curvature(root, scfg)
    hashes = set(curv.config_hash.astype(str).unique())
    assert hashes == {cfg.expected_hash}, hashes

    align_path = resolve_path(root, cfg.alignment_dir) / "joined_alignment.parquet"
    align = pd.read_parquet(align_path) if align_path.exists() else None

    data = load_prepare(resolve_path(root, cfg.prepare_dir))
    X = data["X"].astype(np.float64)
    y = load_labels_for_selection(root, scfg, data["sample_ids"])
    train = np.asarray(data["train_local"])
    anchors = select_anchors(data, 384)
    sid_to_ai = {int(data["sample_ids"][a]): i for i, a in enumerate(anchors)}

    # chart membership for clustered bootstrap
    Wpath = resolve_path(root, cfg.structure_dir) / "grid" / "n6_d8" / "memberships_csr.npz"
    chart_id = {}
    if Wpath.exists():
        W = sparse.load_npz(Wpath)
        for ai, a in enumerate(anchors):
            w = np.asarray(W[a].todense()).ravel()
            chart_id[int(data["sample_ids"][a])] = int(np.argmax(w)) if w.size else -1

    k_max = max(cfg.scales)
    nn = NearestNeighbors(n_neighbors=k_max, metric="euclidean")
    nn.fit(X[train])
    dists, inds = nn.kneighbors(X[anchors])

    marker = out / "anchor_results.parquet"
    if _done(marker, cfg.force):
        res_df = pd.read_parquet(marker)
    else:
        # first pass: rematerialize UB pool for shuffle control at each k
        packs: dict[tuple[int, int], dict] = {}
        UB_pool_by_k: dict[int, list] = {k: [] for k in cfg.scales}
        print("[ablation] rematerializing charts", flush=True)
        for k in cfg.scales:
            for ai, a_local in enumerate(anchors):
                sid = int(data["sample_ids"][a_local])
                neigh = train[inds[ai, :k]]
                neigh = neigh[neigh != a_local]
                pack = rematerialize_chart(X, neigh, ai, k, cfg.seed)
                if pack is None:
                    continue
                packs[(sid, k)] = {**pack, "neigh": neigh, "rho": float(dists[ai, k - 1]), "ai": ai}
                UB_pool_by_k[k].append(pack["UB"])
            print(f"[ablation] k={k} packs={sum(1 for s,kk in packs if kk==k)} rss={_rss():.0f}", flush=True)

        rows = []
        for k in cfg.scales:
            n_rand = cfg.n_random_controls if k == cfg.primary_k else cfg.n_random_controls_secondary
            for ai, a_local in enumerate(anchors):
                if time.time() - t0 > cfg.max_seconds:
                    raise RuntimeError("max_seconds exceeded")
                sid = int(data["sample_ids"][a_local])
                pack = packs.get((sid, k))
                if pack is None:
                    continue
                # verify frozen join
                crow = curv[(curv.sample_id == sid) & (curv.scale_k == k)]
                if len(crow) != 1:
                    continue
                frozen_B0 = float(crow.iloc[0]["B_traceless_fro"])
                rel = abs(pack["B0_fro"] - frozen_B0) / max(frozen_B0, EPS)
                erows = evaluate_anchor(
                    X,
                    y,
                    pack["neigh"],
                    pack,
                    alpha=cfg.probe_alpha,
                    n_folds=cfg.n_folds,
                    ranks=cfg.ranks,
                    n_random=n_rand,
                    seed=cfg.seed + 1009 * ai + k,
                    UB_pool=UB_pool_by_k[k],
                )
                # join prior alignment features
                A_B = C_w = recon = probe_r2 = float("nan")
                if align is not None:
                    ar = align[(align.sample_id == sid) & (align.scale_k == k)]
                    if len(ar):
                        A_B = float(ar.iloc[0].get("A_B", np.nan))
                        C_w = float(ar.iloc[0].get("C_w", np.nan))
                        recon = float(ar.iloc[0].get("reconstruction_error", np.nan))
                        probe_r2 = float(ar.iloc[0].get("probe_r2", np.nan))
                for er in erows:
                    rows.append(
                        {
                            "sample_id": sid,
                            "scale_k": k,
                            "chart_id": chart_id.get(sid, -1),
                            "knn_radius": pack["rho"],
                            "rematerialize_rel_err": rel,
                            "A_B": A_B,
                            "C_w": C_w,
                            "reconstruction_error": recon,
                            "probe_r2_screen": probe_r2,
                            "config_hash": cfg.expected_hash,
                            **er,
                        }
                    )
                if (ai + 1) % 32 == 0:
                    peak = max(peak, _rss())
                    print(f"[ablation] k={k} anchors {ai+1}/{len(anchors)} rss={peak:.0f}", flush=True)
        res_df = pd.DataFrame(rows)
        res_df.to_parquet(marker, index=False)

    peak = max(peak, _rss())

    # disjoint-neighbour sensitivity on subset at primary k
    disjoint_rows = []
    rng = np.random.default_rng(cfg.seed + 5)
    primary_sids = res_df[(res_df.scale_k == cfg.primary_k) & (res_df["rank"] == "full")]["sample_id"].unique()
    take = np.sort(rng.choice(primary_sids, size=min(cfg.n_disjoint_anchors, len(primary_sids)), replace=False))
    print(f"[ablation] disjoint-neighbour sensitivity n={len(take)}", flush=True)
    for sid in take:
        ai = sid_to_ai[int(sid)]
        a_local = int(anchors[ai])
        k = cfg.primary_k
        neigh = train[inds[ai, :k]]
        neigh = neigh[neigh != a_local]
        rng_i = np.random.default_rng(cfg.seed + 333 * ai + k)
        order = neigh.copy()
        rng_i.shuffle(order)
        n = len(order)
        # thirds: geom | probe-train | eval
        n1, n2 = n // 3, 2 * (n // 3)
        geom_idx, ptr_idx, ev_idx = order[:n1], order[n1:n2], order[n2:]
        if min(len(geom_idx), len(ptr_idx), len(ev_idx)) < 20:
            continue
        pack = rematerialize_chart(X, geom_idx, ai, k, cfg.seed)
        if pack is None:
            continue
        # features on probe-train+eval using geom-estimated bases
        use = np.concatenate([ptr_idx, ev_idx])
        feats = build_features(X[use], pack["x0u"], pack["T"], pack["UB"], pack["B0"])
        y_use = y[use]
        # single split: first len(ptr) train
        ntr = len(ptr_idx)
        Z_T = feats["z_T"]
        Z_TB = np.column_stack([feats["z_T"], feats["z_B_obs"]])
        r_T = ridge_r2_features(Z_T[:ntr], y_use[:ntr], Z_T[ntr:], y_use[ntr:], alpha=cfg.probe_alpha)
        r_TB = ridge_r2_features(Z_TB[:ntr], y_use[:ntr], Z_TB[ntr:], y_use[ntr:], alpha=cfg.probe_alpha)
        disjoint_rows.append({"sample_id": int(sid), "R2_MT": r_T, "R2_MTBobs": r_TB, "delta_B": r_TB - r_T})
    disjoint_df = pd.DataFrame(disjoint_rows)
    disjoint_df.to_csv(out / "disjoint_neighbour_sensitivity.csv", index=False)
    disjoint_summary = {
        "n": int(len(disjoint_df)),
        "mean_delta_B": float(disjoint_df.delta_B.mean()) if len(disjoint_df) else float("nan"),
        "frac_pos": float((disjoint_df.delta_B > 0).mean()) if len(disjoint_df) else float("nan"),
        "survives": bool(len(disjoint_df) and disjoint_df.delta_B.mean() > 0 and (disjoint_df.delta_B > 0).mean() >= 0.55),
    }

    # summaries per scale × rank
    summary_rows = []
    for k in cfg.scales:
        for rank in [str(r) for r in cfg.ranks]:
            g = res_df[(res_df.scale_k == k) & (res_df["rank"] == rank)]
            if len(g) == 0:
                continue
            sB = summarize_deltas(g.delta_B.to_numpy(float), cfg.n_bootstrap, cfg.seed + k)
            sR = summarize_deltas(g.spec_vs_random.to_numpy(float), cfg.n_bootstrap, cfg.seed + 3 + k)
            sP = summarize_deltas(g.spec_vs_pca.to_numpy(float), cfg.n_bootstrap, cfg.seed + 5 + k)
            sS = summarize_deltas(g.spec_vs_shuffled.to_numpy(float), cfg.n_bootstrap, cfg.seed + 7 + k)
            # chart-clustered bootstrap for delta_B
            rng = np.random.default_rng(cfg.seed + 11 + k)
            charts = g.chart_id.fillna(-1).to_numpy()
            uniq = np.unique(charts)
            cboots = []
            for _ in range(min(cfg.n_bootstrap, 500)):
                chs = rng.choice(uniq, size=len(uniq), replace=True)
                parts = [g.loc[charts == c, "delta_B"].to_numpy(float) for c in chs]
                samp = np.concatenate(parts) if parts else np.array([])
                samp = samp[np.isfinite(samp)]
                if len(samp):
                    cboots.append(float(np.mean(samp)))
            # permutation null for mean ΔB: sign-flip
            nulls = []
            v = g.delta_B.to_numpy(float)
            v = v[np.isfinite(v)]
            for _ in range(cfg.n_bootstrap):
                signs = rng.choice([-1.0, 1.0], size=len(v))
                nulls.append(float(np.mean(signs * v)))
            p_dB, B = mc_pvalue_greater(sB["mean"], np.asarray(nulls))
            vsp = g.spec_vs_random.to_numpy(float)
            vsp = vsp[np.isfinite(vsp)]
            null_sp = [
                float(np.mean(rng.choice([-1.0, 1.0], size=len(vsp)) * vsp))
                for _ in range(cfg.n_bootstrap)
            ]
            p_spec, _ = mc_pvalue_greater(
                float(np.mean(vsp)) if len(vsp) else float("nan"), np.asarray(null_sp)
            )

            # relations
            rel = {}
            for col in ("A_B", "C_w", "knn_radius", "reconstruction_error"):
                if col in g.columns:
                    rel[f"spearman_deltaB_{col}"] = spearman_dict(
                        g.delta_B.to_numpy(float), g[col].to_numpy(float)
                    )

            summary_rows.append(
                {
                    "scale_k": k,
                    "rank": rank,
                    "role": "primary" if k == cfg.primary_k and rank == "full" else "sensitivity",
                    "R2_MT_mean": float(g.R2_MT.mean()),
                    "R2_MTBobs_mean": float(g.R2_MTBobs.mean()),
                    "R2_MTQ_mean": float(g.R2_MTQ.mean()),
                    "R2_MTBpred_mean": float(g.R2_MTBpred.mean()),
                    "R2_Mfull_mean": float(g.R2_Mfull.mean()),
                    "delta_B_mean": sB["mean"],
                    "delta_B_median": sB["median"],
                    "delta_B_trimmed_mean": sB["trimmed_mean"],
                    "delta_B_frac_pos": sB["frac_pos"],
                    "delta_B_ci95_lo": sB["ci95_lo"],
                    "delta_B_ci95_hi": sB["ci95_hi"],
                    "delta_B_cluster_ci95_lo": float(np.quantile(cboots, 0.025)) if cboots else float("nan"),
                    "delta_B_cluster_ci95_hi": float(np.quantile(cboots, 0.975)) if cboots else float("nan"),
                    "p_delta_B": p_dB,
                    "B_mc": B,
                    "delta_Bpred_mean": float(g.delta_Bpred.mean()),
                    "delta_Q_mean": float(g.delta_Q.mean()),
                    "delta_full_mean": float(g.delta_full.mean()),
                    "spec_vs_random_mean": sR["mean"],
                    "spec_vs_random_ci95_lo": sR["ci95_lo"],
                    "spec_vs_random_ci95_hi": sR["ci95_hi"],
                    "p_spec_vs_random": p_spec,
                    "spec_vs_pca_mean": sP["mean"],
                    "spec_vs_shuffled_mean": sS["mean"],
                    "frac_random_beaten_mean": float(g.frac_random_beaten.mean()),
                    "MTQ_MTBpred_pred_corr_mean": float(g.MTQ_MTBpred_pred_corr.mean()),
                    "MTQ_MTBpred_abs_deltaR2_mean": float(g.MTQ_MTBpred_abs_deltaR2.mean()),
                    "rematerialize_max_rel_err": float(g.rematerialize_rel_err.max()),
                    **{f"rel_{kk}": vv.get("rho", float("nan")) for kk, vv in rel.items()},
                    **{f"relp_{kk}": vv.get("pvalue", float("nan")) for kk, vv in rel.items()},
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "summary_by_scale_rank.csv", index=False)

    primary = summary_df[(summary_df.scale_k == cfg.primary_k) & (summary_df["rank"] == "full")]
    primary_d = primary.iloc[0].to_dict() if len(primary) else {}
    label, statement = choose_label(primary_d, disjoint_summary, synth["pass"])

    # plots
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    g = res_df[(res_df.scale_k == cfg.primary_k) & (res_df["rank"] == "full")]
    if len(g):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        axes[0].hist(g.delta_B, bins=30, color="#1f4e79", alpha=0.85)
        axes[0].axvline(0, color="gray")
        axes[0].set_title("ΔB distribution k=2048")
        axes[1].scatter(g.delta_random_median, g.delta_B, s=12, alpha=0.5)
        axes[1].plot([g.delta_B.min(), g.delta_B.max()], [g.delta_B.min(), g.delta_B.max()], "k--", lw=0.8)
        axes[1].set_xlabel("median Δrandom")
        axes[1].set_ylabel("ΔB")
        axes[1].set_title("vs random normal")
        axes[2].scatter(g.A_B, g.delta_B, s=12, alpha=0.5, c="#b85c38")
        axes[2].set_xlabel("A_B (frozen)")
        axes[2].set_ylabel("ΔB")
        axes[2].set_title("ΔB vs prior alignment")
        fig.tight_layout()
        fig.savefig(fig_dir / "primary_deltaB_diagnostics.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for rank, marker in [("full", "o"), ("16", "s"), ("8", "^")]:
            for k, ls in [(2048, "-"), (1024, "--")]:
                row = summary_df[(summary_df.scale_k == k) & (summary_df["rank"] == rank)]
                if len(row):
                    ax.errorbar(
                        [k + ({"full": 0, "16": 40, "8": 80}[rank])],
                        row.delta_B_mean,
                        yerr=[[row.delta_B_mean.iloc[0] - row.delta_B_ci95_lo.iloc[0]], [row.delta_B_ci95_hi.iloc[0] - row.delta_B_mean.iloc[0]]],
                        fmt=marker,
                        label=f"k={k} rank={rank}",
                    )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("mean ΔB")
        ax.set_title("ΔB by scale and rank")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "deltaB_by_scale_rank.png", dpi=140)
        plt.close(fig)

    report = f"""# Curvature-subspace probe ablation

Frozen hash `{cfg.expected_hash}` verified. Rematerialize max rel err on primary full-rank:
{primary_d.get('rematerialize_max_rel_err', float('nan'))}.
Probe α={cfg.probe_alpha}. No curvature retune. No retrieval/Fisher/JS.

## Synthetics

pass={synth['pass']} checks={json.dumps(synth['checks'])}

## Primary result (k={cfg.primary_k}, rank=full)

{json.dumps({k: primary_d[k] for k in primary_d if k in ('delta_B_mean','delta_B_median','delta_B_frac_pos','delta_B_ci95_lo','delta_B_ci95_hi','delta_B_cluster_ci95_lo','delta_B_cluster_ci95_hi','p_delta_B','delta_Bpred_mean','delta_Q_mean','delta_full_mean','spec_vs_random_mean','p_spec_vs_random','spec_vs_pca_mean','spec_vs_shuffled_mean','frac_random_beaten_mean','MTQ_MTBpred_pred_corr_mean','R2_MT_mean','R2_MTBobs_mean','R2_MTQ_mean','R2_MTBpred_mean')}, indent=2)}

## Disjoint-neighbour sensitivity

{json.dumps(disjoint_summary, indent=2)}

## Summary table

{summary_df.to_string(index=False)}

## Interpretation

`{label}`

{statement}

Gates: MTBobs>MT; beats random/PCA/(shuffle); MTBpred or MTQ recovers substantial gain; disjoint-neighbour survives.

## Runtime

seconds={time.time()-t0:.1f} peak_rss_mb={max(peak, _rss()):.1f}
"""
    (out / "REPORT.md").write_text(report)
    analysis = {
        "label": label,
        "statement": statement,
        "primary": primary_d,
        "disjoint": disjoint_summary,
        "synthetic_pass": synth["pass"],
        "seconds": time.time() - t0,
        "peak_rss_mb": max(peak, _rss()),
        "config_hash": cfg.expected_hash,
        "command": (
            "PYTHONPATH=experiments python -m geometry.run_curvature_probe_subspace_ablation "
            "--force --n-random-controls 50 --seed 0"
        ),
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"[ablation] label={label} ΔB_mean={primary_d.get('delta_B_mean')} "
          f"spec_rand={primary_d.get('spec_vs_random_mean')} rss={analysis['peak_rss_mb']:.0f}", flush=True)
    return analysis
