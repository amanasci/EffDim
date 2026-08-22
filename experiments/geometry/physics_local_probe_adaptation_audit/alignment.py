"""Reconstruct current-anchor A_H^G and A_B^G without refitting curvature."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.global_probe_curvature_alignment import EPS, projection_energies
from geometry.physics_activation_atlas.global_probe_curvature_magnitude import a_h_from_w_H

from .config import MODEL, PRIMARY_D, PRIMARY_K, TARGET, SOURCE_ALIGN, SOURCE_GEOM, SOURCE_H, WEIGHT_COS_RELIABLE


def _geom_pack(cache: Path, ai: int) -> dict | None:
    p = cache / f"k{PRIMARY_K}_ai{int(ai):04d}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def global_fold_weights(gw: pd.DataFrame, target: str) -> dict[int, tuple[np.ndarray, float]]:
    sub = gw[(gw.target == target)]
    out = {}
    for f, g in sub.groupby("fold"):
        # coef stored as list in parquet? check
        row = g.iloc[0]
        coef = np.asarray(row["coef"] if "coef" in row else row.get("coef_vec", []), dtype=float)
        if coef.size == 0 and "coef_norm" in row:
            # only norm stored - fall back to npz pooled weight
            return {}
        out[int(f)] = (coef, float(row["intercept"]))
    return out


def alignment_global_one(
    *,
    w: np.ndarray,
    pack: dict,
    H: np.ndarray,
) -> dict[str, float]:
    T = np.asarray(pack["T"], dtype=float)
    x0u = np.asarray(pack["x0u"], dtype=float)
    UB = np.asarray(pack["UB"], dtype=float)
    UN = np.asarray(pack["UNPCA"], dtype=float)
    en = projection_energies(w, T, x0u, UB, UN)
    a_h, unstable = a_h_from_w_H(w, T, x0u, H)
    return {
        "A_B_G": float(en["A_B_normal"]),
        "A_H_G": float(a_h),
        "A_N_G": float(en["A_N"]),
        "H_unstable": bool(unstable),
        "w_norm_G": float(np.linalg.norm(w)),
    }


def patch_weights_oof(
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    neigh_idx: np.ndarray,
    *,
    alpha: float,
) -> tuple[list[np.ndarray], float]:
    """Return list of fold weight vectors and median fold cosine stability."""
    from geometry.physics_local_probe_adaptation.ridge import ridge_fit_intercept

    idx = np.asarray(neigh_idx, dtype=int)
    ws = []
    for f in sorted(set(fold[idx].tolist())):
        tr = idx[fold[idx] != f]
        if len(tr) < 32:
            continue
        w, b, info = ridge_fit_intercept(X[tr], y[tr], alpha=alpha)
        if info["ok"]:
            ws.append(w)
    cos = []
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            u, v = ws[i], ws[j]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 0 and nv > 0:
                cos.append(float(np.dot(u, v) / (nu * nv)))
    stab = float(np.median(cos)) if cos else float("nan")
    return ws, stab


def alignment_patch_from_weights(
    ws: list[np.ndarray],
    pack: dict,
    H: np.ndarray,
    fold_counts: np.ndarray | None = None,
) -> dict[str, float]:
    if not ws:
        return {k: float("nan") for k in ("A_B_P", "A_H_P", "A_B_P_wmean", "A_H_P_wmean")}
    T = np.asarray(pack["T"], dtype=float)
    x0u = np.asarray(pack["x0u"], dtype=float)
    UB = np.asarray(pack["UB"], dtype=float)
    UN = np.asarray(pack["UNPCA"], dtype=float)
    ab, ah = [], []
    for w in ws:
        en = projection_energies(w, T, x0u, UB, UN)
        a_h, _ = a_h_from_w_H(w, T, x0u, H)
        ab.append(en["A_B_normal"])
        ah.append(a_h)
    wts = np.ones(len(ws)) if fold_counts is None else np.asarray(fold_counts[: len(ws)], float)
    wts = wts / max(wts.sum(), 1e-12)
    return {
        "A_B_P": float(np.mean(ab)),
        "A_H_P": float(np.mean(ah)),
        "A_B_P_wmean": float(np.dot(wts, ab)),
        "A_H_P_wmean": float(np.dot(wts, ah)),
        "n_patch_folds": int(len(ws)),
    }


def weight_angle(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    return float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)))


def build_alignment_table(
    root: Path,
    *,
    sids: list[int],
    sid_to_ai: dict[int, int],
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    neigh: np.ndarray,
    gw: pd.DataFrame,
    w_pooled: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    cache = root / SOURCE_GEOM.replace("outputs/geometry/", "outputs/geometry/")
    cache = root / "outputs/geometry/physics_curvature_probe_multitarget/geometry_cache"
    hdir = root / "outputs/geometry/physics_nested_dimension_curvature/H_vectors"
    rows = []
    fold_w = global_fold_weights(gw, TARGET)
    use_pooled = len(fold_w) == 0
    for sid in sids:
        ai = sid_to_ai[int(sid)]
        pack = _geom_pack(cache, ai)
        if pack is None:
            continue
        hp = hdir / f"{int(sid)}.npz"
        if not hp.exists():
            continue
        H = np.asarray(np.load(hp)["H16"], dtype=float)
        N = neigh[ai, :PRIMARY_K]
        # global alignment: fold-weighted by neighbours in each fold
        if use_pooled:
            g = alignment_global_one(w=w_pooled, pack=pack, H=H)
            wG = w_pooled
        else:
            abs_, ahs_, wsum = [], [], []
            wG = None
            for f, (coef, _b) in fold_w.items():
                cnt = int(np.sum(fold[N] == f))
                if cnt < 1:
                    continue
                g1 = alignment_global_one(w=coef, pack=pack, H=H)
                abs_.append(g1["A_B_G"])
                ahs_.append(g1["A_H_G"])
                wsum.append(cnt)
                if wG is None:
                    wG = coef
            ws = np.asarray(wsum, float)
            ws = ws / max(ws.sum(), 1e-12)
            g = {
                "A_B_G": float(np.dot(ws, abs_)),
                "A_H_G": float(np.dot(ws, ahs_)),
                "A_N_G": float("nan"),
                "H_unstable": False,
                "w_norm_G": float(np.linalg.norm(wG)) if wG is not None else float("nan"),
            }
        pws, stab = patch_weights_oof(X, y, fold, N, alpha=alpha)
        ap = alignment_patch_from_weights(pws, pack, H)
        wP_mean = np.mean(pws, axis=0) if pws else np.zeros_like(wG)
        ang = weight_angle(wG, wP_mean) if wG is not None and len(pws) else float("nan")
        T = np.asarray(pack["T"], dtype=float)
        wG_t = T @ (T.T @ wG) if wG is not None else np.zeros(X.shape[1])
        wP_t = T @ (T.T @ wP_mean) if len(pws) else np.zeros(X.shape[1])
        ang_t = weight_angle(wG_t, wP_t)
        rows.append(
            {
                "sample_id": int(sid),
                "anchor_local": int(ai),
                **g,
                **ap,
                "dA_B": float(ap["A_B_P_wmean"] - g["A_B_G"]) if np.isfinite(ap["A_B_P_wmean"]) else float("nan"),
                "dA_H": float(ap["A_H_P_wmean"] - g["A_H_G"]) if np.isfinite(ap["A_H_P_wmean"]) else float("nan"),
                "weight_angle": ang,
                "tangent_weight_angle": ang_t,
                "P_fold_cosine_med": stab,
                "direction_reliable": bool(np.isfinite(stab) and stab >= WEIGHT_COS_RELIABLE),
            }
        )
    return pd.DataFrame(rows)
