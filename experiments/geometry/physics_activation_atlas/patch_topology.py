"""Bounded patchwise Rips diagnostics on local PCA coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _max_persistence(dgm: np.ndarray) -> float:
    if dgm is None or len(dgm) == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[np.isfinite(pers)]
    return float(pers.max()) if len(pers) else 0.0


def rips_h0_h1(U: np.ndarray, *, maxdim: int = 1) -> dict:
    try:
        from ripser import ripser
    except ImportError:
        try:
            import gudhi

            rc = gudhi.RipsComplex(points=U, max_edge_length=float(np.percentile(
                np.linalg.norm(U - U.mean(0), axis=1), 90
            )))
            st = rc.create_simplex_tree(max_dimension=maxdim)
            st.compute_persistence()
            dgms = []
            for d in range(maxdim + 1):
                iv = np.asarray(st.persistence_intervals_in_dimension(d), dtype=np.float64)
                if iv.size == 0:
                    iv = np.zeros((0, 2))
                dgms.append(iv)
            return {"H0_max_pers": _max_persistence(dgms[0]), "H1_max_pers": _max_persistence(dgms[1]) if maxdim >= 1 else 0.0}
        except ImportError:
            return {"error": "no_ripser_or_gudhi", "H0_max_pers": float("nan"), "H1_max_pers": float("nan")}
    out = ripser(U, maxdim=maxdim)
    dgms = out["dgms"]
    return {
        "H0_max_pers": _max_persistence(dgms[0]),
        "H1_max_pers": _max_persistence(dgms[1]) if len(dgms) > 1 else 0.0,
        "n_points": int(len(U)),
    }


def gaussian_null_max_h1(U: np.ndarray, *, seed: int, n_null: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    mu = U.mean(axis=0)
    cov = np.cov(U.T) + 1e-4 * np.eye(U.shape[1])
    vals = []
    for i in range(n_null):
        Z = rng.multivariate_normal(mu, cov, size=len(U))
        vals.append(rips_h0_h1(Z.astype(np.float32))["H1_max_pers"])
    return {"null_H1": vals, "null_H1_mean": float(np.nanmean(vals)), "null_H1_p90": float(np.nanpercentile(vals, 90))}


def patchwise_topology(
    coords_by_chart: dict[int, np.ndarray],
    weights_by_chart: dict[int, np.ndarray],
    *,
    max_points: int = 400,
    radius_quantiles: list[float] | None = None,
    min_chart_points: int = 80,
    seed: int = 0,
) -> dict:
    radius_quantiles = radius_quantiles or [0.5, 0.75, 1.0]
    rows = []
    for c, U in coords_by_chart.items():
        w = weights_by_chart[c]
        mask = w > 1e-6
        U = U[mask]
        w = w[mask]
        if len(U) < min_chart_points:
            continue
        # distance to chart origin in latent
        r = np.linalg.norm(U, axis=1)
        for q in radius_quantiles:
            thr = float(np.quantile(r, q))
            sel = r <= thr
            Us = U[sel]
            ws = w[sel]
            if len(Us) < 40:
                continue
            rng = np.random.default_rng(seed + c * 17 + int(100 * q))
            if len(Us) > max_points:
                p = ws / ws.sum()
                idx = rng.choice(len(Us), size=max_points, replace=False, p=p)
                Us = Us[idx]
            real = rips_h0_h1(Us)
            null = gaussian_null_max_h1(Us, seed=seed + c + int(10 * q), n_null=3)
            excess = float(real["H1_max_pers"] - null["null_H1_p90"])
            rows.append(
                {
                    "chart": int(c),
                    "radius_quantile": float(q),
                    "n_points": int(len(Us)),
                    "H1_max_pers": real["H1_max_pers"],
                    "null_H1_p90": null["null_H1_p90"],
                    "excess_over_null_p90": excess,
                    "excess_flag": bool(excess > 0 and np.isfinite(excess)),
                }
            )
    n_excess = sum(1 for r in rows if r["excess_flag"])
    return {
        "patches": rows,
        "n_patches": len(rows),
        "n_excess_H1": n_excess,
        "label": "patch_topology_not_detected" if n_excess == 0 else "patch_H1_exploratory_excess",
        "note": "exploratory; nontrivial patch H1 may reflect holes, folding, stratification or sampling",
    }


def save_patch_topology(out: Path, result: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "patch_topology.json").write_text(json.dumps(result, indent=2))
