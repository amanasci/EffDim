"""Permutation, paired bootstrap, and variance-threshold crossings.

Permutations operate on the small per-anchor scalar table only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.curvature_probe_screen import partial_spearman, spearman_dict


CONTROLS = ("log_knn_radius", "local_label_variance", "local_evaluation_count")


def _mask(x, y, Z=None):
    m = np.isfinite(x) & np.isfinite(y)
    if Z is not None:
        m = m & np.all(np.isfinite(Z), axis=1)
    return m


def associate(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> dict[str, float]:
    raw = spearman_dict(x, y)
    if Z is None:
        return {"raw": raw["rho"], "controlled": float("nan"), "n": raw["n"], "p_raw": raw["pvalue"]}
    ctl = partial_spearman(x, y, Z)
    return {"raw": raw["rho"], "controlled": ctl["rho"], "n": raw["n"], "p_raw": raw["pvalue"], "p_ctl": ctl["pvalue"]}


def control_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df[c].fillna(0).to_numpy(float) for c in CONTROLS])


def curve_from_panel(panel: pd.DataFrame, ds: list[int], *, ycol: str, xcol: str) -> pd.DataFrame:
    rows = []
    for d in ds:
        g = panel[panel.d == d]
        if not len(g):
            rows.append({"d": int(d), "raw": float("nan"), "controlled": float("nan"), "n": 0})
            continue
        y = g[ycol].to_numpy(float)
        x = g[xcol].to_numpy(float)
        Z = control_matrix(g)
        rec = associate(x, y, Z)
        rec["d"] = int(d)
        rows.append(rec)
    return pd.DataFrame(rows)


def permute_probe_once(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(y)


def freedman_lane_y(y: np.ndarray, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank-space Freedman–Lane: permute residuals of ranked y on ranked Z."""
    from scipy.stats import rankdata

    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y2 = y.copy()
    if int(m.sum()) < 12:
        return permute_probe_once(y, rng)
    yr = rankdata(y[m]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    b, *_ = np.linalg.lstsq(A, yr, rcond=None)
    fit = A @ b
    resid = yr - fit
    y2[m] = fit + rng.permutation(resid)
    return y2


def permutation_curves(
    wide: pd.DataFrame,
    ds: list[int],
    *,
    ycol: str,
    x_prefix: str,
    n_perm: int,
    seed: int,
    controlled: bool,
) -> dict[str, Any]:
    """wide: one row per anchor, columns x_prefix{d} and ycol and controls."""
    y = wide[ycol].to_numpy(float)
    Z = control_matrix(wide) if all(c in wide.columns for c in CONTROLS) else None
    xs = {d: wide[f"{x_prefix}{d}"].to_numpy(float) for d in ds}
    obs = {}
    for d in ds:
        obs[d] = associate(xs[d], y, Z if controlled else None)
    rng = np.random.default_rng(seed)
    null = {d: np.empty(n_perm) for d in ds}
    tmax = np.empty(n_perm)
    for b in range(n_perm):
        if controlled and Z is not None:
            yp = freedman_lane_y(y, Z, rng)
        else:
            yp = permute_probe_once(y, rng)
        rhos = []
        for d in ds:
            r = associate(xs[d], yp, Z if controlled else None)
            val = r["controlled"] if controlled else r["raw"]
            null[d][b] = val
            rhos.append(abs(val) if np.isfinite(val) else 0.0)
        tmax[b] = float(np.max(rhos)) if rhos else float("nan")
    rows = []
    finite_obs = [abs(obs[d]["controlled" if controlled else "raw"]) for d in ds if np.isfinite(obs[d]["controlled" if controlled else "raw"])]
    t_obs = max(finite_obs) if finite_obs else float("nan")
    for d in ds:
        val = obs[d]["controlled" if controlled else "raw"]
        pv = float(np.mean(np.abs(null[d]) >= abs(val))) if np.isfinite(val) else float("nan")
        pf = float(np.mean(tmax >= abs(val))) if np.isfinite(val) else float("nan")
        rows.append(
            {
                "d": int(d),
                "kind": "controlled" if controlled else "raw",
                "rho": val,
                "p_pointwise": pv,
                "p_fwer": pf,
                "n": obs[d]["n"],
            }
        )
    return {
        "obs": obs,
        "table": pd.DataFrame(rows),
        "tmax_obs": t_obs,
        "p_global": float(np.mean(tmax >= t_obs)) if np.isfinite(t_obs) else float("nan"),
        "null_envelope": {d: (float(np.quantile(np.abs(null[d]), 0.95))) for d in ds},
        "null": null,
        "tmax": tmax,
    }


def paired_bootstrap_curves(
    wide: pd.DataFrame,
    ds: list[int],
    *,
    ycol: str,
    x_prefix: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    n = len(wide)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    raw_b = np.full((n_boot, len(ds)), np.nan)
    ctl_b = np.full((n_boot, len(ds)), np.nan)
    y = wide[ycol].to_numpy(float)
    Z = control_matrix(wide)
    xs = {d: wide[f"{x_prefix}{d}"].to_numpy(float) for d in ds}
    for b in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        for j, d in enumerate(ds):
            rec = associate(xs[d][take], y[take], Z[take])
            raw_b[b, j] = rec["raw"]
            ctl_b[b, j] = rec["controlled"]
    obs_raw = np.array([associate(xs[d], y, Z)["raw"] for d in ds])
    obs_ctl = np.array([associate(xs[d], y, Z)["controlled"] for d in ds])

    def bands(obs, boot):
        lo = np.nanquantile(boot, 0.025, axis=0)
        hi = np.nanquantile(boot, 0.975, axis=0)
        center = boot - obs[None, :]
        dev = np.nanmax(np.abs(center), axis=1)
        q = float(np.nanquantile(dev, 0.95))
        return lo, hi, obs - q, obs + q, q

    rlo, rhi, rslo, rshi, rq = bands(obs_raw, raw_b)
    clo, chi, cslo, cshi, cq = bands(obs_ctl, ctl_b)
    peak_raw = [int(ds[int(np.nanargmax(np.abs(raw_b[b])))]) for b in range(n_boot) if np.any(np.isfinite(raw_b[b]))]
    peak_ctl = [int(ds[int(np.nanargmax(np.abs(ctl_b[b])))]) for b in range(n_boot) if np.any(np.isfinite(ctl_b[b]))]
    i12 = ds.index(12) if 12 in ds else None
    i16 = ds.index(16) if 16 in ds else None
    mid = [j for j, d in enumerate(ds) if 13 <= d <= 16]
    rows = []
    for j, d in enumerate(ds):
        rows.append(
            {
                "d": int(d),
                "raw": obs_raw[j],
                "raw_lo": rlo[j],
                "raw_hi": rhi[j],
                "raw_sim_lo": rslo[j],
                "raw_sim_hi": rshi[j],
                "controlled": obs_ctl[j],
                "ctl_lo": clo[j],
                "ctl_hi": chi[j],
                "ctl_sim_lo": cslo[j],
                "ctl_sim_hi": cshi[j],
            }
        )
    extra = {
        "delta_16_12_raw": float(obs_raw[i16] - obs_raw[i12]) if i12 is not None and i16 is not None else float("nan"),
        "delta_16_12_raw_lo": float(np.nanquantile(raw_b[:, i16] - raw_b[:, i12], 0.025)) if i12 is not None and i16 is not None else float("nan"),
        "delta_16_12_raw_hi": float(np.nanquantile(raw_b[:, i16] - raw_b[:, i12], 0.975)) if i12 is not None and i16 is not None else float("nan"),
        "delta_16_12_ctl": float(obs_ctl[i16] - obs_ctl[i12]) if i12 is not None and i16 is not None else float("nan"),
        "delta_16_12_ctl_lo": float(np.nanquantile(ctl_b[:, i16] - ctl_b[:, i12], 0.025)) if i12 is not None and i16 is not None else float("nan"),
        "delta_16_12_ctl_hi": float(np.nanquantile(ctl_b[:, i16] - ctl_b[:, i12], 0.975)) if i12 is not None and i16 is not None else float("nan"),
        "mean_13_16_raw": float(np.nanmean(obs_raw[mid])) if mid else float("nan"),
        "mean_13_16_ctl": float(np.nanmean(obs_ctl[mid])) if mid else float("nan"),
        "peak_raw_mode": int(pd.Series(peak_raw).mode().iloc[0]) if peak_raw else None,
        "peak_ctl_mode": int(pd.Series(peak_ctl).mode().iloc[0]) if peak_ctl else None,
        "sim_q_raw": rq,
        "sim_q_ctl": cq,
    }
    return {
        "table": pd.DataFrame(rows),
        "extra": extra,
        "peak_raw": peak_raw,
        "peak_ctl": peak_ctl,
        "raw_boot": raw_b,
        "ctl_boot": ctl_b,
        "ds": ds,
    }


def crossing_d(ds: np.ndarray, r2: np.ndarray, tau: float):
    hit = [int(d) for d, v in zip(ds, r2) if np.isfinite(v) and v >= tau]
    return int(min(hit)) if hit else "not_reached"


def bootstrap_crossings(
    per_anchor: pd.DataFrame,
    ds: list[int],
    taus: list[float],
    *,
    n_boot: int,
    seed: int,
    r2_col: str = "lin_r2",
) -> dict[str, Any]:
    """Pooled crossing uses energy if present, else mean lin_r2."""
    loc = per_anchor.groupby(["sample_id", "d"], as_index=False).mean(numeric_only=True)
    sids = loc.sample_id.unique()
    rng = np.random.default_rng(seed)

    def curve(sids_take):
        g = loc[loc.sample_id.isin(sids_take)]
        out = []
        for d in ds:
            gd = g[g.d == d]
            if "test_energy" in gd.columns and "test_sse_lin" in gd.columns and gd.test_energy.sum() > 0:
                r2 = 1.0 - float(gd.test_sse_lin.sum() / gd.test_energy.sum())
            else:
                r2 = float(gd[r2_col].mean()) if r2_col in gd.columns and len(gd) else float("nan")
            out.append(r2)
        return np.array(out)

    obs = curve(sids)
    boots = np.array([curve(rng.choice(sids, size=len(sids), replace=True)) for _ in range(n_boot)])
    rows = []
    for tau in taus:
        xs = [crossing_d(np.array(ds), boots[b], tau) for b in range(n_boot)]
        nums = [x for x in xs if isinstance(x, int)]
        rows.append(
            {
                "tau": float(tau),
                "d_tau": crossing_d(np.array(ds), obs, tau),
                "d_tau_lo": int(np.quantile(nums, 0.025)) if nums else "not_reached",
                "d_tau_hi": int(np.quantile(nums, 0.975)) if nums else "not_reached",
                "frac_reached": float(len(nums) / max(n_boot, 1)),
                "post_hoc": bool(abs(tau - 0.85) < 1e-9),
            }
        )
    ve = pd.DataFrame({"d": ds, "r2_L_pooled": obs})
    for q, name in ((0.025, "r2_lo"), (0.975, "r2_hi")):
        ve[name] = np.nanquantile(boots, q, axis=0)
    return {"crossings": pd.DataFrame(rows), "curve": ve}
