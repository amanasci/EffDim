"""Association stability, field reliability, and combined resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix

from .config import (
    ANCHOR_BOOT_SEED,
    COMBINED_SEED,
    CONTROLS,
    GATE_MEDIAN_TOL,
    GATE_SIGN_FRAC,
    N_BOOT_COMBINED,
)


def spearman_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8 or float(np.std(a[m])) < 1e-15 or float(np.std(b[m])) < 1e-15:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def _ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, dtype=np.float64)
    r[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return r


def residualize(y, Z) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    A = np.column_stack([np.ones(len(y)), np.asarray(Z, dtype=np.float64)])
    m = np.isfinite(y) & np.all(np.isfinite(A), axis=1)
    out = np.full(len(y), np.nan)
    if int(m.sum()) < A.shape[1] + 2:
        return out
    beta, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
    out[m] = y[m] - A[m] @ beta
    return out


def assoc_field(kh: np.ndarray, df: pd.DataFrame, *, extra_z: np.ndarray | None = None) -> dict[str, dict]:
    Z = control_matrix(df)
    if extra_z is not None:
        Z = np.column_stack([Z, np.asarray(extra_z, dtype=np.float64)])
    out = {}
    for name, col in (
        ("r2_G", "r2_G"),
        ("mse_G", "mse_G"),
        ("r2_P", "r2_P"),
        ("mse_P", "mse_P"),
        ("delta_adapt", "delta_adapt"),
    ):
        if col not in df.columns:
            continue
        out[name] = associate(np.asarray(kh, float), df[col].to_numpy(float), Z)
    return out


def summarize_replicates(vals: np.ndarray, original: float, expected_sign: int) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": float("nan")}
    lo, hi = np.percentile(v, [2.5, 97.5])
    q1, q3 = np.percentile(v, [25, 75])
    return {
        "n": int(v.size),
        "original": float(original),
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "iqr": float(q3 - q1),
        "q025": float(lo),
        "q975": float(hi),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "frac_original_sign": float(np.mean(np.sign(v) == np.sign(expected_sign) if expected_sign != 0 else np.sign(v) == np.sign(original))),
        "frac_abs_ge_original": float(np.mean(np.abs(v) >= abs(original))) if np.isfinite(original) else float("nan"),
        "frac_within_0.05": float(np.mean(np.abs(v - original) <= 0.05)) if np.isfinite(original) else float("nan"),
        "interval_excludes_zero_expected": bool((lo > 0 and expected_sign > 0) or (hi < 0 and expected_sign < 0)),
        "label": "geometry_resampling_stability_interval",
    }


def field_reliability(wide: pd.DataFrame, frozen: np.ndarray, extra_controls: dict[str, np.ndarray] | None = None) -> dict:
    cols = [c for c in wide.columns if str(c).startswith("r")]
    mats = wide[cols].to_numpy(float)
    r_b0 = [spearman_safe(mats[:, j], frozen) for j in range(mats.shape[1])]
    pairs = []
    jac_hi, jac_lo, sign_ag, rmse = [], [], [], []
    n = mats.shape[0]
    for a in range(mats.shape[1]):
        for b in range(a + 1, mats.shape[1]):
            xa, xb = mats[:, a], mats[:, b]
            pairs.append(spearman_safe(xa, xb))
            ha, hb = xa >= np.nanpercentile(xa, 75), xb >= np.nanpercentile(xb, 75)
            la, lb = xa <= np.nanpercentile(xa, 25), xb <= np.nanpercentile(xb, 25)
            jac_hi.append(float(np.sum(ha & hb) / max(np.sum(ha | hb), 1)))
            jac_lo.append(float(np.sum(la & lb) / max(np.sum(la | lb), 1)))
            sa, sb = np.sign(xa), np.sign(xb)
            m = np.isfinite(sa) & np.isfinite(sb)
            sign_ag.append(float(np.mean(sa[m] == sb[m])))
            ra, rb = _ranks(np.nan_to_num(xa)), _ranks(np.nan_to_num(xb))
            rmse.append(float(np.sqrt(np.mean((ra - rb) ** 2))))
    # rank ICC (two-way, consistency): variance of row means / total on ranks
    R = np.column_stack([_ranks(np.nan_to_num(mats[:, j])) for j in range(mats.shape[1])])
    row_var = float(np.var(R.mean(1), ddof=1))
    tot_var = float(np.var(R, ddof=1))
    icc = row_var / max(tot_var, 1e-18)
    rec = {
        "n_replicates": int(mats.shape[1]),
        "n_anchors": int(n),
        "r_b0_median": float(np.nanmedian(r_b0)),
        "r_b0_iqr": float(np.nanpercentile(r_b0, 75) - np.nanpercentile(r_b0, 25)),
        "r_b0_q025": float(np.nanpercentile(r_b0, 2.5)),
        "r_b0_q975": float(np.nanpercentile(r_b0, 97.5)),
        "pairwise_median": float(np.nanmedian(pairs)) if pairs else float("nan"),
        "rank_icc": float(icc),
        "top_quartile_jaccard_median": float(np.nanmedian(jac_hi)) if jac_hi else float("nan"),
        "bottom_quartile_jaccard_median": float(np.nanmedian(jac_lo)) if jac_lo else float("nan"),
        "sign_agreement_median": float(np.nanmedian(sign_ag)) if sign_ag else float("nan"),
        "rank_rmse_median": float(np.nanmedian(rmse)) if rmse else float("nan"),
        "r_b0": r_b0,
    }
    if extra_controls:
        # geometry-only partial agreement: median r_b0 after residualizing ranks on controls
        Z = np.column_stack([np.asarray(v, float) for v in extra_controls.values()])
        zr = np.column_stack([_ranks(Z[:, j]) for j in range(Z.shape[1])])
        fr = residualize(_ranks(frozen), zr)
        pb = []
        for j in range(mats.shape[1]):
            pb.append(spearman_safe(residualize(_ranks(mats[:, j]), zr), fr))
        rec["partial_r_b0_median"] = float(np.nanmedian(pb))
        rec["partial_controls"] = list(extra_controls.keys())
    return rec


def anchor_only_bootstrap(kh: np.ndarray, df: pd.DataFrame, n_boot: int = N_BOOT_COMBINED, seed: int = ANCHOR_BOOT_SEED) -> dict[str, dict]:
    rng = np.random.default_rng(int(seed))
    n = len(df)
    store = {k: [] for k in ("r2_G", "mse_G", "r2_P", "mse_P", "delta_adapt")}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        sub = df.iloc[idx].reset_index(drop=True)
        rec = assoc_field(np.asarray(kh)[idx], sub)
        for k, v in rec.items():
            store[k].append(v["controlled"])
    out = {}
    signs = {"r2_G": -1, "mse_G": 1, "r2_P": 1, "mse_P": -1, "delta_adapt": 1}
    obs = assoc_field(kh, df)
    for k, vals in store.items():
        if not vals:
            continue
        a = np.asarray(vals, float)
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        out[k] = {
            "original": obs[k]["controlled"],
            "q025": float(lo),
            "q975": float(hi),
            "label": "anchor_only_bootstrap_interval",
        }
    return out


def combined_interval(
    fields: pd.DataFrame,
    df: pd.DataFrame,
    n_boot: int = N_BOOT_COMBINED,
    seed: int = COMBINED_SEED,
) -> dict[str, dict]:
    reps = sorted(fields.replicate.unique())
    rng = np.random.default_rng(int(seed))
    n = len(df)
    sid = df.sample_id.to_numpy(int)
    store = {k: [] for k in ("r2_G", "mse_G", "r2_P", "mse_P", "delta_adapt")}
    for _ in range(int(n_boot)):
        b = int(rng.choice(reps))
        subf = fields[fields.replicate == b].set_index("sample_id").loc[sid]
        idx = rng.integers(0, n, n)
        rec = assoc_field(subf["K_H_cross"].to_numpy(float)[idx], df.iloc[idx].reset_index(drop=True))
        for k, v in rec.items():
            store[k].append(v["controlled"])
    out = {}
    for k, vals in store.items():
        if not vals:
            continue
        a = np.asarray(vals, float)
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        out[k] = {"q025": float(lo), "q975": float(hi), "label": "combined_geometry_anchor_resampling_interval"}
    return out


def global_pass(stab: dict) -> bool:
    r2, mse = stab["r2_G"], stab["mse_G"]
    return bool(
        r2.get("frac_original_sign", 0) >= GATE_SIGN_FRAC
        and mse.get("frac_original_sign", 0) >= GATE_SIGN_FRAC
        and r2.get("interval_excludes_zero_expected", False)
        and mse.get("interval_excludes_zero_expected", False)
        and abs(r2["median"] - r2["original"]) <= GATE_MEDIAN_TOL
        and abs(mse["median"] - mse["original"]) <= GATE_MEDIAN_TOL
    )


def adapt_pass(stab: dict) -> bool:
    a = stab["delta_adapt"]
    return bool(
        a.get("frac_original_sign", 0) >= GATE_SIGN_FRAC
        and a.get("interval_excludes_zero_expected", False)
        and abs(a["median"] - a["original"]) <= GATE_MEDIAN_TOL
    )
