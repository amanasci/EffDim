"""Cell associations, paired contrasts, reliability, drift, decision label."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_curvature_probe_rank_sweep.inference import CONTROLS, associate, control_matrix

from .config import N_BOOT, N_PERM, PRIMARY_D, DECISION_LABELS
from .io_util import fisher_z, p_mc


def _ctrl_df(g: pd.DataFrame, prefix: str = "ctl_") -> pd.DataFrame:
    cols = {}
    for c in CONTROLS:
        key = prefix + c
        if key in g.columns:
            cols[c] = g[key].to_numpy(float)
        elif c in g.columns:
            cols[c] = g[c].to_numpy(float)
        else:
            cols[c] = np.full(len(g), np.nan)
    return pd.DataFrame(cols)


def _assoc_one(g: pd.DataFrame, xcol: str, ycol: str, prefix: str) -> dict[str, float]:
    # reset_index so control columns assign by row position, not leftover groupby labels
    sub = g.reset_index(drop=True).copy()
    ctrl = _ctrl_df(sub, prefix)
    for c in CONTROLS:
        sub[c] = ctrl[c].to_numpy(dtype=float)
    return associate(sub[xcol].to_numpy(float), sub[ycol].to_numpy(float), control_matrix(sub))


def summarize_cells(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (R, m, d, rep), g in df.groupby(["R", "m", "d", "replicate"]):
        gg = g[g.ok.fillna(True).astype(bool)]
        if len(gg) < 8:
            continue
        rec = {
            "R": int(R),
            "m": int(m),
            "d": int(d),
            "replicate": int(rep),
            "n": int(len(gg)),
            "R_H_med": float(gg.R_H.median()) if "R_H" in gg else float("nan"),
            "fail_frac": float((~g.ok.fillna(True).astype(bool)).mean()),
        }
        a = _assoc_one(gg, "K_H_cross", "r2_k2048", "ctl_")
        rec.update({f"r2fix_{k}": v for k, v in a.items()})
        b = _assoc_one(gg, "K_H_cross", "mse_k2048", "ctl_")
        rec.update({f"msefix_{k}": v for k, v in b.items()})
        c = _assoc_one(gg, "K_H_cross", "sst_k2048", "ctl_")
        rec.update({f"sstfix_{k}": v for k, v in c.items()})
        dlt = _assoc_one(gg, "K_H_cross", "r2_matched", "ctlR_")
        rec.update({f"r2match_{k}": v for k, v in dlt.items()})
        e = _assoc_one(gg, "K_H_cross", "mse_matched", "ctlR_")
        rec.update({f"msematch_{k}": v for k, v in e.items()})
        rows.append(rec)
    return pd.DataFrame(rows)


def _cell_slice(df: pd.DataFrame, R: int, m: int, d: int) -> pd.DataFrame:
    return df[(df.R == R) & (df.m == m) & (df.d == d)].copy()


def _rho_on_ids(g: pd.DataFrame, ycol: str, ids: np.ndarray | list[int]) -> float:
    """Controlled Spearman on one replicate restricted to `ids` (with replacement OK)."""
    ids = np.asarray(ids, dtype=int)
    # preserve bootstrap multiplicity
    parts = []
    by = g.set_index("sample_id")
    for i, sid in enumerate(ids):
        if sid not in by.index:
            continue
        row = by.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rec = row.to_dict()
        rec["sample_id"] = int(sid)
        rec["_boot"] = i
        parts.append(rec)
    if len(parts) < 8:
        return float("nan")
    sub = pd.DataFrame(parts)
    return float(_assoc_one(sub, "K_H_cross", ycol, "ctl_")["controlled"])


def _median_rho_cell(g: pd.DataFrame, ycol: str, ids: np.ndarray | list[int] | None = None) -> float:
    """Replicate-median controlled ρ for one (R,m,d) cell — primary cell summary."""
    if ids is None:
        ids = sorted(int(s) for s in g.sample_id.unique())
    rhos = []
    for _, gg in g.groupby("replicate"):
        rhos.append(_rho_on_ids(gg, ycol, ids))
    return float(np.nanmedian(rhos)) if rhos else float("nan")


def contrast_stats(
    df: pd.DataFrame,
    *,
    ycol: str,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """Primary contrasts on replicate-median controlled Spearman (not ρ of median K_H)."""
    d = PRIMARY_D
    g_hi = _cell_slice(df, 2048, 2048, d)
    g_mid = _cell_slice(df, 2048, 1024, d)
    g_loR = _cell_slice(df, 1024, 1024, d)
    g_midR = _cell_slice(df, 1536, 1024, d)
    sids = sorted(
        set(g_hi.sample_id.unique()) & set(g_mid.sample_id.unique()) & set(g_loR.sample_id.unique())
    )
    rho_hi = _median_rho_cell(g_hi, ycol, sids)
    rho_mid = _median_rho_cell(g_mid, ycol, sids)
    rho_loR = _median_rho_cell(g_loR, ycol, sids)
    rho_midR = _median_rho_cell(g_midR, ycol, sids)
    d_count = rho_hi - rho_mid
    d_radius = rho_mid - rho_loR

    # Probe outcomes are fixed across curvature replicates; permute at the anchor level.
    y_map = g_mid.drop_duplicates("sample_id").set_index("sample_id")[ycol].to_dict()

    def with_y(g: pd.DataFrame, ydict: dict) -> pd.DataFrame:
        out = g.copy()
        out[ycol] = out.sample_id.map(ydict)
        return out

    rng = np.random.default_rng(seed)
    boot_c, boot_r = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(sids, size=len(sids), replace=True)
        boot_c[b] = _median_rho_cell(g_hi, ycol, draw) - _median_rho_cell(g_mid, ycol, draw)
        boot_r[b] = _median_rho_cell(g_mid, ycol, draw) - _median_rho_cell(g_loR, ycol, draw)

    yvals = np.array([y_map[s] for s in sids], dtype=float)
    perm_c, perm_r = np.empty(n_perm), np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(yvals)
        ydict = {s: float(v) for s, v in zip(sids, yp)}
        hi_p, mid_p, lo_p = with_y(g_hi, ydict), with_y(g_mid, ydict), with_y(g_loR, ydict)
        perm_c[b] = _median_rho_cell(hi_p, ycol, sids) - _median_rho_cell(mid_p, ycol, sids)
        perm_r[b] = _median_rho_cell(mid_p, ycol, sids) - _median_rho_cell(lo_p, ycol, sids)

    def pack(name, obs, boot, perm, rho_a, rho_b):
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        b = int(np.sum(np.abs(perm) >= abs(obs))) if np.isfinite(obs) else n_perm
        return {
            "name": name,
            "estimate": float(obs),
            "fisher_z_delta": float(fisher_z(rho_a) - fisher_z(rho_b))
            if np.isfinite(rho_a) and np.isfinite(rho_b)
            else float("nan"),
            "ci95": [float(lo), float(hi)],
            "p_mc": p_mc(b, n_perm),
        }

    c1 = pack("delta_count", d_count, boot_c, perm_c, rho_hi, rho_mid)
    c2 = pack("delta_radius", d_radius, boot_r, perm_r, rho_mid, rho_loR)
    ps = np.array([c1["p_mc"], c2["p_mc"]])
    order = np.argsort(ps)
    holm = np.empty(2)
    for rank, i in enumerate(order):
        holm[i] = min(1.0, (2 - rank) * ps[i])
    c1["p_holm"] = float(holm[0])
    c2["p_holm"] = float(holm[1])
    return {
        "ycol": ycol,
        "estimand": "replicate_median_controlled_spearman",
        "rho_R2048_m2048": rho_hi,
        "rho_R2048_m1024": rho_mid,
        "rho_R1024_m1024": rho_loR,
        "rho_R1536_m1024": rho_midR,
        "delta_count": c1,
        "delta_radius": c2,
        "n_boot": n_boot,
        "n_perm": n_perm,
    }


def reliability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (R, m, d), g in df.groupby(["R", "m", "d"]):
        wide = g.pivot_table(index="sample_id", columns="replicate", values="K_H_cross")
        pairs = []
        cols = list(wide.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = wide[cols[i]].to_numpy(float)
                b = wide[cols[j]].to_numpy(float)
                msk = np.isfinite(a) & np.isfinite(b)
                if msk.sum() >= 8:
                    pairs.append(float(pd.Series(a[msk]).corr(pd.Series(b[msk]), method="spearman")))
        rows.append(
            {
                "R": int(R),
                "m": int(m),
                "d": int(d),
                "R_H_med": float(g.R_H.median()) if "R_H" in g.columns else float("nan"),
                "repeat_spearman_med": float(np.median(pairs)) if pairs else float("nan"),
                "kh_cv_med": float(g.groupby("sample_id").K_H_cross.std().median() / max(abs(g.K_H_cross.median()), 1e-12)),
                "n": int(g.sample_id.nunique()),
            }
        )
    return pd.DataFrame(rows)


def drift_table(out: Path, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = PRIMARY_D
    path_a = out / "cells" / f"H_R1024_m1024_d{d}.npz"
    path_b = out / "cells" / f"H_R2048_m1024_d{d}.npz"
    if not (path_a.exists() and path_b.exists()):
        return pd.DataFrame()
    A = np.load(path_a)
    B = np.load(path_b)
    # pair on sample_id, replicate 0
    def first_rep(z):
        m = z["replicate"] == 0
        return {int(s): z["H"][i] for i, s in enumerate(z["sample_id"]) if m[i]}

    ha, hb = first_rep(A), first_rep(B)
    for sid in sorted(set(ha) & set(hb)):
        u, v = ha[sid], hb[sid]
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            continue
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        cos = float(np.dot(u, v) / max(nu * nv, 1e-18))
        rows.append(
            {
                "sample_id": int(sid),
                "cosine": cos,
                "norm_ratio": float(nv / max(nu, 1e-18)),
                "l2_diff": float(np.linalg.norm(u - v)),
            }
        )
    return pd.DataFrame(rows)


def decide(contrasts: dict, rel: pd.DataFrame, drift: pd.DataFrame) -> dict[str, Any]:
    c = contrasts["delta_count"]
    r = contrasts["delta_radius"]
    rel16 = rel[rel.d == PRIMARY_D]
    r_m = rel16[rel16.R == 2048].sort_values("m")
    rel_up = False
    if len(r_m) >= 2:
        rel_up = bool(r_m.iloc[-1].repeat_spearman_med >= r_m.iloc[0].repeat_spearman_med - 1e-6)
    count_sig = c["p_holm"] <= 0.05
    radius_sig = r["p_holm"] <= 0.05
    count_neg = c["estimate"] < 0  # more negative association as m grows
    radius_pos = r["estimate"] > 0  # toward zero as R grows (if rhos negative)
    cos_med = float(drift.cosine.median()) if len(drift) else float("nan")
    hetero = bool(np.isfinite(cos_med) and cos_med < 0.7)

    if count_sig and count_neg and rel_up and not (radius_sig and radius_pos):
        label = "finite_sample_attenuation_supported"
    elif radius_sig and radius_pos and hetero and not (count_sig and count_neg and rel_up and abs(c["estimate"]) > abs(r["estimate"])):
        label = "geometric_washout_supported"
    elif count_sig and radius_sig:
        label = "mixed_bias_variance"
    else:
        label = "mechanism_unresolved"
    assert label in DECISION_LABELS
    return {
        "label": label,
        "count_strengthens_negative": bool(count_neg),
        "radius_moves_toward_zero": bool(radius_pos),
        "reliability_rises_with_m": rel_up,
        "median_cross_radius_cosine": cos_med,
        "contrasts": contrasts,
        "note_R_H": (
            "R_H is split-half concordance of H, not classical test-retest reliability of K_H_cross; "
            "repeat Spearman of K_H_cross across seeds is used for the attenuation diagnostic. "
            "Correlations are not disattenuated."
        ),
    }


def manuscript_action(label: str) -> dict[str, str]:
    retain = (
        "The association is supported at the frozen k=2048 scale and is neighbourhood-scale dependent."
    )
    if label == "finite_sample_attenuation_supported":
        return {
            "action": "revise_toward_finite_sample_attenuation",
            "sentence": (
                "A factorial split of neighbourhood radius and curvature-fit sample count indicates that "
                "the weaker associations at smaller k are consistent with finite-sample attenuation of the "
                "cross-split curvature statistic, not with geometric averaging over the tested radii."
            ),
        }
    if label == "geometric_washout_supported":
        return {
            "action": "revise_toward_genuine_geometric_averaging",
            "sentence": (
                "Holding the curvature-fit sample count fixed, expanding geometric support moved the "
                "association toward zero, consistent with averaging of spatially heterogeneous curvature."
            ),
        }
    if label == "mixed_bias_variance":
        return {
            "action": "leave_mechanism_unresolved_or_report_both",
            "sentence": (
                "Both finite-sample attenuation and geometric support radius contribute independently "
                "to neighbourhood-scale sensitivity; the manuscript should retain scale-dependence language."
            ),
        }
    return {
        "action": "retain_current_not_robust_across_scale_language",
        "sentence": retain + " The (R, m) factorial did not isolate a single mechanism.",
    }


summarize_cells = summarize_cells
contrast_stats = contrast_stats
reliability_table = reliability_table
drift_table = drift_table
decide = decide
manuscript_action = manuscript_action
