"""Phases 6–7: dependence-preserving global corrections and transition tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ALPHA,
    DISCOVERY_DATASET,
    DISCOVERY_LABEL,
    FROZEN_CTL,
    FROZEN_D80,
    FROZEN_D85,
    FROZEN_RAW,
    SHARED_CORE_CONTROLS,
)
from .inventory import load_frozen_probe
from .parity import load_catalog_label
from .pipeline import (
    AuditConfig,
    associate,
    control_matrix,
    delta_85_80,
    p_monte_carlo_ci,
    p_report,
    p_value,
    spearman_dict,
    write_df,
)


def freedman_lane_y(y: np.ndarray, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.stats import rankdata

    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y2 = y.copy()
    if int(m.sum()) < 12:
        return rng.permutation(y)
    yr = rankdata(y[m]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    b, *_ = np.linalg.lstsq(A, yr, rcond=None)
    y2[m] = (A @ b) + rng.permutation(yr - A @ b)
    return y2


def curve_rhos(xs: dict[int, np.ndarray], y: np.ndarray, Z: np.ndarray | None) -> dict[int, float]:
    out = {}
    for d, x in xs.items():
        rec = associate(x, y, Z)
        out[int(d)] = rec["controlled"] if Z is not None else rec["raw"]
    return out


def curve_p_from_null(obs: dict[int, float], null: dict[int, np.ndarray], n_perm: int) -> float:
    """Within-label curve p: FWER over ranks using max |ρ|."""
    finite = [abs(v) for v in obs.values() if np.isfinite(v)]
    if not finite:
        return float("nan")
    t = max(finite)
    tmax = np.zeros(n_perm)
    for d, arr in null.items():
        tmax = np.maximum(tmax, np.abs(arr))
    return p_value(int(np.sum(tmax >= t)), n_perm)


def studentize(obs: float, null: np.ndarray) -> float:
    mu = float(np.nanmean(null))
    sd = float(np.nanstd(null, ddof=1))
    if not np.isfinite(sd) or sd < 1e-15:
        return float("nan")
    return float((obs - mu) / sd)


def westfall_young_minp(label_curve_p: list[float], label_null_curve_p: np.ndarray) -> dict[str, Any]:
    """label_null_curve_p: (n_perm, n_labels) curve-level p under the synchronized null."""
    obs = float(np.nanmin(label_curve_p)) if label_curve_p else float("nan")
    null_min = np.nanmin(label_null_curve_p, axis=1)
    n = len(null_min)
    ex = int(np.sum(null_min <= obs + 1e-15)) if np.isfinite(obs) else 0
    lo, hi = p_monte_carlo_ci(ex, n)
    return {"p": p_value(ex, n), "p_report": p_report(ex, n), "t_obs_minp": obs, "exceed": ex, "n_perm": n, "ci_lo": lo, "ci_hi": hi}


def build_jobs(root: Path, cfg: AuditConfig, inv: dict[str, Any], *, include_desi: bool) -> list[dict[str, Any]]:
    adcp = cfg.adcp(root)
    panel = pd.read_parquet(adcp / "per_anchor_curvature.parquet")
    jobs = []

    phys = panel[panel.dataset_id == "physics_vit_base"]
    sids = [int(s) for s in sorted(phys.sample_id.unique())]
    ds = sorted(int(d) for d in phys.d.unique())
    agg = phys.groupby(["sample_id", "d"], as_index=False)["K_H_cross"].mean()
    xs = {d: agg[agg.d == d].set_index("sample_id").reindex(sids)["K_H_cross"].to_numpy(float) for d in ds}
    mm = cfg.mm(root)
    probe = load_frozen_probe(mm).set_index("sample_id")
    Z_disc = None
    if all(c in probe.columns for c in SHARED_CORE_CONTROLS):
        Z_disc = control_matrix(probe.reindex(sids).reset_index())

    # discovery local_r2 (reference, not confirmatory)
    y_r2 = probe.reindex(sids)["local_r2"].to_numpy(float)
    jobs.append(
        {
            "dataset_id": DISCOVERY_DATASET,
            "label": "mag_r_desi_local_r2",
            "canonical": "mag_r_desi",
            "group": "physics_vit_base",
            "is_discovery": True,
            "scientific": True,
            "mag_like": True,
            "sids": sids,
            "xs": xs,
            "y": y_r2,
            "Z": Z_disc,
            "ds": [d for d in ds if d in (12, 16, 20) or 8 <= d <= 20],
        }
    )
    for name in ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass", "sfr"):
        y = load_catalog_label(root, name, sids)
        jobs.append(
            {
                "dataset_id": "physics_vit_base",
                "label": name,
                "canonical": name if name != "mag_r_desi" else "mag_r",
                "group": "physics_vit_base",
                "is_discovery": name == "mag_r_desi",
                "scientific": True,
                "mag_like": name == "mag_r_desi",
                "sids": sids,
                "xs": xs,
                "y": y,
                "Z": Z_disc,
                "ds": ds,
            }
        )

    if include_desi:
        desi = panel[panel.dataset_id == "desi_vit_base_hsc"]
        if len(desi):
            dsids = [int(s) for s in sorted(desi.sample_id.unique())]
            dds = sorted(int(d) for d in desi.d.unique())
            dagg = desi.groupby(["sample_id", "d"], as_index=False)["K_H_cross"].mean()
            dxs = {d: dagg[dagg.d == d].set_index("sample_id").reindex(dsids)["K_H_cross"].to_numpy(float) for d in dds}
            lab = np.load(adcp / "cache" / "desi_smith42_labels.npz", allow_pickle=True)
            for key, canon, mag in (("spec_z", "spec_z", False), ("mag_r", "mag_r", True)):
                y = np.asarray([lab[key][s] if 0 <= s < len(lab[key]) else np.nan for s in dsids], dtype=float)
                jobs.append(
                    {
                        "dataset_id": "desi_vit_base_hsc",
                        "label": canon,
                        "canonical": canon,
                        "group": "desi_vit_base_hsc",
                        "is_discovery": False,
                        "scientific": False,
                        "mag_like": mag,
                        "sids": dsids,
                        "xs": dxs,
                        "y": y,
                        "Z": None,
                        "ds": dds,
                    }
                )
    return jobs


def _permute_job(job: dict[str, Any], y: np.ndarray) -> dict[int, float]:
    return curve_rhos(job["xs"], y, job["Z"])


def run_global(root: Path, cfg: AuditConfig, inv: dict[str, Any], *, include_desi: bool, tag: str) -> dict[str, Any]:
    n_perm, _ = cfg.perm_boot()
    jobs = [j for j in build_jobs(root, cfg, inv, include_desi=include_desi) if not j["is_discovery"]]
    if tag == "scientific":
        jobs = [j for j in jobs if j["scientific"]]
    rng = np.random.default_rng(cfg.seed + 101)

    obs = []
    for j in jobs:
        rhos = curve_rhos(j["xs"], j["y"], j["Z"])
        j["obs"] = rhos
        j["tmax"] = max((abs(v) for v in rhos.values() if np.isfinite(v)), default=float("nan"))
        obs.append(j)

    # Observed unstudentized max-|ρ|
    t_obs_raw = max((j["tmax"] for j in jobs if np.isfinite(j["tmax"])), default=float("nan"))

    # Permutation: one index shuffle per group
    groups: dict[str, list[int]] = {}
    for i, j in enumerate(jobs):
        groups.setdefault(j["group"], []).append(i)

    null_rho = [{d: np.empty(n_perm) for d in j["ds"]} for j in jobs]
    null_tmax = np.empty(n_perm)
    null_minp_proxy = np.empty((n_perm, len(jobs)))  # store tmax per label as curve stat
    for b in range(n_perm):
        parts = []
        for grp, idxs in groups.items():
            n = len(jobs[idxs[0]]["y"])
            perm_idx = rng.permutation(n)
            for i in idxs:
                j = jobs[i]
                y0 = j["y"]
                if j["Z"] is not None:
                    yp = freedman_lane_y(y0, j["Z"], rng)
                else:
                    yp = y0[perm_idx]
                rhos = _permute_job(j, yp)
                tmax = max((abs(v) for v in rhos.values() if np.isfinite(v)), default=0.0)
                parts.append(tmax)
                null_minp_proxy[b, i] = tmax
                for d, v in rhos.items():
                    null_rho[i][d][b] = v
        null_tmax[b] = max(parts) if parts else 0.0

    # Old-style global max |ρ|
    ex_raw = int(np.sum(null_tmax >= t_obs_raw)) if np.isfinite(t_obs_raw) else 0
    raw_global = {
        "method": "unstudentized_max_abs_rho",
        "p": p_value(ex_raw, n_perm),
        "p_report": p_report(ex_raw, n_perm),
        "t_obs": t_obs_raw,
        "exceed": ex_raw,
        "n_perm": n_perm,
        **dict(zip(("ci_lo", "ci_hi"), p_monte_carlo_ci(ex_raw, n_perm))),
    }

    # Curve-level p per label from its own tmax null
    curve_p = []
    curve_rows = []
    for i, j in enumerate(jobs):
        tmax_null = null_minp_proxy[:, i]
        ex = int(np.sum(tmax_null >= j["tmax"])) if np.isfinite(j["tmax"]) else 0
        pj = p_value(ex, n_perm)
        curve_p.append(pj)
        j["curve_p"] = pj
        j["curve_p_report"] = p_report(ex, n_perm)
        curve_rows.append(
            {
                "dataset_id": j["dataset_id"],
                "label": j["label"],
                "scientific": j["scientific"],
                "tmax": j["tmax"],
                "curve_p": pj,
                "curve_p_report": p_report(ex, n_perm),
                "n_perm": n_perm,
                "family": tag,
            }
        )

    # WY min-p: convert each perm's per-label tmax into a curve p against the same null
    # p*_b,ℓ = (# of null tmax_ℓ ≥ tmax_b,ℓ + 1) / (B+1) using the same null (conservative, includes itself)
    null_curve_p = np.empty((n_perm, len(jobs)))
    for i, j in enumerate(jobs):
        tmax_null = null_minp_proxy[:, i]
        # rank-based: fraction of null as large as this perm's tmax
        order = np.argsort(-tmax_null, kind="mergesort")
        ranks = np.empty(n_perm)
        ranks[order] = np.arange(1, n_perm + 1)
        null_curve_p[:, i] = ranks / (n_perm + 1)
    wy = westfall_young_minp(curve_p, null_curve_p)
    wy["method"] = "westfall_young_minp"
    wy["family"] = tag
    wy_rows = [{**r, "wy_adjusted_p": wy["p"], "survives_wy": r["curve_p"] <= wy["p"] + 1e-15 and r["curve_p"] <= ALPHA} for r in curve_rows]
    # Standard WY adjusted p for each label: share of perms with min_p* <= p_ℓ
    for r, pj in zip(wy_rows, curve_p):
        ex = int(np.sum(np.nanmin(null_curve_p, axis=1) <= pj + 1e-15))
        r["wy_adjusted_p"] = p_value(ex, n_perm)
        r["wy_adjusted_p_report"] = p_report(ex, n_perm)
        r["survives_wy"] = r["wy_adjusted_p"] <= ALPHA

    # Studentized max-T
    t_obs_parts = []
    st_rows = []
    null_maxT = np.empty(n_perm)
    cell_T_null = []
    for i, j in enumerate(jobs):
        for d in j["ds"]:
            arr = null_rho[i][d]
            T = studentize(j["obs"][d], arr)
            t_obs_parts.append(abs(T) if np.isfinite(T) else 0.0)
            Tn = (arr - np.nanmean(arr)) / max(float(np.nanstd(arr, ddof=1)), 1e-15)
            cell_T_null.append(np.abs(Tn))
            st_rows.append(
                {
                    "dataset_id": j["dataset_id"],
                    "label": j["label"],
                    "d": int(d),
                    "rho": j["obs"][d],
                    "null_mean": float(np.nanmean(arr)),
                    "null_sd": float(np.nanstd(arr, ddof=1)),
                    "T": T,
                    "scientific": j["scientific"],
                    "family": tag,
                }
            )
    t_obs_T = max(t_obs_parts) if t_obs_parts else float("nan")
    stacked = np.vstack(cell_T_null) if cell_T_null else np.zeros((1, n_perm))
    null_maxT = stacked.max(axis=0)
    ex_T = int(np.sum(null_maxT >= t_obs_T)) if np.isfinite(t_obs_T) else 0
    maxT = {
        "method": "studentized_maxT",
        "p": p_value(ex_T, n_perm),
        "p_report": p_report(ex_T, n_perm),
        "t_obs": t_obs_T,
        "exceed": ex_T,
        "n_perm": n_perm,
        "family": tag,
        **dict(zip(("ci_lo", "ci_hi"), p_monte_carlo_ci(ex_T, n_perm))),
    }
    for r in st_rows:
        r["maxT_global_p"] = maxT["p"]
        r["survives_maxT"] = bool(np.isfinite(r["T"]) and abs(r["T"]) >= t_obs_T - 1e-12 and maxT["p"] <= ALPHA)

    return {
        "jobs": jobs,
        "raw_global": raw_global,
        "wy": wy,
        "maxT": maxT,
        "wy_table": pd.DataFrame(wy_rows),
        "maxT_table": pd.DataFrame(st_rows),
        "n_perm": n_perm,
    }


def run_transitions(root: Path, cfg: AuditConfig, controls: dict[str, Any], sizes: pd.DataFrame) -> pd.DataFrame:
    out = cfg.resolved(root)
    adcp = cfg.adcp(root)
    rank = pd.read_csv(adcp / "dataset_rank_associations.csv") if (adcp / "dataset_rank_associations.csv").exists() else pd.DataFrame()
    ranges = pd.read_csv(adcp / "geometry_dimension_ranges.csv") if (adcp / "geometry_dimension_ranges.csv").exists() else pd.DataFrame()
    rows = []

    # Frozen discovery contrast from the frozen curve (local_r2 + frozen controls)
    rows.append(
        {
            "hypothesis": "magnitude_transition_replication",
            "curve": "frozen_discovery_local_r2_controlled",
            "dataset_id": DISCOVERY_DATASET,
            "label": DISCOVERY_LABEL,
            "d_80": FROZEN_D80,
            "d_85": FROZEN_D85,
            "rho_80": FROZEN_CTL[FROZEN_D80],
            "rho_85": FROZEN_CTL[FROZEN_D85],
            "delta_85_80": FROZEN_CTL[FROZEN_D85] - FROZEN_CTL[FROZEN_D80],
            "predicted_sign": "negative",
            "sign_match": (FROZEN_CTL[FROZEN_D85] - FROZEN_CTL[FROZEN_D80]) < 0,
            "n_independent_replications": 0,
            "note": "discovery reference; not a replication. DESI excluded (alignment unproven).",
        }
    )
    rows.append(
        {
            "hypothesis": "magnitude_transition_replication",
            "curve": "harmonized_catalog_controlled",
            "dataset_id": DISCOVERY_DATASET,
            "label": "catalog_mag_r_desi",
            "d_80": FROZEN_D80,
            "d_85": FROZEN_D85,
            "rho_80": float(controls["side"].loc[controls["side"].d == 12, "harmonized_control_catalog_mag"].iloc[0])
            if len(controls["side"])
            else np.nan,
            "rho_85": float(controls["side"].loc[controls["side"].d == 20, "harmonized_control_catalog_mag"].iloc[0])
            if len(controls["side"])
            else np.nan,
            "delta_85_80": controls["delta_harmonized_ctl"],
            "predicted_sign": "negative",
            "sign_match": bool(np.isfinite(controls["delta_harmonized_ctl"]) and controls["delta_harmonized_ctl"] < 0),
            "n_independent_replications": 0,
            "note": "not discovery parity; different y (catalog mag). No independent mag replication.",
        }
    )

    # Any-association: filled after global tests in stages
    rows.append(
        {
            "hypothesis": "any_association",
            "curve": "confirmatory_physics_catalog_labels",
            "dataset_id": "physics_vit_base",
            "label": "family",
            "d_80": "",
            "d_85": "",
            "rho_80": np.nan,
            "rho_85": np.nan,
            "delta_85_80": np.nan,
            "predicted_sign": "not_assumed",
            "sign_match": "",
            "n_independent_replications": "",
            "note": "see global_minp_results / global_studentized_maxT_results; not a common-transition test",
        }
    )

    # Exploratory redshift comparison
    if len(rank):
        phys_z = rank[(rank.dataset_id == "physics_vit_base") & (rank.label == "photo_z")]
        desi_z = rank[(rank.dataset_id == "desi_vit_base_hsc") & (rank.label == "spec_z")]
        rows.append(
            {
                "hypothesis": "exploratory_redshift",
                "curve": "photo_z_vs_spec_z_post_hoc",
                "dataset_id": "both",
                "label": "redshift",
                "d_80": "",
                "d_85": "",
                "rho_80": float(phys_z.loc[phys_z.d == 12, "controlled"].iloc[0]) if len(phys_z[phys_z.d == 12]) else np.nan,
                "rho_85": float(desi_z.loc[desi_z.d == 17, "controlled"].iloc[0]) if len(desi_z[desi_z.d == 17]) else np.nan,
                "delta_85_80": np.nan,
                "predicted_sign": "not_preregistered",
                "sign_match": "",
                "n_independent_replications": 0,
                "note": "post hoc; DESI spec_z not a scientific result (alignment unresolved). No signed meta-analysis.",
            }
        )

    df = pd.DataFrame(rows)
    write_df(out / "transition_specific_results.csv", df, force=cfg.force)
    return df
