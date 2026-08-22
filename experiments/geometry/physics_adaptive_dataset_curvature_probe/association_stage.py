"""Label associations after geometry freeze. Discovery excluded from confirmatory aggregates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.curvature_probe_screen import partial_spearman, spearman_dict
from geometry.physics_activation_atlas.paths import resolve_path

from .config import (
    DISCOVERY_DATASET,
    DISCOVERY_LABEL,
    SHARED_CORE_CONTROLS,
    SOURCE_MM,
    VAR_COMPARE_GRID,
)
from .pipeline import AdaptiveProbeConfig, crossing_d, p_value, write_df, write_json


def associate(x: np.ndarray, y: np.ndarray, Z: np.ndarray | None) -> dict[str, float]:
    raw = spearman_dict(x, y)
    if Z is None:
        return {"raw": raw["rho"], "controlled": float("nan"), "n": raw["n"], "p_raw": raw["pvalue"]}
    ctl = partial_spearman(x, y, Z)
    return {"raw": raw["rho"], "controlled": ctl["rho"], "n": raw["n"], "p_raw": raw["pvalue"], "p_ctl": ctl["pvalue"]}


def local_confounders(
    y: np.ndarray,
    neigh: np.ndarray,
    sids: list[int],
    sid_to_row: dict[int, int],
    X: np.ndarray,
    k: int,
) -> pd.DataFrame:
    rows = []
    for ai, sid in enumerate(sids):
        row = sid_to_row[int(sid)]
        N = neigh[ai, :k]
        yn = y[N] if y.size == X.shape[0] else y[[sid_to_row.get(int(s), int(s)) for s in N]]
        # y is aligned to X rows
        yn = y[N]
        m = np.isfinite(yn)
        x0 = X[row]
        xn = X[N]
        # chord radius of the k-th neighbour
        d = np.linalg.norm(xn - x0[None, :], axis=1)
        rho = float(np.max(d)) if len(d) else np.nan
        rows.append(
            {
                "sample_id": int(sid),
                "log_knn_radius": float(np.log(max(rho, 1e-12))),
                "local_label_variance": float(np.var(yn[m])) if int(m.sum()) >= 2 else float("nan"),
                "local_evaluation_count": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def physics_probe_confounders(root: Path, cfg: AdaptiveProbeConfig, sids: list[int], k: int) -> pd.DataFrame | None:
    p = cfg.mm(root) / "local_probe_fields.parquet"
    if not p.exists():
        return None
    geo = pd.read_parquet(p)
    geo = geo[(geo.model == "vit_base") & (geo.neighbourhood == "model")]
    if "scale_k" in geo.columns:
        geo = geo[geo.scale_k == k]
    geo = geo[geo.sample_id.isin(sids)]
    keep = ["sample_id"] + [c for c in SHARED_CORE_CONTROLS if c in geo.columns]
    return geo[keep].drop_duplicates("sample_id")


def load_physics_labels(root: Path) -> dict[str, np.ndarray]:
    z = np.load(root / "data_hf/physics/vit_base_test_labels.npz")
    out = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    if "stellar_mass" in out:
        y = out["stellar_mass"].copy()
        y[y == -99.0] = np.nan
        out["stellar_mass"] = y
    return out


def load_desi_labels(out: Path) -> dict[str, np.ndarray]:
    z = np.load(out / "cache" / "desi_smith42_labels.npz", allow_pickle=True)
    return {
        "spec_z": np.asarray(z["spec_z"], dtype=np.float64),
        "mag_r": np.asarray(z["mag_r"], dtype=np.float64),
    }


def label_on_anchors(y_full: np.ndarray, sids: list[int], sid_to_full: dict[int, int] | None) -> np.ndarray:
    if sid_to_full is None:
        return y_full[np.asarray(sids, dtype=np.int64)]
    return np.asarray([y_full[sid_to_full[int(s)]] for s in sids], dtype=np.float64)


def wide_from_panel(panel: pd.DataFrame, ds: list[int], y: np.ndarray, sids: list[int], conf: pd.DataFrame, xcol: str = "K_H_cross") -> pd.DataFrame:
    agg = panel.groupby(["sample_id", "d"], as_index=False)[xcol].mean()
    w = pd.DataFrame({"sample_id": sids, "y": y})
    for d in ds:
        g = agg[agg.d == d][["sample_id", xcol]].rename(columns={xcol: f"KH{d}"})
        w = w.merge(g, on="sample_id", how="left")
    w = w.merge(conf, on="sample_id", how="left")
    return w


def control_matrix(df: pd.DataFrame) -> np.ndarray | None:
    if not all(c in df.columns for c in SHARED_CORE_CONTROLS):
        return None
    return np.column_stack([df[c].fillna(0).to_numpy(float) for c in SHARED_CORE_CONTROLS])


def permute_once(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(y)


def freedman_lane_y(y: np.ndarray, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from scipy.stats import rankdata

    m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y2 = y.copy()
    if int(m.sum()) < 12:
        return permute_once(y, rng)
    yr = rankdata(y[m]).astype(np.float64)
    Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
    A = np.column_stack([np.ones(int(m.sum())), Zr])
    b, *_ = np.linalg.lstsq(A, yr, rcond=None)
    fit = A @ b
    y2[m] = fit + rng.permutation(yr - fit)
    return y2


def permutation_block(
    wide: pd.DataFrame,
    ds: list[int],
    *,
    n_perm: int,
    seed: int,
    controlled: bool,
) -> dict[str, Any]:
    y = wide["y"].to_numpy(float)
    Z = control_matrix(wide)
    xs = {d: wide[f"KH{d}"].to_numpy(float) for d in ds}
    obs = {d: associate(xs[d], y, Z if controlled else None) for d in ds}
    rng = np.random.default_rng(seed)
    null = {d: np.empty(n_perm) for d in ds}
    tmax = np.empty(n_perm)
    for b in range(n_perm):
        yp = freedman_lane_y(y, Z, rng) if controlled and Z is not None else permute_once(y, rng)
        rhos = []
        for d in ds:
            r = associate(xs[d], yp, Z if controlled else None)
            val = r["controlled"] if controlled else r["raw"]
            null[d][b] = val
            rhos.append(abs(val) if np.isfinite(val) else 0.0)
        tmax[b] = float(np.max(rhos)) if rhos else np.nan
    finite = [abs(obs[d]["controlled" if controlled else "raw"]) for d in ds if np.isfinite(obs[d]["controlled" if controlled else "raw"])]
    t_obs = max(finite) if finite else float("nan")
    rows = []
    for d in ds:
        val = obs[d]["controlled" if controlled else "raw"]
        ex_pt = int(np.sum(np.abs(null[d]) >= abs(val))) if np.isfinite(val) else 0
        ex_fw = int(np.sum(tmax >= abs(val))) if np.isfinite(val) else 0
        rows.append(
            {
                "d": int(d),
                "kind": "controlled" if controlled else "raw",
                "rho": val,
                "p_pointwise": p_value(ex_pt, n_perm) if np.isfinite(val) else float("nan"),
                "p_fwer": p_value(ex_fw, n_perm) if np.isfinite(val) else float("nan"),
                "n": obs[d]["n"],
                "exceed_pointwise": ex_pt,
                "exceed_fwer": ex_fw,
            }
        )
    return {
        "obs": obs,
        "table": pd.DataFrame(rows),
        "tmax_obs": t_obs,
        "p_global": p_value(int(np.sum(tmax >= t_obs)), n_perm) if np.isfinite(t_obs) else float("nan"),
        "null_envelope": {d: float(np.quantile(np.abs(null[d]), 0.95)) for d in ds},
        "tmax": tmax,
        "null": null,
    }


def bootstrap_block(wide: pd.DataFrame, ds: list[int], n_boot: int, seed: int) -> pd.DataFrame:
    n = len(wide)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    y = wide["y"].to_numpy(float)
    Z = control_matrix(wide)
    xs = {d: wide[f"KH{d}"].to_numpy(float) for d in ds}
    raw_b = np.full((n_boot, len(ds)), np.nan)
    ctl_b = np.full((n_boot, len(ds)), np.nan)
    for b in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        for j, d in enumerate(ds):
            rec = associate(xs[d][take], y[take], Z[take] if Z is not None else None)
            raw_b[b, j] = rec["raw"]
            ctl_b[b, j] = rec["controlled"]
    rows = []
    for j, d in enumerate(ds):
        obs = associate(xs[d], y, Z)
        rows.append(
            {
                "d": int(d),
                "raw": obs["raw"],
                "raw_lo": float(np.nanquantile(raw_b[:, j], 0.025)),
                "raw_hi": float(np.nanquantile(raw_b[:, j], 0.975)),
                "controlled": obs["controlled"],
                "ctl_lo": float(np.nanquantile(ctl_b[:, j], 0.025)),
                "ctl_hi": float(np.nanquantile(ctl_b[:, j], 0.975)),
            }
        )
    return pd.DataFrame(rows)


def interpolate_rho_on_tau(ds: list[int], r2: np.ndarray, rho: np.ndarray, grid=VAR_COMPARE_GRID) -> pd.DataFrame:
    """Interpolate ρ vs τ only between observed reliable ranks. No extrapolation."""
    rows = []
    order = np.argsort(ds)
    d_s = np.asarray(ds, dtype=float)[order]
    t_s = np.asarray(r2, dtype=float)[order]
    r_s = np.asarray(rho, dtype=float)[order]
    m = np.isfinite(t_s) & np.isfinite(r_s)
    d_s, t_s, r_s = d_s[m], t_s[m], r_s[m]
    if len(t_s) < 2:
        return pd.DataFrame(rows)
    tmin, tmax = float(t_s.min()), float(t_s.max())
    for tau in grid:
        if tau < tmin - 1e-12 or tau > tmax + 1e-12:
            rows.append({"tau": float(tau), "rho": np.nan, "interpolated": False, "in_range": False})
            continue
        rows.append(
            {
                "tau": float(tau),
                "rho": float(np.interp(tau, t_s, r_s)),
                "interpolated": True,
                "in_range": True,
            }
        )
    return pd.DataFrame(rows)


def delta_85_80(crossings: dict, rho_by_d: dict[int, float]) -> float:
    d80, d85 = crossings.get("d_80"), crossings.get("d_85")
    if not isinstance(d80, int) or not isinstance(d85, int):
        return float("nan")
    a, b = rho_by_d.get(d80), rho_by_d.get(d85)
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(b - a)


def run_associations(
    root: Path,
    cfg: AdaptiveProbeConfig,
    *,
    jobs: list[dict[str, Any]],
    n_perm: int,
    n_boot: int,
) -> dict[str, Any]:
    """jobs: dataset_id, label, canonical, is_discovery, mag_like, panel, sids, y, conf, ds, k, crossings, r2_by_d, group."""
    out = cfg.resolved(root)
    rank_rows = []
    var_rows = []
    perm_rows = []
    boot_rows = []
    contrast_rows = []
    group_perms: dict[str, list[dict[str, Any]]] = {}

    for job in jobs:
        ds = job["ds"]
        wide = wide_from_panel(job["panel"], ds, job["y"], job["sids"], job["conf"])
        Z = control_matrix(wide)
        missing = [c for c in SHARED_CORE_CONTROLS if c not in wide.columns]
        job["missing_controls"] = missing
        raw_p = permutation_block(wide, ds, n_perm=n_perm, seed=cfg.seed + 11, controlled=False)
        ctl_p = permutation_block(wide, ds, n_perm=n_perm, seed=cfg.seed + 13, controlled=True)
        boot = bootstrap_block(wide, ds, n_boot, cfg.seed + 17)
        r2_map = job["r2_by_d"]
        rho_raw = {}
        rho_ctl = {}
        for d in ds:
            rec = associate(wide[f"KH{d}"].to_numpy(float), wide["y"].to_numpy(float), Z)
            rho_raw[d] = rec["raw"]
            rho_ctl[d] = rec["controlled"]
            pr = raw_p["table"][raw_p["table"].d == d].iloc[0]
            pc = ctl_p["table"][ctl_p["table"].d == d].iloc[0]
            br = boot[boot.d == d].iloc[0]
            rank_rows.append(
                {
                    "dataset_id": job["dataset_id"],
                    "label": job["label"],
                    "canonical_label": job["canonical"],
                    "is_discovery": job["is_discovery"],
                    "d": int(d),
                    "k": job["k"],
                    "raw": rec["raw"],
                    "controlled": rec["controlled"],
                    "n": rec["n"],
                    "p_raw_pointwise": pr["p_pointwise"],
                    "p_raw_fwer": pr["p_fwer"],
                    "p_ctl_pointwise": pc["p_pointwise"],
                    "p_ctl_fwer": pc["p_fwer"],
                    "raw_lo": br["raw_lo"],
                    "raw_hi": br["raw_hi"],
                    "ctl_lo": br["ctl_lo"],
                    "ctl_hi": br["ctl_hi"],
                    "r2_L": r2_map.get(d, float("nan")),
                    "missing_controls": ",".join(missing),
                }
            )
        for kind, rhos in (("raw", rho_raw), ("controlled", rho_ctl)):
            r2s = np.array([r2_map.get(d, np.nan) for d in ds])
            rh = np.array([rhos[d] for d in ds])
            idf = interpolate_rho_on_tau(ds, r2s, rh)
            for _, r in idf.iterrows():
                var_rows.append(
                    {
                        "dataset_id": job["dataset_id"],
                        "label": job["label"],
                        "canonical_label": job["canonical"],
                        "is_discovery": job["is_discovery"],
                        "kind": kind,
                        "tau": r.tau,
                        "rho": r.rho,
                        "in_range": bool(r.in_range),
                    }
                )
        perm_rows.append(
            {
                "dataset_id": job["dataset_id"],
                "label": job["label"],
                "canonical_label": job["canonical"],
                "is_discovery": job["is_discovery"],
                "p_global_raw": raw_p["p_global"],
                "p_global_ctl": ctl_p["p_global"],
                "tmax_raw": raw_p["tmax_obs"],
                "tmax_ctl": ctl_p["tmax_obs"],
                "n_perm": n_perm,
            }
        )
        for _, r in boot.iterrows():
            boot_rows.append({"dataset_id": job["dataset_id"], "label": job["label"], "d": int(r.d), **r.to_dict()})
        dlt_raw = delta_85_80(job["crossings"], rho_raw)
        dlt_ctl = delta_85_80(job["crossings"], rho_ctl)
        contrast_rows.append(
            {
                "dataset_id": job["dataset_id"],
                "label": job["label"],
                "canonical_label": job["canonical"],
                "is_discovery": job["is_discovery"],
                "mag_like": job["mag_like"],
                "d_80": job["crossings"].get("d_80"),
                "d_85": job["crossings"].get("d_85"),
                "delta_85_80_raw": dlt_raw,
                "delta_85_80_ctl": dlt_ctl,
                "predicted_sign": "negative" if job["mag_like"] else "not_assumed",
                "sign_consistent_raw": bool(np.isfinite(dlt_raw) and dlt_raw < 0) if job["mag_like"] else "",
            }
        )
        if not job["is_discovery"]:
            group_perms.setdefault(job["group"], []).append(
                {"job": job, "wide": wide, "ds": ds, "Z": Z, "ctl_null_tmax": ctl_p["tmax"], "tmax_obs": ctl_p["tmax_obs"]}
            )

    # Global max over confirmatory family. Same-object groups share one permutation.
    rng = np.random.default_rng(cfg.seed + 101)
    family = [j for j in jobs if not j["is_discovery"]]
    t_obs_parts = []
    for j in family:
        hit = [r for r in perm_rows if r["dataset_id"] == j["dataset_id"] and r["label"] == j["label"]]
        if hit and np.isfinite(hit[0]["tmax_ctl"]):
            t_obs_parts.append(abs(hit[0]["tmax_ctl"]))
    t_obs = max(t_obs_parts) if t_obs_parts else float("nan")
    tmax_g = np.empty(n_perm)
    for b in range(n_perm):
        parts = []
        for _grp, items in group_perms.items():
            # one shared permutation of objects inside the group
            if not items:
                continue
            y0 = items[0]["wide"]["y"].to_numpy(float)
            # different labels: permute indices once, apply to each label vector
            perm_idx = rng.permutation(len(y0))
            for it in items:
                wide = it["wide"]
                y = wide["y"].to_numpy(float)[perm_idx]
                Z = it["Z"]
                rhos = []
                for d in it["ds"]:
                    rec = associate(wide[f"KH{d}"].to_numpy(float), y, Z)
                    val = rec["controlled"]
                    rhos.append(abs(val) if np.isfinite(val) else 0.0)
                parts.append(max(rhos) if rhos else 0.0)
        tmax_g[b] = max(parts) if parts else 0.0
    p_global = p_value(int(np.sum(tmax_g >= t_obs)), n_perm) if np.isfinite(t_obs) else float("nan")

    # leave-one-dataset-out on mag-like Δ85-80
    mag = [r for r in contrast_rows if r["mag_like"] and not r["is_discovery"] and np.isfinite(r.get("delta_85_80_ctl", np.nan) if isinstance(r.get("delta_85_80_ctl"), float) else np.nan)]
    lodo = []
    ds_ids = sorted({r["dataset_id"] for r in mag})
    for leave in ds_ids or ["none"]:
        keep = [r["delta_85_80_ctl"] for r in mag if r["dataset_id"] != leave]
        lodo.append(
            {
                "left_out": leave,
                "n": len(keep),
                "median_delta": float(np.median(keep)) if keep else float("nan"),
                "frac_negative": float(np.mean(np.array(keep) < 0)) if keep else float("nan"),
            }
        )
    if not ds_ids:
        keep = [r["delta_85_80_ctl"] for r in contrast_rows if r["mag_like"] and np.isfinite(r.get("delta_85_80_ctl", np.nan) if isinstance(r.get("delta_85_80_ctl"), (int, float)) else np.nan)]
        lodo.append({"left_out": "none_confirmatory_mag", "n": len(keep), "median_delta": float(np.median(keep)) if keep else float("nan"), "frac_negative": float(np.mean(np.asarray(keep) < 0)) if keep else float("nan")})

    write_df(out / "dataset_rank_associations.csv", pd.DataFrame(rank_rows), force=cfg.force)
    write_df(out / "dataset_variance_associations.csv", pd.DataFrame(var_rows), force=cfg.force)
    write_df(out / "dataset_permutation_results.csv", pd.DataFrame(perm_rows), force=cfg.force)
    write_df(out / "bootstrap_results.csv", pd.DataFrame(boot_rows), force=cfg.force)
    write_df(out / "replication_contrasts.csv", pd.DataFrame(contrast_rows), force=cfg.force)
    write_df(out / "leave_one_dataset_out.csv", pd.DataFrame(lodo), force=cfg.force)
    write_json(
        out / "global_permutation_results.csv".replace(".csv", ".json"),
        {
            "n_perm": n_perm,
            "t_obs_ctl_max": t_obs,
            "p_global_ctl": p_global,
            "family": [{"dataset_id": j["dataset_id"], "label": j["label"]} for j in family],
            "discovery_excluded": {"dataset_id": DISCOVERY_DATASET, "label": DISCOVERY_LABEL},
            "dependence": "same-object groups share one object permutation; independent datasets contribute separately to the joint max",
        },
        force=cfg.force,
    )
    # also csv one-row
    write_df(
        out / "global_permutation_results.csv",
        pd.DataFrame(
            [
                {
                    "n_perm": n_perm,
                    "t_obs_ctl_max": t_obs,
                    "p_global_ctl": p_global,
                    "n_family": len(family),
                    "discovery_excluded": f"{DISCOVERY_DATASET}:{DISCOVERY_LABEL}",
                }
            ]
        ),
        force=cfg.force,
    )
    return {
        "rank": pd.DataFrame(rank_rows),
        "var": pd.DataFrame(var_rows),
        "perm": pd.DataFrame(perm_rows),
        "contrasts": pd.DataFrame(contrast_rows),
        "lodo": pd.DataFrame(lodo),
        "p_global": p_global,
        "t_obs": t_obs,
        "n_family": len(family),
    }
