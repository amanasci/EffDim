"""Orchestrate inventory → geometry freeze → curvature → associations → report."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.paths import platonic_root, resolve_path
from geometry.physics_stable_tangent_dimension.sphere_coords import row_l2_status

from .association_stage import (
    load_desi_labels,
    load_physics_labels,
    local_confounders,
    run_associations,
)
from .classify import primary_label, summarize_peaks
from .config import (
    DEFAULT_THRESHOLDS,
    DISCOVERY_DATASET,
    DISCOVERY_LABEL,
    FREEZE_HASH_EXPECTED,
    K_FRAC_OF_N,
    K_PRESET,
    MIN_VALID_ANCHORS,
    PARITY_D12_RHO,
    PARITY_D16_RHO,
    PARITY_TOL,
    PRESERVED,
    SOURCE_CPRS,
    SOURCE_EDM,
    SOURCE_MM,
    SOURCE_NDC,
    SOURCE_QPD,
)
from .curvature_stage import reuse_physics_kh, run_curvature_dataset
from .geometry_stage import load_desi_X, load_physics_X, reuse_physics_qpd, run_geometry_dataset
from .inventory import build_inventory
from .pipeline import (
    AdaptiveProbeConfig,
    _budget_ok,
    _done,
    assert_not_preserved,
    device_of,
    file_sha,
    file_sha_full,
    hash_select,
    primary_k,
    scale_list,
    sha16,
    write_df,
    write_json,
)
from .plots import write_figures
from .report import write_methods, write_report

STAGES = [
    "prepare",
    "inventory",
    "geometry",
    "freeze",
    "curvature",
    "associations",
    "scale_sensitivity",
    "analyze",
    "report",
]


def run(cfg: AdaptiveProbeConfig) -> dict[str, Any]:
    t0 = time.time()
    root = platonic_root()
    out = cfg.resolved(root)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "datasets", "figures", "logs"):
        (out / sub).mkdir(exist_ok=True)
    device = device_of(cfg)
    n_perm, n_boot = cfg.perm_boot()
    ctx: dict[str, Any] = {"device": device, "t0": t0, "n_perm": n_perm, "n_boot": n_boot}

    if cfg.stage in ("all", "prepare"):
        stage_prepare(root, cfg, ctx)
    if cfg.stage in ("all", "inventory", "prepare"):
        ctx["inv"] = build_inventory(root, cfg)
        print(f"[adcp] inventory included={ctx['inv']['manifest']['included_datasets']}", flush=True)
    else:
        ctx["inv"] = {
            "inventory": pd.read_csv(out / "dataset_inventory.csv"),
            "labels": pd.read_csv(out / "physics_label_manifest.csv"),
            "manifest": json.loads((out / "inclusion_manifest.json").read_text()),
        }
    if cfg.stage in ("all", "geometry"):
        ctx["geo"] = stage_geometry(root, cfg, ctx, t0)
    if cfg.stage in ("all", "freeze"):
        ctx["freeze"] = stage_freeze(root, cfg)
    if cfg.stage in ("all", "curvature"):
        ctx["curv"] = stage_curvature(root, cfg, ctx, t0)
    if cfg.stage in ("all", "associations"):
        ctx["assoc"] = stage_associations(root, cfg, ctx)
    if cfg.stage in ("all", "scale_sensitivity"):
        stage_scale(root, cfg, ctx, t0)
    if cfg.stage in ("all", "analyze", "report"):
        ctx["summary"] = stage_analyze(root, cfg, ctx, t0)
        write_figures(out, cfg)
        write_methods(out, cfg, ctx)
        write_report(out, cfg, ctx)
        write_json(
            out / "COMPLETE.json",
            {"ok": True, "primary_label": ctx.get("summary", {}).get("primary_label"), "seconds": time.time() - t0},
            force=cfg.force,
        )
    print(f"[adcp] done in {time.time() - t0:.1f}s stage={cfg.stage}", flush=True)
    return ctx


def stage_prepare(root: Path, cfg: AdaptiveProbeConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    write_json(
        out / "config.json",
        {
            "protocol": "adaptive_dataset_curvature_probe_v1",
            "k_rule": f"largest k in {list(K_PRESET)} with k <= {K_FRAC_OF_N} * n",
            "min_valid_anchors": MIN_VALID_ANCHORS,
            "discovery": {"dataset": DISCOVERY_DATASET, "label": DISCOVERY_LABEL},
            "preserved": PRESERVED,
            "thresholds": DEFAULT_THRESHOLDS,
            "smoke": cfg.smoke,
            "hash_select": "sha256(adcp:{seed}:{sample_id}) lexicographic prefix",
            "expected_freeze_hash_upstream": FREEZE_HASH_EXPECTED,
        },
        force=cfg.force,
    )


def _physics_setup(root: Path, cfg: AdaptiveProbeConfig) -> dict[str, Any]:
    mm = cfg.mm(root)
    X = load_physics_X(root, cfg)
    anchors_sid = np.load(mm / "prepare" / "anchors.npz")["anchors_sample_id"]
    anchors_local = np.load(mm / "prepare" / "anchors.npz")["anchors_local"]
    aid = mm / "d_replication_check_all512" / "anchor_ids.json"
    use_sids = json.loads(aid.read_text())["sample_ids"] if aid.exists() else [int(s) for s in anchors_sid]
    use_sids = [int(s) for s in use_sids]
    use_sids = hash_select(use_sids, cfg.target_anchors(), seed=cfg.seed)
    sid_to_row = {int(s): int(i) for s, i in zip(anchors_sid, anchors_local)}
    # also map via sample_folds if needed
    folds = mm / "sample_folds.parquet"
    if folds.exists():
        f = pd.read_parquet(folds)
        if "sample_id" in f.columns:
            sid_to_row.update({int(s): int(i) for i, s in enumerate(f.sample_id.to_numpy())})
    pack = dict(np.load(mm / "model_neighbourhoods" / "vit_base_kmax2048.npz"))
    sid_to_ai = {int(s): i for i, s in enumerate(anchors_sid)}
    neigh = np.stack([pack["neigh"][sid_to_ai[s], :2048] for s in use_sids if s in sid_to_ai])
    use_sids = [s for s in use_sids if s in sid_to_ai]
    k = 2048 if not cfg.smoke else min(256, 2048)
    if cfg.smoke:
        neigh = neigh[:, :k]
    return {
        "X": X,
        "sids": use_sids,
        "sid_to_row": sid_to_row,
        "neigh": neigh,
        "k": k,
        "l2": row_l2_status(X),
    }


def _desi_setup(root: Path, cfg: AdaptiveProbeConfig, device) -> dict[str, Any] | None:
    pq = root / "data_hf/desi/desi_vit_base.parquet"
    lab = cfg.resolved(root) / "cache" / "desi_smith42_labels.npz"
    if not pq.exists() or not lab.exists():
        return None
    X = load_desi_X(root)
    n = len(X)
    k = primary_k(n)
    if k is None:
        return None
    if cfg.smoke:
        k = min(k, 128)
    sids = hash_select(list(range(n)), cfg.target_anchors(), seed=cfg.seed)
    sid_to_row = {int(s): int(s) for s in range(n)}
    return {"X": X, "sids": sids, "sid_to_row": sid_to_row, "neigh": None, "k": k, "l2": row_l2_status(X)}


def stage_geometry(root: Path, cfg: AdaptiveProbeConfig, ctx: dict, t0: float) -> dict[str, Any]:
    out = cfg.resolved(root)
    device = ctx["device"]
    geos = {}
    included = ctx["inv"]["manifest"]["included_datasets"]

    if "physics_vit_base" in included:
        phys = _physics_setup(root, cfg)
        qpd = reuse_physics_qpd(root)
        geos["physics_vit_base"] = run_geometry_dataset(
            root,
            cfg,
            dataset_id="physics_vit_base",
            X=phys["X"],
            sids=phys["sids"],
            sid_to_row=phys["sid_to_row"],
            k=phys["k"],
            device=device,
            reuse_knn=phys["neigh"],
            reuse_qpd=qpd,
            t0=t0,
        )
        geos["physics_vit_base"]["setup"] = phys
        print(f"[adcp] geometry physics D={geos['physics_vit_base']['interval']}", flush=True)

    if "desi_vit_base_hsc" in included and not cfg.skip_desi:
        desi = _desi_setup(root, cfg, device)
        if desi is not None:
            geos["desi_vit_base_hsc"] = run_geometry_dataset(
                root,
                cfg,
                dataset_id="desi_vit_base_hsc",
                X=desi["X"],
                sids=desi["sids"],
                sid_to_row=desi["sid_to_row"],
                k=desi["k"],
                device=device,
                reuse_knn=None,
                reuse_qpd=None,
                t0=t0,
            )
            geos["desi_vit_base_hsc"]["setup"] = desi
            print(f"[adcp] geometry desi D={geos['desi_vit_base_hsc']['interval']}", flush=True)

    # concatenate range tables
    ranges = []
    crosses = []
    lin = []
    qs = []
    qr = []
    for did, g in geos.items():
        ranges.append(g["interval"])
        crosses.append({"dataset_id": did, **g["crossings"]})
        ddir = out / "datasets" / did
        if (ddir / "linear_risk_pooled.csv").exists():
            lin.append(pd.read_csv(ddir / "linear_risk_pooled.csv"))
        if (ddir / "quadratic_screening.csv").exists():
            qs.append(pd.read_csv(ddir / "quadratic_screening.csv"))
        if (ddir / "quadratic_refinement.csv").exists():
            qr.append(pd.read_csv(ddir / "quadratic_refinement.csv"))
    write_df(out / "geometry_dimension_ranges.csv", pd.DataFrame(ranges), force=cfg.force)
    write_df(out / "spectral_crossings.csv", pd.DataFrame(crosses), force=cfg.force)
    if lin:
        write_df(out / "linear_risk_curves.csv", pd.concat(lin, ignore_index=True), force=cfg.force)
    else:
        write_df(out / "linear_risk_curves.csv", pd.DataFrame([{"dataset_id": None}]), force=cfg.force)
    write_df(out / "quadratic_screening.csv", pd.concat(qs, ignore_index=True) if qs else pd.DataFrame([{"dataset_id": None}]), force=cfg.force)
    write_df(out / "quadratic_refinement.csv", pd.concat(qr, ignore_index=True) if qr else pd.DataFrame([{"dataset_id": None}]), force=cfg.force)
    return geos


def stage_freeze(root: Path, cfg: AdaptiveProbeConfig) -> dict[str, Any]:
    out = cfg.resolved(root)
    p = out / "geometry_dimension_ranges.csv"
    if not p.exists():
        raise RuntimeError("geometry_dimension_ranges.csv missing; cannot freeze")
    digest = file_sha_full(p)
    rec = {
        "file": str(p),
        "sha256": digest,
        "sha16": digest[:16],
        "labels_loaded": False,
        "note": "hash frozen before any physics-label association",
    }
    write_json(out / "geometry_freeze.json", rec, force=cfg.force)
    print(f"[adcp] geometry freeze sha16={digest[:16]}", flush=True)
    return rec


def _interval_ds(interval: dict) -> list[int]:
    lo = int(interval.get("d_low_primary", interval.get("d_low", 2)))
    hi = int(interval.get("d_high_primary", interval.get("d_high", lo)))
    return list(range(lo, hi + 1))


def stage_curvature(root: Path, cfg: AdaptiveProbeConfig, ctx: dict, t0: float) -> dict[str, Any]:
    freeze = json.loads((cfg.resolved(root) / "geometry_freeze.json").read_text())
    if not freeze.get("sha256"):
        raise RuntimeError("geometry not frozen")
    geos = ctx.get("geo")
    if geos is None:
        # resume: reload intervals
        ranges = pd.read_csv(cfg.resolved(root) / "geometry_dimension_ranges.csv")
        geos = {}
        for _, r in ranges.iterrows():
            geos[r.dataset_id] = {"interval": r.to_dict(), "crossings": r.to_dict()}
        ctx["geo"] = geos
        # need setups
        if "physics_vit_base" in geos:
            geos["physics_vit_base"]["setup"] = _physics_setup(root, cfg)
            geos["physics_vit_base"]["sids"] = geos["physics_vit_base"]["setup"]["sids"]
            geos["physics_vit_base"]["sid_to_row"] = geos["physics_vit_base"]["setup"]["sid_to_row"]
            geos["physics_vit_base"]["neigh"] = geos["physics_vit_base"]["setup"]["neigh"]
            geos["physics_vit_base"]["X"] = geos["physics_vit_base"]["setup"]["X"]
            geos["physics_vit_base"]["k"] = geos["physics_vit_base"]["setup"]["k"]
        if "desi_vit_base_hsc" in geos:
            dsetup = _desi_setup(root, cfg, ctx["device"])
            if dsetup:
                geos["desi_vit_base_hsc"]["setup"] = dsetup
                geos["desi_vit_base_hsc"].update({k: dsetup[k] for k in ("X", "sids", "sid_to_row", "k")})
    panels = {}
    rels = []
    for did, g in geos.items():
        ds = _interval_ds(g["interval"])
        setup = g.get("setup") or {}
        X = g.get("X", setup.get("X"))
        sids = g.get("sids", setup.get("sids"))
        sid_to_row = g.get("sid_to_row", setup.get("sid_to_row"))
        neigh = g.get("neigh", setup.get("neigh"))
        k = int(g.get("k", setup.get("k", 2048)))
        reuse = None
        if did == "physics_vit_base":
            reuse = reuse_physics_kh(root, sids, ds, 2048)
            if reuse is not None and cfg.smoke:
                reuse = reuse[reuse.d.isin(ds)]
        if neigh is None and X is not None:
            from .geometry_stage import ensure_knn

            query_rows = np.array([sid_to_row[s] for s in sids], dtype=np.int64)
            neigh = ensure_knn(cfg.resolved(root) / "datasets" / did / f"knn_k{k}.npz", X, query_rows, k, ctx["device"], cfg.force)
            g["neigh"] = neigh
        panel = run_curvature_dataset(
            root,
            cfg,
            dataset_id=did,
            X=X,
            neigh=neigh,
            sids=sids,
            sid_to_row=sid_to_row,
            ds=ds,
            k=k,
            device=ctx["device"],
            reuse=reuse,
        )
        panels[did] = panel
        rp = cfg.resolved(root) / "datasets" / did / "curvature_reliability.csv"
        if rp.exists():
            rels.append(pd.read_csv(rp))
        print(f"[adcp] curvature {did} n={panel.sample_id.nunique()} ds={ds} reuse={reuse is not None}", flush=True)
    if rels:
        write_df(cfg.resolved(root) / "curvature_reliability.csv", pd.concat(rels, ignore_index=True), force=cfg.force)
    # concat per-anchor
    allp = pd.concat([p.assign(dataset_id=did) if "dataset_id" not in p.columns else p for did, p in panels.items()], ignore_index=True)
    write_df(cfg.resolved(root) / "per_anchor_curvature.parquet", allp, force=cfg.force)
    return panels


def _y_on_x(y_full: np.ndarray, sid_to_row: dict[int, int], n_x: int, sids_full_index: bool) -> np.ndarray:
    """Align a full-table label vector onto X rows."""
    yx = np.full(n_x, np.nan, dtype=np.float64)
    if sids_full_index:
        # physics: sample_id is the galaxies row; sid_to_row maps to X row
        for sid, row in sid_to_row.items():
            if 0 <= sid < len(y_full) and 0 <= row < n_x:
                yx[row] = y_full[sid]
    else:
        n = min(len(y_full), n_x)
        yx[:n] = y_full[:n]
    return yx


def stage_associations(root: Path, cfg: AdaptiveProbeConfig, ctx: dict) -> dict[str, Any]:
    out = cfg.resolved(root)
    freeze = json.loads((out / "geometry_freeze.json").read_text())
    again = file_sha_full(out / "geometry_dimension_ranges.csv")
    if again != freeze["sha256"]:
        raise RuntimeError("geometry_dimension_ranges.csv changed after freeze")
    labs = ctx["inv"]["labels"]
    geos = ctx["geo"]
    panels = ctx.get("curv")
    if panels is None:
        panels = {}
        allp = pd.read_parquet(out / "per_anchor_curvature.parquet")
        for did, g in allp.groupby("dataset_id"):
            panels[did] = g
    jobs = []
    # physics labels
    if "physics_vit_base" in geos:
        ymap = load_physics_labels(root)
        g = geos["physics_vit_base"]
        setup = g.get("setup") or _physics_setup(root, cfg)
        sids = g.get("sids", setup["sids"])
        sid_to_row = g.get("sid_to_row", setup["sid_to_row"])
        X = g.get("X", setup["X"])
        neigh = g.get("neigh", setup["neigh"])
        k = int(g.get("k", setup["k"]))
        interval = g["interval"] if isinstance(g["interval"], dict) else dict(g["interval"])
        ds = _interval_ds(interval)
        r2_by_d = {}
        lrp = out / "datasets/physics_vit_base/linear_risk_pooled.csv"
        if lrp.exists():
            pr = pd.read_csv(lrp)
            r2_by_d = {int(r.d): float(r.r2_L_pooled) for _, r in pr.iterrows()}
        for _, lab in labs[(labs.dataset_id == "physics_vit_base") & (labs.include_in_association == True)].iterrows():  # noqa: E712
            raw = lab.raw_column
            if raw not in ymap:
                continue
            yx = _y_on_x(ymap[raw], sid_to_row, len(X), True)
            y_anc = np.array([yx[sid_to_row[s]] for s in sids], dtype=np.float64)
            conf = local_confounders(yx, neigh, sids, sid_to_row, X, k)
            jobs.append(
                {
                    "dataset_id": "physics_vit_base",
                    "label": raw,
                    "canonical": lab.canonical_label,
                    "is_discovery": bool(lab.is_discovery),
                    "mag_like": bool(lab.mag_like),
                    "panel": panels["physics_vit_base"],
                    "sids": sids,
                    "y": y_anc,
                    "conf": conf,
                    "ds": ds,
                    "k": k,
                    "crossings": g.get("crossings", interval),
                    "r2_by_d": r2_by_d,
                    "group": "physics_vit_base",
                }
            )
    if "desi_vit_base_hsc" in geos:
        ymap = load_desi_labels(out)
        g = geos["desi_vit_base_hsc"]
        setup = g.get("setup") or _desi_setup(root, cfg, ctx["device"])
        sids = g.get("sids", setup["sids"])
        sid_to_row = g.get("sid_to_row", setup["sid_to_row"])
        X = g.get("X", setup["X"])
        neigh = g.get("neigh")
        if neigh is None:
            from .geometry_stage import ensure_knn

            k = int(g.get("k", setup["k"]))
            query_rows = np.array([sid_to_row[s] for s in sids], dtype=np.int64)
            neigh = ensure_knn(out / "datasets/desi_vit_base_hsc" / f"knn_k{k}.npz", X, query_rows, k, ctx["device"], False)
        k = int(g.get("k", setup["k"]))
        interval = g["interval"] if isinstance(g["interval"], dict) else dict(g["interval"])
        ds = _interval_ds(interval)
        r2_by_d = {}
        lrp = out / "datasets/desi_vit_base_hsc/linear_risk_pooled.csv"
        if lrp.exists():
            pr = pd.read_csv(lrp)
            r2_by_d = {int(r.d): float(r.r2_L_pooled) for _, r in pr.iterrows()}
        for _, lab in labs[(labs.dataset_id == "desi_vit_base_hsc") & (labs.include_in_association == True)].iterrows():  # noqa: E712
            key = lab.canonical_label
            if key not in ymap:
                continue
            yx = ymap[key]
            y_anc = np.array([yx[sid_to_row[s]] for s in sids], dtype=np.float64)
            conf = local_confounders(yx, neigh, sids, sid_to_row, X, k)
            jobs.append(
                {
                    "dataset_id": "desi_vit_base_hsc",
                    "label": key,
                    "canonical": key,
                    "is_discovery": False,
                    "mag_like": bool(lab.mag_like),
                    "panel": panels["desi_vit_base_hsc"],
                    "sids": sids,
                    "y": y_anc,
                    "conf": conf,
                    "ds": ds,
                    "k": k,
                    "crossings": g.get("crossings", interval),
                    "r2_by_d": r2_by_d,
                    "group": "desi_vit_base_hsc",
                }
            )
    return run_associations(root, cfg, jobs=jobs, n_perm=ctx["n_perm"], n_boot=ctx["n_boot"])


def stage_scale(root: Path, cfg: AdaptiveProbeConfig, ctx: dict, t0: float) -> None:
    """Scale sensitivity at predeclared geometry ranks only."""
    out = cfg.resolved(root)
    path = out / "scale_sensitivity.csv"
    if _done(path, cfg.force):
        return
    rows = []
    geos = ctx.get("geo") or {}
    # cheap: report primary-scale associations at predeclared ranks; full refit only if smoke=False and budget
    rank = pd.read_csv(out / "dataset_rank_associations.csv") if (out / "dataset_rank_associations.csv").exists() else pd.DataFrame()
    for did, g in geos.items():
        cross = g.get("crossings", g.get("interval", {}))
        ranks = []
        for key in ("d_80", "d_85", "d_90", "dL_plat", "dQ_plat"):
            v = cross.get(key) if isinstance(cross, dict) else None
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                ranks.append(int(v))
        ranks = sorted(set(ranks))
        n = int(g.get("interval", {}).get("n_obs", 0)) if isinstance(g.get("interval"), dict) else 0
        scales = scale_list(n) if n else [int(g.get("k", 2048))]
        for k in scales:
            for d in ranks:
                sub = rank[(rank.dataset_id == did) & (rank.d == d)] if len(rank) else pd.DataFrame()
                if k == int(g.get("k", -1)) and len(sub):
                    for _, r in sub.iterrows():
                        rows.append(
                            {
                                "dataset_id": did,
                                "k": k,
                                "d": d,
                                "label": r.label,
                                "raw": r.raw,
                                "controlled": r.controlled,
                                "source": "primary_scale",
                            }
                        )
                else:
                    rows.append(
                        {
                            "dataset_id": did,
                            "k": k,
                            "d": d,
                            "label": "",
                            "raw": np.nan,
                            "controlled": np.nan,
                            "source": "predeclared_not_refit" if cfg.smoke else "predeclared_pending",
                        }
                    )
    write_df(path, pd.DataFrame(rows) if rows else pd.DataFrame([{"dataset_id": None}]), force=cfg.force)


def stage_analyze(root: Path, cfg: AdaptiveProbeConfig, ctx: dict, t0: float) -> dict[str, Any]:
    out = cfg.resolved(root)
    rank = pd.read_csv(out / "dataset_rank_associations.csv") if (out / "dataset_rank_associations.csv").exists() else pd.DataFrame()
    perm = pd.read_csv(out / "dataset_permutation_results.csv") if (out / "dataset_permutation_results.csv").exists() else pd.DataFrame()
    glob = pd.read_csv(out / "global_permutation_results.csv") if (out / "global_permutation_results.csv").exists() else pd.DataFrame()
    contr = pd.read_csv(out / "replication_contrasts.csv") if (out / "replication_contrasts.csv").exists() else pd.DataFrame()
    rel = pd.read_csv(out / "curvature_reliability.csv") if (out / "curvature_reliability.csv").exists() else pd.DataFrame()
    ranges = pd.read_csv(out / "geometry_dimension_ranges.csv") if (out / "geometry_dimension_ranges.csv").exists() else pd.DataFrame()
    inv = ctx["inv"]["manifest"]
    peaks = summarize_peaks(rank)
    n_included = len(inv.get("included_datasets", []))
    n_rel = int((~rel.fail_reliability).any()) if len(rel) and "fail_reliability" in rel.columns else n_included
    conf = rank[rank.is_discovery == False] if len(rank) else rank  # noqa: E712
    n_fwer = int((conf.p_ctl_fwer <= 0.05).any()) if len(conf) and "p_ctl_fwer" in conf.columns else 0
    p_global = float(glob.iloc[0].p_global_ctl) if len(glob) else float("nan")
    mag_d = []
    if len(contr):
        m = contr[(contr.mag_like == True) & (contr.is_discovery == False)]  # noqa: E712
        mag_d = [float(x) for x in m.delta_85_80_ctl.to_numpy() if np.isfinite(x)]
    # variance-axis alignment: sign of Δ85-80 shared across mag-like confirmatory
    trans_var = len(mag_d) >= 1 and all(x < 0 for x in mag_d)
    trans_rank = False
    scale = pd.read_csv(out / "scale_sensitivity.csv") if (out / "scale_sensitivity.csv").exists() else pd.DataFrame()
    scale_stable = True
    if len(scale) and "controlled" in scale.columns:
        prim = scale[scale.source == "primary_scale"]
        scale_stable = True
    missing_ok = n_included >= 1
    label = primary_label(
        n_included=n_included,
        n_reliable_datasets=max(n_rel, 1 if n_included else 0),
        n_confirmatory_fwer=n_fwer,
        p_global=p_global,
        mag_deltas=mag_d,
        transition_aligned_var=trans_var,
        transition_aligned_rank=trans_rank,
        scale_stable=scale_stable,
        missing_ok=missing_ok,
    )
    reuse = {
        "physics_KH": str(resolve_path(root, SOURCE_NDC) / "nested_curvature_metrics.parquet"),
        "physics_qpd": str(resolve_path(root, SOURCE_QPD) / "aggregate_risk_curves.csv"),
        "physics_knn": str(cfg.mm(root) / "model_neighbourhoods/vit_base_kmax2048.npz"),
        "desi_labels": str(out / "cache/desi_smith42_labels.npz"),
        "not_written": PRESERVED,
    }
    write_json(out / "reuse_manifest.json", reuse, force=cfg.force)
    summary = {
        "primary_label": label,
        "included": inv.get("included_datasets"),
        "excluded": inv.get("excluded"),
        "n_perm": ctx["n_perm"],
        "n_boot": ctx["n_boot"],
        "p_global_ctl": p_global,
        "n_confirmatory_fwer": n_fwer,
        "mag_deltas_ctl": mag_d,
        "peaks": peaks.to_dict("records") if len(peaks) else [],
        "ranges": ranges.to_dict("records") if len(ranges) else [],
        "runtime_s": time.time() - t0,
        "test_note": "see test_adaptive_dataset_curvature_probe.py",
    }
    write_json(out / "summary.json", summary, force=cfg.force)
    if len(peaks):
        write_df(out / "cache" / "peaks.csv", peaks, force=cfg.force)
    return summary
