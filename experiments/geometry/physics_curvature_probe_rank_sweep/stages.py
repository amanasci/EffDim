"""Stages: reuse, parity, inference, optional scale refit, report."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from geometry.physics_activation_atlas.curvature_probe_screen import partial_spearman, spearman_dict
from geometry.physics_activation_atlas.effdim_curvature_metrics import metric_scalars
from geometry.physics_activation_atlas.nested_dimension_curvature import _fit_rank, ensure_neigh, nested_pca_frame
from geometry.physics_activation_atlas.paths import platonic_root, resolve_path

from .classify import DEFAULT_THRESHOLDS, primary_label
from .inference import (
    CONTROLS,
    associate,
    bootstrap_crossings,
    control_matrix,
    curve_from_panel,
    paired_bootstrap_curves,
    permutation_curves,
)
from .pipeline import (
    FREEZE_HASH_EXPECTED,
    SOURCE_NDC,
    SOURCE_QPD,
    SOURCE_STD,
    PARITY_D12_RHO,
    PARITY_D16_RHO,
    PARITY_NDC_D12_RAW,
    PARITY_NDC_D16_CTL,
    PARITY_NDC_D16_RAW,
    PARITY_TOL,
    PRESERVED,
    RankSweepConfig,
    _budget_ok,
    _done,
    _file_sha,
    assert_not_preserved,
    hash_select,
    kh_trace_identity,
    load_ctx,
    write_df,
)
from .synthetics import SYNTH_SEEDS, eval_family

STAGES = [
    "prepare",
    "reuse",
    "parity",
    "synthetic_validation",
    "assemble",
    "associations",
    "permutation",
    "bootstrap",
    "reliability",
    "variance",
    "scale_sensitivity",
    "analyze",
    "report",
]


def stage_prepare(root: Path, cfg: RankSweepConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    out.mkdir(parents=True, exist_ok=True)
    assert_not_preserved(out, root)
    for sub in ("cache", "figures", "logs", "synth"):
        (out / sub).mkdir(exist_ok=True)
    meta = {
        "config": {**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__}},
        "protocol": "curvature_probe_rank_sweep_v1",
        "preserved": PRESERVED,
        "n_anchors": len(ctx["use_sids"]),
        "scale_anchor_ids": ctx["scale_sids"],
        "scale_anchor_rule": "sha256(cprs:{seed}:{sample_id}) prefix, disclosed before any scale fit",
        "primary_k": cfg.primary_k,
        "ds": cfg.ds(),
        "primary_ds": cfg.primary_ds(),
        "no_probe_selection": True,
        "rank_conditional": "K_H^{(d)} is the curvature estimate under a rank-d chart, not one geometric object",
        "not_preregistered": True,
        "l2_status": ctx["l2"],
        "hashes": {
            "freeze": ctx["freeze"].get("dimension_config_hash"),
            "ndc_metrics": _file_sha(ctx["ndc"] / "nested_curvature_metrics.parquet"),
            "qpd_curves": _file_sha(ctx["qpd"] / "aggregate_risk_curves.csv") if (ctx["qpd"] / "aggregate_risk_curves.csv").exists() else None,
        },
        "expected_freeze_hash": FREEZE_HASH_EXPECTED,
        "seeds": {"analysis": cfg.seed, "synth": SYNTH_SEEDS},
        "expected_runtime_note": "primary path reuses nested K_H; inference is scalar-table only (~minutes). scale fit only if K_H missing at other k.",
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[cprs] prepare n={len(ctx['use_sids'])} ds={cfg.ds()} scale_subset={len(ctx['scale_sids'])}", flush=True)
    return meta


def stage_reuse(root: Path, cfg: RankSweepConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "reuse_manifest.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    ndc_p = ctx["ndc"] / "nested_curvature_metrics.parquet"
    ndc = pd.read_parquet(ndc_p)
    if "model" in ndc.columns:
        ndc = ndc[ndc.model == cfg.model]
    ndc = ndc[ndc.k == cfg.primary_k]
    have_d = sorted(int(x) for x in ndc.d.unique())
    missing_d = [d for d in cfg.ds() if d not in have_d]
    reuse = {
        "primary_KH_source": str(ndc_p),
        "primary_KH_sha": _file_sha(ndc_p),
        "primary_k": cfg.primary_k,
        "reused_d": have_d,
        "missing_d_requiring_refit": missing_d,
        "n_rows": int(len(ndc)),
        "n_anchors": int(ndc.sample_id.nunique()),
        "estimator": "nested_dimension_curvature.K_H_cross (split-half inner product of H_S)",
        "recomputed": [
            "permutation tests",
            "paired bootstrap",
            "simultaneous bands",
            "variance crossings from qpd lin_r2",
            "figures and labels",
        ],
        "not_recomputed": [
            "embeddings",
            "kNN",
            "sphere-log / nested PCA at k=2048",
            "quadratic B_S / K_H at k=2048 d=8..20",
        ],
        "probe_source": str(ctx["mm"] / "local_probe_fields.parquet"),
        "probe_column": "local_r2 filtered by target=mag_r_desi",
        "qpd_source": str(ctx["qpd"] / "per_anchor_metrics.parquet"),
        "trace_acceleration": "not used for production; identity tested only",
    }
    write_df(out / "cache" / "reused_nested_curvature.parquet", ndc, force=cfg.force)
    path.write_text(json.dumps(reuse, indent=2))
    print(f"[cprs] reuse anchors={reuse['n_anchors']} d={have_d} missing={missing_d}", flush=True)
    if missing_d:
        raise RuntimeError(f"primary K_H missing ranks {missing_d}; refuse silent estimator change")
    return reuse


def stage_parity(root: Path, cfg: RankSweepConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "parity.json"
    if _done(path, cfg.force):
        return json.loads(path.read_text())
    result: dict[str, Any] = {"ok": True, "corrections": []}
    freeze_hash = ctx["freeze"].get("dimension_config_hash")
    result["freeze_hash"] = freeze_hash
    result["freeze_hash_ok"] = freeze_hash == FREEZE_HASH_EXPECTED
    result["n_anchors"] = len(ctx["use_sids"])
    result["n_anchors_ok"] = len(ctx["use_sids"]) >= (8 if cfg.smoke else 500)
    result["l2_unit_normalized"] = bool(ctx["l2"].get("unit_normalized", False))
    result["probe_target"] = cfg.target
    result["probe_column"] = "local_r2"
    result["coord_note"] = "nested curvature uses mean-centred sphere-tangent PCA (not qpd uncentred log SVD)"
    result["KH_definition"] = "K_H_cross = <H_A, H_B>; H = mean_a B[:,a,a]; Q_R removed; B_S sphere-normal"
    result["whitening"] = "per-dim RMS of fit-split tangent coords before ridge, RIDGES=[1e-4..3]"
    result["confounders"] = list(CONTROLS)
    result["permutation_model"] = "raw: permute probe; controlled: rank-space Freedman-Lane residual permutation"
    hard = bool(result["freeze_hash_ok"] and result["n_anchors_ok"] and result["l2_unit_normalized"])
    geo = ctx["geo"][ctx["geo"].scale_k == cfg.primary_k]
    result["probe_n"] = int(geo.sample_id.nunique())
    result["probe_align_ok"] = set(ctx["use_sids"]).issubset(set(geo.sample_id.astype(int)))
    hard = bool(hard and result["probe_align_ok"])
    try:
        cov = pd.read_parquet(ctx["cov"] / "model_reliability_anchor_mean.parquet")
        c16 = cov[(cov.model == cfg.model) & (cov.k == cfg.primary_k) & (cov.d == 16)]
        m16 = geo.merge(c16, on="sample_id", how="inner")
        rho16, _ = spearmanr(m16.K_H_cross, m16.local_r2)
        edm = pd.read_parquet(ctx["edm"] / "crossfit_curvature_metrics.parquet")
        e12 = edm[(edm.model == cfg.model) & (edm.k == cfg.primary_k) & (edm.role == "d_star") & (edm.d == 12)]
        m12 = geo.merge(e12, on="sample_id", how="inner")
        rho12, _ = spearmanr(m12.K_H_cross, m12.local_r2)
        result["upstream_d16"] = {"rho": float(rho16), "expected": PARITY_D16_RHO, "ok": abs(float(rho16) - PARITY_D16_RHO) <= PARITY_TOL, "n": int(len(m16))}
        result["upstream_d12"] = {"rho": float(rho12), "expected": PARITY_D12_RHO, "ok": abs(float(rho12) - PARITY_D12_RHO) <= PARITY_TOL, "n": int(len(m12))}
    except Exception as e:  # noqa: BLE001
        result["corrections"].append(f"upstream:{e}")
        result["upstream_d16"] = {"ok": False}
        result["upstream_d12"] = {"ok": False}
    ndc = pd.read_parquet(out / "cache" / "reused_nested_curvature.parquet")
    agg = ndc.groupby(["sample_id", "d"], as_index=False)[["K_H_cross", "K_aniso_cross", "K_dir_cross"]].mean()
    geo_p = geo

    def rho_at(d, col="K_H_cross", controlled=False):
        g = geo_p.merge(agg[agg.d == d], on="sample_id")
        if controlled:
            return partial_spearman(g[col].to_numpy(float), g.local_r2.to_numpy(float), control_matrix(g))["rho"]
        return spearman_dict(g[col].to_numpy(float), g.local_r2.to_numpy(float))["rho"]

    r12 = rho_at(12)
    r16 = rho_at(16)
    c16 = rho_at(16, controlled=True)
    result["nested_d12_raw"] = {"rho": r12, "expected": PARITY_NDC_D12_RAW, "ok": abs(r12 - PARITY_NDC_D12_RAW) <= PARITY_TOL}
    result["nested_d16_raw"] = {"rho": r16, "expected": PARITY_NDC_D16_RAW, "ok": abs(r16 - PARITY_NDC_D16_RAW) <= PARITY_TOL}
    result["nested_d16_controlled"] = {"rho": c16, "expected": PARITY_NDC_D16_CTL, "ok": abs(c16 - PARITY_NDC_D16_CTL) <= PARITY_TOL}
    result["weak_near_d12"] = bool(abs(r12) < 0.12)
    result["preserved_hashes"] = {rel: _file_sha(resolve_path(root, rel) / name) if (resolve_path(root, rel) / name).exists() else None for rel, name in ((SOURCE_NDC, "nested_curvature_metrics.parquet"), (SOURCE_QPD, "COMPLETE.json"), (SOURCE_STD, "REPORT.md"))}
    # trace identity on H_vectors if present
    hv = ctx["ndc"] / "H_vectors"
    trace_ok = True
    n_tr = 0
    if hv.exists():
        for sid in ctx["use_sids"][: cfg.n_parity_anchors]:
            fp = hv / f"{int(sid)}.npz"
            if not fp.exists():
                continue
            z = np.load(fp)
            if "BS16_A" in z.files:
                kh = metric_scalars(z["BS16_A"], 16)["K_H"]
                tr = kh_trace_identity(z["BS16_A"], 16)
                if abs(kh - tr) > DEFAULT_THRESHOLDS["trace_atol"]:
                    trace_ok = False
                n_tr += 1
    result["trace_identity"] = {"n": n_tr, "ok": trace_ok, "note": "||mean diag B|| matches metric_scalars K_H; production uses reused K_H_cross"}
    result["ok"] = bool(hard and result["nested_d16_raw"]["ok"] and result["nested_d12_raw"]["ok"] and result["upstream_d16"].get("ok") and result["upstream_d12"].get("ok"))
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[cprs] parity ok={result['ok']} corrections={result['corrections']}", flush=True)
    if not result["ok"]:
        raise RuntimeError("parity failed; see parity.json")
    return result


def _panel(ctx: dict, cfg: RankSweepConfig, ndc: pd.DataFrame, k: int) -> pd.DataFrame:
    geo = ctx["geo"]
    # nearest available probe scale
    scales = sorted(geo.scale_k.unique())
    sk = k if k in scales else min(scales, key=lambda s: abs(int(s) - k))
    g = geo[geo.scale_k == sk]
    agg = ndc[ndc.k == k].groupby(["sample_id", "d", "k"], as_index=False).mean(numeric_only=True)
    panel = g.merge(agg, on="sample_id", how="inner")
    panel["probe_scale_k"] = sk
    panel["probe_scale_matched"] = sk == k
    return panel


def _wide(panel: pd.DataFrame, ds: list[int], xcol: str = "K_H_cross") -> pd.DataFrame:
    base = panel.drop_duplicates("sample_id")[["sample_id", "local_r2", *CONTROLS]].copy()
    for d in ds:
        gd = panel[panel.d == d][["sample_id", xcol]].drop_duplicates("sample_id")
        base = base.merge(gd.rename(columns={xcol: f"KH{d}"}), on="sample_id", how="left")
    return base


def stage_synthetic_validation(root: Path, cfg: RankSweepConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "synthetic_validation.csv"
    if _done(path, cfg.force):
        return
    ds = cfg.primary_ds()
    rows = []
    for kind in ("null", "var_only", "confound", "planted16"):
        rec = eval_family(kind, ds, seed=SYNTH_SEEDS["evaluation"] + {"null": 1, "var_only": 2, "confound": 3, "planted16": 4}[kind], n=200 if cfg.smoke else 320, n_perm=min(cfg.n_perm, 400 if cfg.smoke else 800))
        rows.append(rec)
    write_df(path, pd.DataFrame(rows), force=cfg.force)
    (out / "thresholds.json").write_text(json.dumps(DEFAULT_THRESHOLDS, indent=2))
    print(f"[cprs] synthetic_validation n={len(rows)}", flush=True)


def stage_assemble(root: Path, cfg: RankSweepConfig, ctx: dict) -> pd.DataFrame:
    out = cfg.resolved(root)
    ndc = pd.read_parquet(out / "cache" / "reused_nested_curvature.parquet")
    panel = _panel(ctx, cfg, ndc, cfg.primary_k)
    panel = panel[panel.sample_id.isin(ctx["use_sids"])]
    write_df(out / "per_anchor_rank_curve.parquet", panel, force=cfg.force)
    print(f"[cprs] assemble panel n={panel.sample_id.nunique()} rows={len(panel)}", flush=True)
    return panel


def stage_associations(root: Path, cfg: RankSweepConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    panel = pd.read_parquet(out / "per_anchor_rank_curve.parquet")
    ds = cfg.ds()
    # common complete-case: anchors with all d
    counts = panel.groupby("sample_id").d.nunique()
    common = set(counts[counts >= len(ds)].index)
    rows = []
    for mask_name, sids in (("dimension_specific", set(panel.sample_id)), ("complete_case", common)):
        sub = panel[panel.sample_id.isin(sids)]
        for xcol, lab in (("K_H_cross", "K_H"), ("K_aniso_cross", "traceless"), ("K_dir_cross", "total"), ("dS", "Delta_S")):
            if xcol not in sub.columns:
                continue
            cur = curve_from_panel(sub, ds, ycol="local_r2", xcol=xcol)
            cur["metric"] = lab
            cur["mask"] = mask_name
            rows.append(cur)
    allc = pd.concat(rows, ignore_index=True)
    write_df(out / "dimension_associations.csv", allc[allc.metric == "K_H"], force=cfg.force)
    write_df(out / "controlled_dimension_associations.csv", allc, force=cfg.force)
    print(f"[cprs] associations rows={len(allc)}", flush=True)


def stage_permutation(root: Path, cfg: RankSweepConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "permutation_results.csv"
    if _done(path, cfg.force):
        return
    panel = pd.read_parquet(out / "per_anchor_rank_curve.parquet")
    wide = _wide(panel, cfg.ds())
    raw = permutation_curves(wide, cfg.primary_ds(), ycol="local_r2", x_prefix="KH", n_perm=cfg.n_perm, seed=cfg.seed + 11, controlled=False)
    ctl = permutation_curves(wide, cfg.primary_ds(), ycol="local_r2", x_prefix="KH", n_perm=cfg.n_perm, seed=cfg.seed + 17, controlled=True)
    tab = pd.concat([raw["table"], ctl["table"]], ignore_index=True)
    tab["p_global_raw"] = raw["p_global"]
    tab["p_global_ctl"] = ctl["p_global"]
    write_df(path, tab, force=cfg.force)
    env = []
    for d in cfg.primary_ds():
        env.append({"d": d, "raw_null95": raw["null_envelope"][d], "ctl_null95": ctl["null_envelope"][d]})
    write_df(out / "cache" / "perm_envelopes.csv", pd.DataFrame(env), force=cfg.force)
    (out / "cache" / "perm_global.json").write_text(json.dumps({"raw": raw["p_global"], "controlled": ctl["p_global"], "tmax_raw": raw["tmax_obs"], "tmax_ctl": ctl["tmax_obs"]}))
    print(f"[cprs] permutation n_perm={cfg.n_perm} p_global_ctl={ctl['p_global']:.4f}", flush=True)


def stage_bootstrap(root: Path, cfg: RankSweepConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "bootstrap_results.csv"
    if _done(path, cfg.force):
        return
    panel = pd.read_parquet(out / "per_anchor_rank_curve.parquet")
    wide = _wide(panel, cfg.ds())
    boot = paired_bootstrap_curves(wide, cfg.ds(), ycol="local_r2", x_prefix="KH", n_boot=cfg.n_boot, seed=cfg.seed + 23)
    write_df(path, boot["table"], force=cfg.force)
    write_df(out / "simultaneous_bands.csv", boot["table"], force=cfg.force)
    pd.DataFrame({"peak_raw": boot["peak_raw"]}).to_csv(out / "cache" / "peak_raw_boot.csv", index=False)
    pd.DataFrame({"peak_ctl": boot["peak_ctl"]}).to_csv(out / "cache" / "peak_ctl_boot.csv", index=False)
    (out / "cache" / "bootstrap_extra.json").write_text(json.dumps(boot["extra"], indent=2, default=str))
    print(f"[cprs] bootstrap n={cfg.n_boot} peak_ctl_mode={boot['extra']['peak_ctl_mode']}", flush=True)


def stage_reliability(root: Path, cfg: RankSweepConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    panel = pd.read_parquet(out / "per_anchor_rank_curve.parquet")
    thr = DEFAULT_THRESHOLDS
    rows = []
    for d, g in panel.groupby("d"):
        n = int(g.sample_id.nunique())
        rh = float(g.R_H.median()) if "R_H" in g.columns else float("nan")
        ds = float(g.dS.median()) if "dS" in g.columns else float("nan")
        fail = bool((np.isfinite(rh) and rh < thr["r_h_fail"]) or (n < thr["valid_frac_fail"] * len(ctx["use_sids"])))
        rows.append(
            {
                "d": int(d),
                "k": int(cfg.primary_k),
                "n": n,
                "R_H_med": rh,
                "dS_med": ds,
                "R_B0_med": float(g.R_B0.median()) if "R_B0" in g.columns else float("nan"),
                "fail_reliability": fail,
                "reason": "low_R_H_or_n" if fail else "ok",
            }
        )
    write_df(out / "curvature_reliability.csv", pd.DataFrame(rows), force=cfg.force)
    miss = panel.groupby("d").sample_id.nunique().reset_index(name="n_valid")
    miss["n_expected"] = len(ctx["use_sids"])
    write_df(out / "cache" / "missingness.csv", miss, force=cfg.force)
    print(f"[cprs] reliability ranks={len(rows)}", flush=True)


def stage_variance(root: Path, cfg: RankSweepConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    qpd = ctx["qpd"] / "per_anchor_metrics.parquet"
    if not qpd.exists():
        write_df(out / "variance_explained.csv", pd.DataFrame(), force=cfg.force)
        write_df(out / "variance_threshold_crossings.csv", pd.DataFrame(), force=cfg.force)
        return
    raw = pd.read_parquet(qpd)
    raw = raw[(raw.k == cfg.primary_k) & (raw.sample_id.isin(ctx["use_sids"]))]
    ds = [d for d in cfg.ds() if d in set(raw.d.astype(int))]
    taus = DEFAULT_THRESHOLDS["var_taus"]
    pack = bootstrap_crossings(raw, ds, taus, n_boot=min(cfg.n_boot, 400 if cfg.smoke else 2000), seed=cfg.seed + 41)
    write_df(out / "variance_explained.csv", pack["curve"], force=cfg.force)
    write_df(out / "variance_threshold_crossings.csv", pack["crossings"], force=cfg.force)
    print(f"[cprs] variance crossings={pack['crossings']['d_tau'].tolist()}", flush=True)


def stage_scale_sensitivity(root: Path, cfg: RankSweepConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "scale_sensitivity.csv"
    if _done(path, cfg.force):
        return
    rows = []
    # always include reused primary k
    assoc = pd.read_csv(out / "dimension_associations.csv")
    sub = assoc[assoc["mask"] == "complete_case"] if "mask" in assoc.columns else assoc
    for _, r in sub.iterrows():
        rows.append({"k": cfg.primary_k, "d": int(r.d), "raw": r.raw, "controlled": r.controlled, "n": r.n, "source": "reused_ndc"})
    extra_ks = [k for k in (512, 1024, 1536) if k != cfg.primary_k]
    if cfg.skip_scale_fit or cfg.smoke:
        write_df(path, pd.DataFrame(rows), force=cfg.force)
        print("[cprs] scale_sensitivity reuse-only (smoke or skip)", flush=True)
        return
    # refit missing k on hash-selected anchors
    for k in extra_ks:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        recs = []
        for sid in ctx["scale_sids"]:
            cache = out / "cache" / f"scale_{cfg.model}_{int(sid)}_k{k}.parquet"
            if cache.exists() and not cfg.force:
                recs.append(pd.read_parquet(cache))
                continue
            ai = ctx["sid_to_ai"][int(sid)]
            N = ensure_neigh(ctx, ai, k)
            Xloc = ctx["X"][N].astype(np.float64)
            x0, J, _, _ = nested_pca_frame(Xloc, cfg.d_max, ctx["device"])
            got = []
            for d in cfg.ds():
                if J.shape[1] < d:
                    continue
                fr = _fit_rank(Xloc, x0, J, d, k, cfg.n_scale_splits, cfg.seed, ai)
                for row in fr:
                    got.append({kk: row[kk] for kk in row if kk not in ("H_mean", "BS_flat_A", "BS_flat_B")})
            df = pd.DataFrame(got)
            if len(df):
                df["sample_id"] = int(sid)
                df["k"] = int(k)
                write_df(cache, df, force=cfg.force)
                recs.append(df)
        if not recs:
            continue
        ndc = pd.concat(recs, ignore_index=True)
        panel = _panel(ctx, cfg, ndc, k)
        cur = curve_from_panel(panel, cfg.ds(), ycol="local_r2", xcol="K_H_cross")
        for _, r in cur.iterrows():
            rows.append({"k": k, "d": int(r.d), "raw": r.raw, "controlled": r.controlled, "n": r.n, "source": "refit_scale_subset"})
        print(f"[cprs] scale k={k} n={panel.sample_id.nunique()}", flush=True)
    write_df(path, pd.DataFrame(rows), force=cfg.force)


def _summary(out: Path, cfg: RankSweepConfig) -> dict[str, Any]:
    assoc = pd.read_csv(out / "dimension_associations.csv")
    assoc = assoc[assoc["mask"] == "complete_case"] if "mask" in assoc.columns else assoc
    perm = pd.read_csv(out / "permutation_results.csv") if (out / "permutation_results.csv").exists() else pd.DataFrame()
    boot = json.loads((out / "cache" / "bootstrap_extra.json").read_text()) if (out / "cache" / "bootstrap_extra.json").exists() else {}
    rel = pd.read_csv(out / "curvature_reliability.csv") if (out / "curvature_reliability.csv").exists() else pd.DataFrame()
    ve = pd.read_csv(out / "variance_threshold_crossings.csv") if (out / "variance_threshold_crossings.csv").exists() else pd.DataFrame()
    scale = pd.read_csv(out / "scale_sensitivity.csv") if (out / "scale_sensitivity.csv").exists() else pd.DataFrame()
    synth = pd.read_csv(out / "synthetic_validation.csv") if (out / "synthetic_validation.csv").exists() else pd.DataFrame()
    gperm = json.loads((out / "cache" / "perm_global.json").read_text()) if (out / "cache" / "perm_global.json").exists() else {}
    raw_map, ctl_map = {}, {}
    for _, r in assoc.iterrows():
        raw_map[int(r.d)] = float(r.raw)
        ctl_map[int(r.d)] = float(r.controlled)
    ctl_p = perm[(perm.kind == "controlled")] if len(perm) else pd.DataFrame()
    fwer_hits = [int(r.d) for _, r in ctl_p.iterrows() if np.isfinite(r.p_fwer) and r.p_fwer <= 0.05] if len(ctl_p) else []
    reliable = {int(r.d): (not bool(r.fail_reliability)) for _, r in rel.iterrows()} if len(rel) else {}
    # tracks reliability?
    if len(rel) and len(assoc):
        m = assoc.merge(rel, on="d")
        if len(m) >= 5:
            rr, _ = spearmanr(np.abs(m.controlled), m.R_H_med)
            tracks = bool(np.isfinite(rr) and abs(rr) >= DEFAULT_THRESHOLDS["reliability_track"])
        else:
            tracks = False
    else:
        tracks = False
    scale_stable = True
    if len(scale) and scale.k.nunique() > 1:
        # compare controlled at d=16 across k
        s16 = scale[scale.d == 16]
        if len(s16) > 1 and s16.controlled.nunique() > 0:
            scale_stable = bool(float(s16.controlled.max() - s16.controlled.min()) <= DEFAULT_THRESHOLDS["scale_disagree"] * 2 or s16.k.nunique() == 1)
    d85 = "not_reached"
    if len(ve):
        hit = ve[np.isclose(ve.tau, 0.85)]
        if len(hit):
            d85 = hit.iloc[0].d_tau
            if isinstance(d85, (float, np.floating)) and np.isfinite(d85):
                d85 = int(d85)
    peak_d = int(max(ctl_map, key=lambda d: abs(ctl_map[d]))) if ctl_map else None
    lab = primary_label(
        fwer_hits=fwer_hits,
        reliable=reliable,
        tracks_rel=tracks,
        scale_stable=scale_stable,
        missing_ok=True,
        thr=DEFAULT_THRESHOLDS,
    )
    return {
        "primary": lab,
        "raw_by_d": raw_map,
        "controlled_by_d": ctl_map,
        "peak_d_controlled": peak_d,
        "peak_rho_controlled": ctl_map.get(peak_d) if peak_d is not None else None,
        "peak_d_raw": int(max(raw_map, key=lambda d: abs(raw_map[d]))) if raw_map else None,
        "fwer_hits_controlled": fwer_hits,
        "p_global_raw": gperm.get("raw"),
        "p_global_controlled": gperm.get("controlled"),
        "d85": d85,
        "rho_at_d85_raw": raw_map.get(int(d85)) if str(d85) not in ("not_reached", "nan") and d85 is not None else None,
        "rho_at_d85_ctl": ctl_map.get(int(d85)) if str(d85) not in ("not_reached", "nan") and d85 is not None else None,
        "bootstrap": boot,
        "scale_stable": scale_stable,
        "tracks_reliability": tracks,
        "synth_not_only12": bool(len(synth) and synth.peak_raw.nunique() > 1) if len(synth) and "peak_raw" in synth.columns else True,
        "not_preregistered": True,
    }


def run(cfg: RankSweepConfig) -> dict[str, Any]:
    root = platonic_root()
    out = cfg.resolved(root)
    t0 = time.time()
    ctx = load_ctx(root, cfg)
    assert_not_preserved(out, root)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    profile: dict[str, Any] = {"stages": {}, "completed": []}
    want = STAGES if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    if "all" in want:
        want = list(STAGES)
    run_set = set(want)

    def mark(name, dt):
        profile["stages"][f"{name}_s"] = dt
        if name not in profile["completed"]:
            profile["completed"].append(name)
        (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    if "prepare" in run_set:
        print("[cprs] stage=prepare", flush=True)
        t1 = time.time()
        stage_prepare(root, cfg, ctx)
        mark("prepare", time.time() - t1)
    if "reuse" in run_set:
        print("[cprs] stage=reuse", flush=True)
        t1 = time.time()
        stage_reuse(root, cfg, ctx)
        mark("reuse", time.time() - t1)
    if "parity" in run_set:
        print("[cprs] stage=parity", flush=True)
        t1 = time.time()
        try:
            stage_parity(root, cfg, ctx)
        except RuntimeError:
            from .report import write_methods, write_report
            parity = json.loads((out / "parity.json").read_text()) if (out / "parity.json").exists() else {"ok": False}
            write_methods(out, cfg, ctx, parity, DEFAULT_THRESHOLDS)
            write_report(out, cfg, ctx, parity, {"primary": "curvature_probe_rank_sweep_unresolved", "parity_failed": True})
            raise
        mark("parity", time.time() - t1)
    if "synthetic_validation" in run_set:
        print("[cprs] stage=synthetic_validation", flush=True)
        t1 = time.time()
        stage_synthetic_validation(root, cfg, ctx)
        mark("synthetic_validation", time.time() - t1)
    if "assemble" in run_set:
        print("[cprs] stage=assemble", flush=True)
        t1 = time.time()
        stage_assemble(root, cfg, ctx)
        mark("assemble", time.time() - t1)
    if "associations" in run_set:
        print("[cprs] stage=associations", flush=True)
        t1 = time.time()
        stage_associations(root, cfg, ctx)
        mark("associations", time.time() - t1)
    if "permutation" in run_set:
        print("[cprs] stage=permutation", flush=True)
        t1 = time.time()
        stage_permutation(root, cfg, ctx)
        mark("permutation", time.time() - t1)
    if "bootstrap" in run_set:
        print("[cprs] stage=bootstrap", flush=True)
        t1 = time.time()
        stage_bootstrap(root, cfg, ctx)
        mark("bootstrap", time.time() - t1)
    if "reliability" in run_set:
        print("[cprs] stage=reliability", flush=True)
        t1 = time.time()
        stage_reliability(root, cfg, ctx)
        mark("reliability", time.time() - t1)
    if "variance" in run_set:
        print("[cprs] stage=variance", flush=True)
        t1 = time.time()
        stage_variance(root, cfg, ctx)
        mark("variance", time.time() - t1)
    if "scale_sensitivity" in run_set:
        print("[cprs] stage=scale_sensitivity", flush=True)
        t1 = time.time()
        stage_scale_sensitivity(root, cfg, ctx, t0)
        mark("scale_sensitivity", time.time() - t1)
    if "analyze" in run_set or "report" in run_set:
        print("[cprs] stage=analyze/report", flush=True)
        t1 = time.time()
        from .plots import write_figures
        from .report import write_methods, write_report
        labels = _summary(out, cfg)
        (out / "summary.json").write_text(json.dumps(labels, indent=2, default=str))
        try:
            write_figures(out, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[cprs] figures failed: {e}", flush=True)
        parity = json.loads((out / "parity.json").read_text()) if (out / "parity.json").exists() else {}
        write_methods(out, cfg, ctx, parity, DEFAULT_THRESHOLDS)
        write_report(out, cfg, ctx, parity, labels)
        tmp = out / "COMPLETE.json.tmp"
        tmp.write_text(json.dumps({"ok": True, "primary": labels.get("primary"), "t": time.time()}, indent=2))
        tmp.replace(out / "COMPLETE.json")
        mark("analyze", time.time() - t1)
        mark("report", 0.0)
    profile["total_seconds"] = time.time() - t0
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[cprs] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
