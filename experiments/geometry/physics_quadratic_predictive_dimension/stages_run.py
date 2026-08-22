"""Aggregate, labels, and the stage runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_stable_tangent_dimension.dimension import paired_bootstrap_ci

from .classify import DEFAULT_THRESHOLDS, adequacy_ranks, plateau_from_curve, primary_label
from .pipeline import (
    QuadPredConfig,
    _done,
    assert_not_preserved,
    load_ctx,
    platonic_root,
    stage_prepare,
    write_df,
)
from .stages import (
    STAGES,
    _eval_synth,
    stage_fit_primary,
    stage_parity,
    stage_scale_sensitivity,
    stage_synthetic_calibration,
)
from .synthetics import SYNTH_KINDS, make_predictive_synthetic, split_seeds


def _load_all_cache(out: Path) -> pd.DataFrame:
    files = list((out / "cache").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def stage_aggregate(root: Path, cfg: QuadPredConfig, ctx: dict, thr: dict) -> None:
    out = cfg.resolved(root)
    raw = _load_all_cache(out)
    if not len(raw):
        return
    write_df(out / "per_anchor_metrics.parquet", raw, force=cfg.force)
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    prim = raw[raw.k == k_ref]
    loc = prim.groupby(["sample_id", "d"], as_index=False).mean(numeric_only=True)
    n_boot = int(thr.get("n_boot", 400))
    rows = []
    for d, g in loc.groupby("d"):
        def boot(col):
            if col not in g.columns:
                return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
            return paired_bootstrap_ci(g[col].to_numpy(), n_boot=n_boot, seed=cfg.seed + int(d))

        bq, bl, br = boot("quad_close_nmse"), boot("lin_nmse"), boot("quad_close_r2")
        pooled_q = (
            float(g.test_sse_close.sum() / max(g.test_energy.sum(), 1e-18))
            if "test_sse_close" in g.columns
            else float("nan")
        )
        pooled_l = (
            float(g.test_sse_lin.sum() / max(g.test_energy.sum(), 1e-18))
            if "test_sse_lin" in g.columns
            else float("nan")
        )
        rows.append(
            {
                "d": int(d), "k": int(k_ref), "n": int(len(g)),
                "nmse_quad_med": bq["point"], "nmse_quad_lo": bq["lo"], "nmse_quad_hi": bq["hi"],
                "nmse_quad_q25": float(g.quad_close_nmse.quantile(0.25)),
                "nmse_quad_q75": float(g.quad_close_nmse.quantile(0.75)),
                "nmse_lin_med": bl["point"], "nmse_lin_lo": bl["lo"], "nmse_lin_hi": bl["hi"],
                "nmse_lin_q25": float(g.lin_nmse.quantile(0.25)),
                "nmse_lin_q75": float(g.lin_nmse.quantile(0.75)),
                "r2_quad_med": br["point"], "r2_quad_lo": br["lo"], "r2_quad_hi": br["hi"],
                "nmse_quad_pooled": pooled_q,
                "r2_quad_pooled": float(1.0 - pooled_q) if np.isfinite(pooled_q) else float("nan"),
                "nmse_lin_pooled": pooled_l,
                "r2_lin_pooled": float(1.0 - pooled_l) if np.isfinite(pooled_l) else float("nan"),
                "nmse_quad_fixed_med": float(g.quad_fixed_nmse.median()) if "quad_fixed_nmse" in g.columns else float("nan"),
                "nmse_quadN_med": float(g.quadN_close_nmse.median()) if "quadN_close_nmse" in g.columns else float("nan"),
                "df_med": float(g.df.median()) if "df" in g.columns else float("nan"),
                "df_frac_med": float(g.df_frac.median()) if "df_frac" in g.columns else float("nan"),
                "lam_med": float(g.lam.median()) if "lam" in g.columns else float("nan"),
                "train_nmse_med": float(g.train_nmse_fixed.median()) if "train_nmse_fixed" in g.columns else float("nan"),
                "boundary_frac": float(g.boundary_frac.median()) if "boundary_frac" in g.columns else float("nan"),
                "mean_n_iter": float(g.mean_n_iter.median()) if "mean_n_iter" in g.columns else float("nan"),
                "r2_E4_med": float(g.r2_E4.median()) if "r2_E4" in g.columns else float("nan"),
                "r2_U8_med": float(g.r2_U8.median()) if "r2_U8" in g.columns else float("nan"),
                "r2_T12_med": float(g.r2_T12.median()) if "r2_T12" in g.columns else float("nan"),
                "r2_U4_med": float(g.r2_U4.median()) if "r2_U4" in g.columns else float("nan"),
                "geo_err_med": float(g.geo_err.median()) if "geo_err" in g.columns else float("nan"),
            }
        )
    curves = pd.DataFrame(rows).sort_values("d")
    paired = []
    for d in curves.d.tolist()[1:]:
        a = loc[loc.d == d - 1].set_index("sample_id")
        b = loc[loc.d == d].set_index("sample_id")
        m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        if "quad_close_nmse_a" in m.columns:
            diff = m.quad_close_nmse_a - m.quad_close_nmse_b
            ci = paired_bootstrap_ci(diff.to_numpy(), n_boot=n_boot, seed=cfg.seed + int(d) * 3)
            paired.append({"d": int(d), "delta_quad_med": ci["point"], "delta_lo": ci["lo"], "delta_hi": ci["hi"]})
    if paired:
        curves = curves.merge(pd.DataFrame(paired), on="d", how="left")
    write_df(out / "aggregate_risk_curves.csv", curves, force=cfg.force)
    plat_rows = []
    for sid, sg in loc.groupby("sample_id"):
        sg = sg.sort_values("d")
        pq = plateau_from_curve(sg.d.to_numpy(), sg.quad_close_nmse.to_numpy(), sg.df_frac.to_numpy() if "df_frac" in sg.columns else None, thr)
        pl = plateau_from_curve(sg.d.to_numpy(), sg.lin_nmse.to_numpy(), None, thr)
        plat_rows.append({"sample_id": int(sid), "dQ": pq["d_plat"], "dL": pl["d_plat"]})
    write_df(out / "plateau_bootstrap.csv", pd.DataFrame(plat_rows), force=cfg.force)
    print(f"[qpd] aggregate n={len(loc)}", flush=True)


def stage_tail_adequacy(root: Path, cfg: QuadPredConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "tail_adequacy.csv"
    raw = pd.read_parquet(out / "per_anchor_metrics.parquet") if (out / "per_anchor_metrics.parquet").exists() else _load_all_cache(out)
    if not len(raw):
        pd.DataFrame().to_csv(path, index=False)
        return
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    loc = raw[raw.k == k_ref].groupby(["sample_id", "d"], as_index=False).mean(numeric_only=True)
    cols = [c for c in ["r2_T12", "r2_E4", "r2_U4", "r2_U8", "r2_outside", "r2_E4_fixed", "r2_E4_normal", "r2_E4_lin", "quad_close_r2"] if c in loc.columns]
    rows = []
    for d, g in loc.groupby("d"):
        rec = {"d": int(d), "k": int(k_ref), "n": int(len(g))}
        for c in cols:
            rec[f"{c}_med"] = float(g[c].median())
            rec[f"{c}_q25"] = float(g[c].quantile(0.25))
            rec[f"{c}_q75"] = float(g[c].quantile(0.75))
        rows.append(rec)
    write_df(path, pd.DataFrame(rows), force=cfg.force)
    thr = json.loads((out / "thresholds.json").read_text()) if (out / "thresholds.json").exists() else DEFAULT_THRESHOLDS
    sc_rows = []
    for k, gk in raw.groupby("k"):
        locg = gk.groupby(["sample_id", "d"], as_index=False).mean(numeric_only=True)
        for sid, sg in locg.groupby("sample_id"):
            pq = plateau_from_curve(sg.d.to_numpy(), sg.quad_close_nmse.to_numpy(), sg.df_frac.to_numpy() if "df_frac" in sg.columns else None, thr)
            sc_rows.append({"k": int(k), "sample_id": int(sid), "dQ": pq["d_plat"], "r2_best": float(sg.quad_close_r2.max())})
    write_df(out / "scale_sensitivity.csv", pd.DataFrame(sc_rows), force=cfg.force)
    print(f"[qpd] tail_adequacy d-rows={len(rows)}", flush=True)


def stage_synthetic_evaluation(root: Path, cfg: QuadPredConfig, ctx: dict, thr: dict) -> None:
    out = cfg.resolved(root)
    path = out / "synthetic_evaluation.csv"
    if _done(path, cfg.force):
        return
    seeds = split_seeds(cfg.n_synth_cal, cfg.n_synth_eval)
    n, D = (200, 20) if cfg.smoke else (480, 32)
    rows, dvals = [], []
    for kind in SYNTH_KINDS:
        for seed in seeds["evaluation_seeds"]:
            pack = make_predictive_synthetic(kind, n=n, D=D, seed=seed)
            est = _eval_synth(pack, cfg, thr)
            est.update({"seed": seed, "split": "evaluation", "err": abs(est.get("dQ", np.nan) - pack["true_d"])})
            rows.append(est)
            if np.isfinite(est.get("dQ", np.nan)):
                dvals.append(int(round(est["dQ"])))
    df = pd.DataFrame(rows)
    df["synth_not_only12"] = bool(len(set(dvals)) >= 2)
    write_df(path, df, force=cfg.force)
    print(f"[qpd] synthetic_evaluation n={len(df)} not_only12={bool(len(set(dvals)) >= 2)}", flush=True)


def _summary(out: Path, cfg: QuadPredConfig, thr: dict) -> dict[str, Any]:
    curves = pd.read_csv(out / "aggregate_risk_curves.csv") if (out / "aggregate_risk_curves.csv").exists() else pd.DataFrame()
    plat = pd.read_csv(out / "plateau_bootstrap.csv") if (out / "plateau_bootstrap.csv").exists() else pd.DataFrame()
    tail = pd.read_csv(out / "tail_adequacy.csv") if (out / "tail_adequacy.csv").exists() else pd.DataFrame()
    seval = pd.read_csv(out / "synthetic_evaluation.csv") if (out / "synthetic_evaluation.csv").exists() else pd.DataFrame()
    scale = pd.read_csv(out / "scale_sensitivity.csv") if (out / "scale_sensitivity.csv").exists() else pd.DataFrame()
    dQ = float(plat.dQ.median()) if len(plat) else float("nan")
    dL = float(plat.dL.median()) if len(plat) else float("nan")
    dQ_lo = float(plat.dQ.quantile(0.25)) if len(plat) else float("nan")
    dQ_hi = float(plat.dQ.quantile(0.75)) if len(plat) else float("nan")
    r2_lo = curves.r2_quad_lo.to_numpy() if len(curves) and "r2_quad_lo" in curves.columns else np.array([])
    ds = curves.d.to_numpy() if len(curves) else np.array([])
    adeq = adequacy_ranks(ds, r2_lo, thr) if len(ds) else {"d90": "not_reached", "d95": "not_reached", "d99": "not_reached"}
    r2tot = float(curves.r2_quad_med.max()) if len(curves) else float("nan")

    def tail_at(d, col):
        if not len(tail) or col not in tail.columns:
            return float("nan")
        hit = tail[tail.d == d]
        return float(hit.iloc[0][col]) if len(hit) else float("nan")

    def at(d, col):
        if not len(curves):
            return float("nan")
        hit = curves[curves.d == d]
        return float(hit.iloc[0][col]) if len(hit) and col in hit.columns else float("nan")

    r2e4, r2u8 = tail_at(12, "r2_E4_med"), tail_at(12, "r2_U8_med")
    r2e4_16, r2u8_16 = tail_at(16, "r2_E4_med"), tail_at(16, "r2_U8_med")
    not_only12 = bool(len(seval) and "synth_not_only12" in seval.columns and bool(seval.synth_not_only12.iloc[0]))
    scale_stable = True
    scale_meds = {}
    if len(scale) and scale.k.nunique() > 1:
        meds = scale.groupby("k").dQ.median()
        scale_meds = {int(k): float(v) for k, v in meds.items()}
        scale_stable = bool(float(meds.max() - meds.min()) <= 3.0)
    lab = primary_label(
        dQ=dQ, dL=dL, d95=adeq.get("d95"), r2_total=r2tot, r2_E4=r2e4, r2_U8=r2u8,
        delta_Q_12_16=at(12, "nmse_quad_med") - at(16, "nmse_quad_med"),
        delta_L_12_16=at(12, "nmse_lin_med") - at(16, "nmse_lin_med"),
        synth_not_only12=not_only12, scale_stable=scale_stable, thr=thr,
    )
    closest = seval.iloc[(seval.dQ - dQ).abs().argsort()[:1]] if len(seval) and np.isfinite(dQ) else pd.DataFrame()
    unexplained_now = float(1.0 - r2e4) if np.isfinite(r2e4) else float("nan")
    recovered_of_prior_unexplained = (
        float((r2e4 - 0.15) / 0.85) if np.isfinite(r2e4) else float("nan")
    )
    plat_set = []
    if len(plat):
        lo, hi = int(np.floor(dQ_lo)) if np.isfinite(dQ_lo) else None, int(np.ceil(dQ_hi)) if np.isfinite(dQ_hi) else None
        if lo is not None and hi is not None:
            plat_set = list(range(lo, hi + 1))
    return {
        "primary": lab, "dQ_plat": dQ, "dL_plat": dL, "dQ_iqr": [dQ_lo, dQ_hi],
        "dQ_plateau_set": plat_set,
        "d90": adeq.get("d90"), "d95": adeq.get("d95"), "d99": adeq.get("d99"),
        "r2_total_best": r2tot,
        "r2_total_d12": at(12, "r2_quad_med"), "r2_total_d16": at(16, "r2_quad_med"),
        "r2_total_pooled_best": float(curves.r2_quad_pooled.max()) if len(curves) and "r2_quad_pooled" in curves.columns else float("nan"),
        "r2_E4_d12": r2e4, "r2_U8_d12": r2u8, "r2_T12_d12": tail_at(12, "r2_T12_med"),
        "r2_U4_d12": tail_at(12, "r2_U4_med"),
        "r2_E4_d16": r2e4_16, "r2_U8_d16": r2u8_16,
        "r2_E4_best": float(tail.r2_E4_med.max()) if len(tail) and "r2_E4_med" in tail.columns else float("nan"),
        "r2_E4_fixed_d12": tail_at(12, "r2_E4_fixed_med"),
        "r2_E4_normal_d12": tail_at(12, "r2_E4_normal_med"),
        "delta_Q_12_16": at(12, "nmse_quad_med") - at(16, "nmse_quad_med"),
        "delta_L_12_16": at(12, "nmse_lin_med") - at(16, "nmse_lin_med"),
        "synth_not_only12": not_only12, "scale_stable": scale_stable, "scale_dQ_medians": scale_meds,
        "prior_E4_explained": 0.15, "prior_E4_unexplained": 0.85,
        "E4_unexplained_now": unexplained_now,
        "E4_recovered_of_prior_unexplained": recovered_of_prior_unexplained,
        "closest_synthetic": str(closest.kind.iloc[0]) if len(closest) else None,
        "n_plateau_anchors": int(len(plat)),
    }


def run(cfg: QuadPredConfig) -> dict[str, Any]:
    root = platonic_root()
    out = cfg.resolved(root)
    t0 = time.time()
    ctx = load_ctx(root, cfg)
    assert_not_preserved(out, root)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out / "logs" / "execution.log"
    log_fh = open(log_path, "a", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    import sys

    prev = sys.stdout
    sys.stdout = _Tee(prev, log_fh)
    try:
        return _run_inner(cfg, root, out, t0, ctx)
    finally:
        sys.stdout = prev
        log_fh.close()


def _run_inner(cfg: QuadPredConfig, root: Path, out: Path, t0: float, ctx: dict) -> dict[str, Any]:
    profile: dict[str, Any] = {"stages": {}, "completed": []}
    want = STAGES if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    if "all" in want:
        want = list(STAGES)
    run_set = set(want)
    if run_set & {"fit_primary", "scale_sensitivity", "aggregate", "tail_adequacy"}:
        run_set.update(["prepare"])
    if run_set & {"analyze", "report"}:
        run_set.update(["prepare", "parity"])

    def mark(name: str, dt: float) -> None:
        profile["stages"][f"{name}_s"] = dt
        if name not in profile["completed"]:
            profile["completed"].append(name)
        (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    if "prepare" in run_set:
        print("[qpd] stage=prepare", flush=True)
        t1 = time.time()
        stage_prepare(root, cfg, ctx)
        mark("prepare", time.time() - t1)

    parity: dict[str, Any] = {}
    if "parity" in run_set:
        print("[qpd] stage=parity", flush=True)
        t1 = time.time()
        try:
            parity = stage_parity(root, cfg, ctx)
        except RuntimeError:
            from .report import write_methods, write_report
            parity = json.loads((out / "parity.json").read_text()) if (out / "parity.json").exists() else {"ok": False}
            write_methods(out, cfg, ctx, parity, dict(DEFAULT_THRESHOLDS))
            write_report(out, cfg, ctx, parity, {"primary": "quadratic_predictive_dimension_unresolved", "parity_failed": True}, dict(DEFAULT_THRESHOLDS))
            raise
        mark("parity", time.time() - t1)
    elif (out / "parity.json").exists():
        parity = json.loads((out / "parity.json").read_text())

    thr = dict(DEFAULT_THRESHOLDS)
    if "synthetic_calibration" in run_set or cfg.stage == "all":
        print("[qpd] stage=synthetic_calibration", flush=True)
        t1 = time.time()
        thr = stage_synthetic_calibration(root, cfg, ctx)
        mark("synthetic_calibration", time.time() - t1)
    elif (out / "thresholds.json").exists():
        thr = json.loads((out / "thresholds.json").read_text())

    if "fit_primary" in run_set:
        print("[qpd] stage=fit_primary", flush=True)
        t1 = time.time()
        stage_fit_primary(root, cfg, ctx, t0, thr)
        mark("fit_primary", time.time() - t1)
    if "scale_sensitivity" in run_set:
        print("[qpd] stage=scale_sensitivity", flush=True)
        t1 = time.time()
        stage_scale_sensitivity(root, cfg, ctx, t0, thr)
        mark("scale_sensitivity", time.time() - t1)
    if "aggregate" in run_set:
        print("[qpd] stage=aggregate", flush=True)
        t1 = time.time()
        stage_aggregate(root, cfg, ctx, thr)
        mark("aggregate", time.time() - t1)
    if "tail_adequacy" in run_set:
        print("[qpd] stage=tail_adequacy", flush=True)
        t1 = time.time()
        stage_tail_adequacy(root, cfg, ctx)
        mark("tail_adequacy", time.time() - t1)
    if "synthetic_evaluation" in run_set or cfg.stage == "all":
        print("[qpd] stage=synthetic_evaluation", flush=True)
        t1 = time.time()
        stage_synthetic_evaluation(root, cfg, ctx, thr)
        mark("synthetic_evaluation", time.time() - t1)
    if "analyze" in run_set or "report" in run_set or cfg.stage == "all":
        print("[qpd] stage=analyze/report", flush=True)
        t1 = time.time()
        from .plots import write_figures
        from .report import write_methods, write_report
        labels = _summary(out, cfg, thr)
        (out / "summary.json").write_text(json.dumps(labels, indent=2, default=str))
        try:
            write_figures(out, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[qpd] figures failed: {e}", flush=True)
        if not parity and (out / "parity.json").exists():
            parity = json.loads((out / "parity.json").read_text())
        write_methods(out, cfg, ctx, parity, thr)
        write_report(out, cfg, ctx, parity, labels, thr)
        tmp = out / "COMPLETE.json.tmp"
        tmp.write_text(json.dumps({"ok": True, "primary": labels.get("primary"), "t": time.time()}, indent=2))
        tmp.replace(out / "COMPLETE.json")
        drift = {}
        for rel, rec in (parity.get("preserved_complete_hashes") or {}).items():
            from geometry.physics_activation_atlas.paths import resolve_path as _rp
            from .pipeline import _file_sha as _fs

            base = _rp(root, rel)
            if not isinstance(rec, dict):
                cpath = base / "COMPLETE.json"
                now = _fs(cpath) if cpath.exists() else None
                if now != rec:
                    drift[rel] = {"before": rec, "after": now}
                continue
            for name, old in rec.items():
                cpath = base / name
                now = None
                if cpath.exists():
                    now = {"sha": _fs(cpath), "mtime": cpath.stat().st_mtime, "size": cpath.stat().st_size}
                if old is None and now is None:
                    continue
                if old is None or now is None or old.get("sha") != now.get("sha") or old.get("size") != now.get("size"):
                    drift[f"{rel}/{name}"] = {"before": old, "after": now}
        if drift:
            raise RuntimeError(f"preserved completed outputs changed: {drift}")
        mark("analyze", time.time() - t1)
        mark("report", 0.0)
    profile["total_seconds"] = time.time() - t0
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[qpd] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
