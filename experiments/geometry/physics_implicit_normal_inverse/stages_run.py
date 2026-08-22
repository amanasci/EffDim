"""Synthetic calibration/evaluation, associations, and the stage runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .classify import DEFAULT_THRESHOLDS, primary_label
from .pipeline import (
    ImplicitNormalConfig,
    _done,
    assert_not_preserved,
    classify_fit,
    dimension_from_labels,
    fit_constraints,
    implicit_q2_from_pack,
    load_ctx,
    platonic_root,
    scaling_for_directions,
    stage_prepare,
)
from .stages import STAGES, stage_parity
from .stages_core import (
    stage_carrier,
    stage_constraint_scaling,
    stage_dimension_bounds,
    stage_gauss_validation,
    stage_geometric_refine,
    stage_implicit_curvature,
    stage_linear_constraints,
    stage_normal_classification,
    stage_quadratic_constraints,
    stage_tail_analysis,
)
from .synthetics import SYNTH_KINDS, make_implicit_synthetic, split_seeds


def _synth_estimate(pack: dict[str, Any], cfg: ImplicitNormalConfig, thr: dict) -> dict[str, Any]:
    Y, rad = pack["Y"], pack["radii"]
    fit = fit_constraints(
        Y,
        rad,
        q_max=min(cfg.q_max, pack["R"]),
        seed=cfg.seed + 99,
        n_null=cfg.n_null_draw,
        refine_steps=0,
    )
    if not fit.get("ok"):
        return {
            "ok": False,
            "cN_hat": float("nan"),
            "d1_hat": float("nan"),
            "q2_hat": float("nan"),
            "kind": pack["kind"],
            "true_d1": pack["true_d1"],
            "true_cN": pack["true_cN"],
            "true_q2": pack["true_q2"],
        }
    sc = scaling_for_directions(Y, rad, fit["UA"], fit["h_pack"])
    labels = classify_fit(fit, sc, [float("nan")] * len(fit["dir_rows"]), thr)
    bounds = dimension_from_labels(labels, pack["R"], fit["ev_lin"], thr)
    q_use = min(max(int(bounds["cN_minus"]), 1), fit["UA"].shape[1])
    q2 = implicit_q2_from_pack(fit["UA"], fit["h_pack"], q_use)["q2"] if bounds["cN_minus"] else 0
    A = fit["UA"][:, :q_use] if q_use else np.zeros((pack["R"], 0))
    from .algebra import projector_overlap

    ovN = projector_overlap(A, pack["N"]) if pack["N"].size and A.shape[1] else float("nan")
    return {
        "ok": True,
        "cN_hat": bounds["cN_minus"],
        "d1_hat": bounds["d1_plus"],
        "d1_minus": bounds["d1_minus"],
        "q2_hat": int(q2),
        "n_flat": bounds["n_flat_prefix"],
        "n_curv": bounds["n_curv_prefix"],
        "overlap_N": ovN,
        "kind": pack["kind"],
        "true_d1": pack["true_d1"],
        "true_cN": pack["true_cN"],
        "true_q2": pack["true_q2"],
    }


def _threshold_score(rows: list[dict[str, Any]]) -> float:
    df = pd.DataFrame(rows)
    if not len(df):
        return float("-inf")
    d12 = df[df.kind.str.contains("d12") & ~df.kind.str.contains("isotropic")]
    d16 = df[df.kind.str.contains("d16")]
    iso = df[df.kind == "isotropic_carrier"]
    gap = 0.0
    if len(d12) and len(d16):
        gap = float(np.nanmean(d16.d1_hat) - np.nanmean(d12.d1_hat))
    iso_pen = 0.0
    if len(iso) and np.isfinite(iso.d1_hat).any() and float(np.nanmean(iso.d1_hat)) < 14:
        iso_pen = 5.0
    rec12 = float(np.mean(np.abs(d12.cN_hat - d12.true_cN) <= 2)) if len(d12) else 0.0
    rec16 = float(np.mean(np.abs(d16.cN_hat - d16.true_cN) <= 2)) if len(d16) else 0.0
    return gap + 2.0 * rec12 + 2.0 * rec16 - iso_pen


def stage_synthetic_calibration(root: Path, cfg: ImplicitNormalConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "synthetic_calibration.csv"
    tpath = out / "thresholds.json"
    if _done(path, cfg.force) and _done(tpath, cfg.force):
        return json.loads(tpath.read_text())
    seeds = split_seeds(cfg.n_synth_cal, cfg.n_synth_eval)
    n = 256 if cfg.smoke else 800
    R = 20 if not cfg.smoke else max(8, int(cfg.R))
    variants = [
        dict(DEFAULT_THRESHOLDS),
        {**DEFAULT_THRESHOLDS, "overlap_min": 0.30, "cancel_r2_min": 0.20, "flat_ratio_max": 0.45},
        {**DEFAULT_THRESHOLDS, "overlap_min": 0.50, "cancel_r2_min": 0.40, "flat_ratio_max": 0.25},
        {**DEFAULT_THRESHOLDS, "tangent_ratio_min": 0.45, "flat_ratio_max": 0.40},
    ]
    best_thr, best_score, best_rows = dict(DEFAULT_THRESHOLDS), float("-inf"), []
    for vi, thr in enumerate(variants):
        rows = []
        for kind in SYNTH_KINDS:
            for seed in seeds["calibration_seeds"]:
                pack = make_implicit_synthetic(kind, n=n, R=R, seed=seed, radius=0.12, noise=0.008)
                est = _synth_estimate(pack, cfg, thr)
                est["variant"] = vi
                est["seed"] = seed
                est["split"] = "calibration"
                rows.append(est)
        score = _threshold_score(rows)
        if score > best_score:
            best_score, best_thr, best_rows = score, dict(thr), rows
    pd.DataFrame(best_rows).to_csv(path, index=False)
    best_thr["calibration_score"] = best_score
    best_thr["synth_R"] = R
    tpath.write_text(json.dumps(best_thr, indent=2))
    print(f"[ini] synthetic_calibration score={best_score:.3f}", flush=True)
    return best_thr


def stage_synthetic_evaluation(root: Path, cfg: ImplicitNormalConfig, ctx: dict, thr: dict) -> None:
    out = cfg.resolved(root)
    path = out / "synthetic_evaluation.csv"
    if _done(path, cfg.force):
        return
    seeds = split_seeds(cfg.n_synth_cal, cfg.n_synth_eval)
    n = 256 if cfg.smoke else 800
    R = 20 if not cfg.smoke else max(8, int(cfg.R))
    rows = []
    d1_vals = []
    for kind in SYNTH_KINDS:
        for seed in seeds["evaluation_seeds"]:
            pack = make_implicit_synthetic(kind, n=n, R=R, seed=seed, radius=0.12, noise=0.008)
            est = _synth_estimate(pack, cfg, thr)
            est["seed"] = seed
            est["split"] = "evaluation"
            est["cN_err"] = abs(est.get("cN_hat", np.nan) - pack["true_cN"])
            est["d1_err"] = abs(est.get("d1_hat", np.nan) - pack["true_d1"])
            est["q2_err"] = abs(est.get("q2_hat", np.nan) - pack["true_q2"])
            est["call_12_8"] = bool(abs(est.get("d1_hat", 99) - 12) <= 2 and abs(est.get("cN_hat", 99) - 8) <= 2)
            est["call_16_4"] = bool(abs(est.get("d1_hat", 99) - 16) <= 2 and abs(est.get("cN_hat", 99) - 4) <= 2)
            rows.append(est)
            if np.isfinite(est.get("d1_hat", np.nan)):
                d1_vals.append(int(round(est["d1_hat"])))
    df = pd.DataFrame(rows)
    df["synth_not_only12"] = bool(len(set(d1_vals)) >= 2)
    df.to_csv(path, index=False)
    print(f"[ini] synthetic_evaluation n={len(df)} not_only12={bool(len(set(d1_vals)) >= 2)}", flush=True)


def stage_associations(root: Path, cfg: ImplicitNormalConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "probe_associations.csv"
    if _done(path, cfg.force):
        return
    proj = out / "normal_projectors.parquet"
    if not proj.exists():
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.read_parquet(proj)
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    loc = df[df.k == k_ref].drop_duplicates("sample_id")
    geo = ctx["geo"][ctx["geo"].scale_k == k_ref][["sample_id", "local_r2"]]
    m = loc.merge(geo, on="sample_id", how="inner")
    tailp = out / "tail_classification.parquet"
    if tailp.exists():
        m = m.merge(pd.read_parquet(tailp), on="sample_id", how="left", suffixes=("", "_tail"))
    curvp = out / "implicit_curvature_rank.csv"
    if curvp.exists():
        curv = pd.read_csv(curvp)
        c8 = curv[curv.q == 8][["sample_id", "q2"]].rename(columns={"q2": "q2_at_q8"}) if 8 in set(curv.q.tolist()) else pd.DataFrame()
        c4 = curv[curv.q == 4][["sample_id", "q2"]].rename(columns={"q2": "q2_at_q4"}) if 4 in set(curv.q.tolist()) else pd.DataFrame()
        if len(c8):
            m = m.merge(c8, on="sample_id", how="left")
        if len(c4):
            m = m.merge(c4, on="sample_id", how="left")
    rows = []
    cols = [
        c
        for c in [
            "cN_minus",
            "d1_plus",
            "d1_minus",
            "n_flat_prefix",
            "n_curv_prefix",
            "e4_normal_frac",
            "overlap_E4",
            "q2_at_q8",
            "q2_at_q4",
        ]
        if c in m.columns
    ]
    for c in cols:
        if m[c].nunique(dropna=True) < 3:
            continue
        rho, p = spearmanr(m[c], m.local_r2, nan_policy="omit")
        rows.append(
            {
                "metric": c,
                "rho_mag_r": float(rho),
                "p": float(p),
                "n": int(m[c].notna().sum()),
                "family_pass": bool(abs(float(rho)) >= 0.15 and p < 0.01),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[ini] associations n={len(rows)}", flush=True)


def _decision_labels(out: Path, cfg: ImplicitNormalConfig) -> dict[str, Any]:
    bounds = pd.read_csv(out / "dimension_bounds.csv") if (out / "dimension_bounds.csv").exists() else pd.DataFrame()
    tail = pd.read_parquet(out / "tail_classification.parquet") if (out / "tail_classification.parquet").exists() else pd.DataFrame()
    seval = pd.read_csv(out / "synthetic_evaluation.csv") if (out / "synthetic_evaluation.csv").exists() else pd.DataFrame()
    curv = pd.read_csv(out / "implicit_curvature_rank.csv") if (out / "implicit_curvature_rank.csv").exists() else pd.DataFrame()
    b = None
    if len(bounds):
        hit = bounds[bounds.k == cfg.primary_k] if "k" in bounds.columns else bounds
        b = hit.iloc[-1] if len(hit) else bounds.iloc[-1]
    cNm = float(b.median_cN_minus) if b is not None else float("nan")
    d1p = float(b.median_d1_plus) if b is not None else float("nan")
    d1m = float(b.median_d1_minus) if b is not None else float("nan")
    e4n = float(tail.e4_normal_frac.median()) if len(tail) and "e4_normal_frac" in tail.columns else float("nan")
    q2 = float("nan")
    if len(curv):
        cq = curv[curv.q == 8] if 8 in set(curv.q.tolist()) else curv
        q2 = float(cq.q2.median()) if len(cq) else float("nan")
    not_only12 = bool(len(seval) and bool(seval.synth_not_only12.iloc[0])) if len(seval) and "synth_not_only12" in seval.columns else False
    lab = primary_label(
        cN_minus=cNm,
        d1_minus=d1m,
        d1_plus=d1p,
        q2=q2,
        R=cfg.R,
        e4_normal_frac=e4n,
        synth_not_only12=not_only12,
    )
    return {
        "primary": lab,
        "median_cN_minus": cNm,
        "median_d1_plus": d1p,
        "median_d1_minus": d1m,
        "median_q2_at_q8": q2,
        "e4_normal_frac": e4n,
        "synth_not_only12": not_only12,
    }


def run(cfg: ImplicitNormalConfig) -> dict[str, Any]:
    root = platonic_root()
    out = cfg.resolved(root)
    t0 = time.time()
    ctx = load_ctx(root, cfg)
    assert_not_preserved(out, root)
    profile: dict[str, Any] = {"stages": {}, "completed": []}
    want = STAGES if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    if "all" in want:
        want = list(STAGES)
    run_set = set(want)
    heavy = {
        "linear_constraints",
        "quadratic_constraints",
        "geometric_refine",
        "constraint_scaling",
        "normal_classification",
        "dimension_bounds",
        "implicit_curvature",
        "tail_analysis",
        "gauss_validation",
    }
    if run_set & heavy:
        run_set.update(["prepare", "carrier"])
    if run_set & {"analyze", "report"}:
        run_set.update(["prepare", "parity"])

    def mark(name: str, dt: float) -> None:
        profile["stages"][f"{name}_s"] = dt
        if name not in profile["completed"]:
            profile["completed"].append(name)
        (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    if "prepare" in run_set:
        t1 = time.time()
        print("[ini] stage=prepare", flush=True)
        stage_prepare(root, cfg, ctx)
        mark("prepare", time.time() - t1)

    parity: dict[str, Any] = {}
    if "parity" in run_set:
        t1 = time.time()
        print("[ini] stage=parity", flush=True)
        parity = stage_parity(root, cfg, ctx)
        mark("parity", time.time() - t1)
        if not parity.get("ok"):
            from .report import write_methods, write_report

            write_methods(out, cfg, ctx, parity, dict(DEFAULT_THRESHOLDS))
            write_report(
                out, cfg, ctx, parity, {"primary": "implicit_normal_inverse_unresolved", "parity_failed": True}
            )
            raise RuntimeError("parity failed; see parity.json")
    elif (out / "parity.json").exists():
        parity = json.loads((out / "parity.json").read_text())

    if "carrier" in run_set:
        t1 = time.time()
        print("[ini] stage=carrier", flush=True)
        stage_carrier(root, cfg, ctx, t0)
        mark("carrier", time.time() - t1)

    if "linear_constraints" in run_set:
        t1 = time.time()
        print("[ini] stage=linear_constraints", flush=True)
        stage_linear_constraints(root, cfg, ctx, t0)
        mark("linear_constraints", time.time() - t1)

    if "quadratic_constraints" in run_set:
        t1 = time.time()
        print("[ini] stage=quadratic_constraints", flush=True)
        stage_quadratic_constraints(root, cfg, ctx, t0)
        mark("quadratic_constraints", time.time() - t1)

    if "geometric_refine" in run_set:
        t1 = time.time()
        print("[ini] stage=geometric_refine", flush=True)
        stage_geometric_refine(root, cfg, ctx, t0)
        mark("geometric_refine", time.time() - t1)

    thr = dict(DEFAULT_THRESHOLDS)
    if "synthetic_calibration" in run_set or cfg.stage == "all":
        t1 = time.time()
        print("[ini] stage=synthetic_calibration", flush=True)
        thr = stage_synthetic_calibration(root, cfg, ctx)
        mark("synthetic_calibration", time.time() - t1)
    elif (out / "thresholds.json").exists():
        thr = json.loads((out / "thresholds.json").read_text())

    if "constraint_scaling" in run_set:
        t1 = time.time()
        print("[ini] stage=constraint_scaling", flush=True)
        stage_constraint_scaling(root, cfg, ctx, t0)
        mark("constraint_scaling", time.time() - t1)

    if "normal_classification" in run_set:
        t1 = time.time()
        print("[ini] stage=normal_classification", flush=True)
        stage_normal_classification(root, cfg, ctx, thr)
        mark("normal_classification", time.time() - t1)

    if "dimension_bounds" in run_set:
        t1 = time.time()
        print("[ini] stage=dimension_bounds", flush=True)
        stage_dimension_bounds(root, cfg, ctx)
        mark("dimension_bounds", time.time() - t1)

    if "implicit_curvature" in run_set:
        t1 = time.time()
        print("[ini] stage=implicit_curvature", flush=True)
        stage_implicit_curvature(root, cfg, ctx)
        mark("implicit_curvature", time.time() - t1)

    if "tail_analysis" in run_set:
        t1 = time.time()
        print("[ini] stage=tail_analysis", flush=True)
        stage_tail_analysis(root, cfg, ctx)
        mark("tail_analysis", time.time() - t1)

    if "gauss_validation" in run_set:
        t1 = time.time()
        print("[ini] stage=gauss_validation", flush=True)
        stage_gauss_validation(root, cfg, ctx)
        mark("gauss_validation", time.time() - t1)

    if "synthetic_evaluation" in run_set or cfg.stage == "all":
        t1 = time.time()
        print("[ini] stage=synthetic_evaluation", flush=True)
        stage_synthetic_evaluation(root, cfg, ctx, thr)
        mark("synthetic_evaluation", time.time() - t1)

    if "associations" in run_set:
        t1 = time.time()
        print("[ini] stage=associations", flush=True)
        stage_associations(root, cfg, ctx)
        mark("associations", time.time() - t1)

    if "analyze" in run_set or "report" in run_set or cfg.stage == "all":
        t1 = time.time()
        print("[ini] stage=analyze/report", flush=True)
        from .plots import write_figures
        from .report import write_methods, write_report

        labels = _decision_labels(out, cfg)
        (out / "decision_labels.json").write_text(json.dumps(labels, indent=2))
        try:
            write_figures(out, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[ini] figures failed: {e}", flush=True)
        if not parity and (out / "parity.json").exists():
            parity = json.loads((out / "parity.json").read_text())
        write_methods(out, cfg, ctx, parity, thr)
        write_report(out, cfg, ctx, parity, labels)
        mark("analyze", time.time() - t1)
        mark("report", 0.0)

    profile["total_seconds"] = time.time() - t0
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[ini] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
