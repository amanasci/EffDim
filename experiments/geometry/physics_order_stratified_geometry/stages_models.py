"""Model comparison, synthetics, associations, replication, and run()."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from geometry.physics_activation_atlas.full_curvature_audit import RIDGES, fit_quad
from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import load_model_X
from geometry.physics_activation_atlas.quadratic import quadratic_features
from geometry.physics_activation_atlas.sphere_normal_quadratic import NestedChart
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices
from geometry.physics_stable_tangent_dimension.dimension import paired_bootstrap_ci
from geometry.physics_stable_tangent_dimension.nested_pca import (
    nested_uncentred_svd,
    radial_stratified_halves,
)
from geometry.physics_stable_tangent_dimension.sphere_coords import angular_radii, rms_tangent_radius

from .algebra import (
    EPS,
    ambient_mse,
    intersection_rank,
    mix_shares,
    mixed_scale_nnls,
    normalize_rows,
    projector_overlap,
    refine_chart_coords,
    svd_quadratic_image,
    truncate_bs_left,
)
from .pipeline import (
    OrderStratConfig,
    _b_path,
    _budget_ok,
    _done,
    _j_path,
    displacements,
    ensure_neigh,
    load_ctx,
    platonic_root,
    stage_prepare,
)
from .rank import DEFAULT_Q_THRESHOLDS, classify_hypothesis, select_q2
from .stages import STAGES, stage_parity
from .stages_core import (
    _heldout_dS_curve,
    _null_energy,
    stage_carrier,
    stage_conditional_tail,
    stage_mixed_scaling,
    stage_odd_even,
    stage_quadratic_rank,
    stage_tail_overlap,
)
from .synthetics import SYNTH_KINDS, closest_synthetic, make_order_synthetic, split_seeds


def stage_model_comparison(root: Path, cfg: OrderStratConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "model_comparison.csv"
    if _done(Path(path), cfg.force):
        return
    X = ctx["X"]
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    rows = []
    d = cfg.d_core
    for sid in ctx["use_sids"]:
        ai = ctx["sid_to_ai"][int(sid)]
        jp, bp = _j_path(out, cfg.model, sid, k_ref), _b_path(out, cfg.model, sid, k_ref)
        if not jp.exists() or not bp.exists():
            continue
        J = np.load(jp)["J"]
        Bpack = np.load(bp)
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        N = ensure_neigh(ctx, ai, k_ref)
        Xloc = X[N].astype(np.float64)
        th = angular_radii(x0, Xloc)
        A, B = radial_stratified_halves(th, cfg.seed + ai)
        if min(len(A), len(B)) < d + 8 or J.shape[1] < cfg.d_ref:
            continue
        q2 = int(Bpack["q2"][0])
        U12 = (Xloc - x0) @ J[:, :d]
        U16 = (Xloc - x0) @ J[:, : cfg.d_ref]
        pred12 = normalize_rows(x0[None, :] + U12 @ J[:, :d].T)
        pred16 = normalize_rows(x0[None, :] + U16 @ J[:, : cfg.d_ref].T)
        e12 = ambient_mse(pred12[B], Xloc[B])
        e16 = ambient_mse(pred16[B], Xloc[B])
        chart = NestedChart(
            x0=x0,
            J=J[:, :d],
            A_flat=Bpack["AA"],
            BS_flat=Bpack["BSA"],
            ridge_A=float(Bpack["ridge_A"][0]),
            ridge_BS=float(Bpack["ridge_BS"][0]),
            coord_scale=np.ones(d),
        )
        rec = {
            "sample_id": int(sid),
            "k": int(k_ref),
            "q2": q2,
            "M12_linear": e12,
            "M16_linear": e16,
            "n_te": int(len(B)),
            "n_tr": int(len(A)),
        }
        for q in range(0, cfg.q_max + 1):
            chq = NestedChart(
                x0=chart.x0,
                J=chart.J,
                A_flat=chart.A_flat,
                BS_flat=truncate_bs_left(chart.BS_flat, d, q),
                ridge_A=chart.ridge_A,
                ridge_BS=chart.ridge_BS,
                coord_scale=chart.coord_scale,
            )
            rec[f"M12_quad_q{q}"] = ambient_mse(chq.decode_TRS(U12[B]), Xloc[B])
        rec["M12_quad_full"] = rec[f"M12_quad_q{cfg.q_max}"]
        rec["M12_quad_qhat"] = rec.get(f"M12_quad_q{q2}", rec["M12_quad_full"])
        # nonlinear encoder sensitivity on a subsample
        te = B[: min(64, len(B))]
        Uref = refine_chart_coords(chart, Xloc[te], U12[te], n_iter=3 if not cfg.smoke else 1)
        rec["M12_quad_qhat_nl"] = ambient_mse(chart.decode_TRS(Uref), Xloc[te])
        rec["delta_M16_minus_M12q"] = rec["M16_linear"] - rec["M12_quad_qhat"]
        rec["obs_per_param_qhat"] = float(len(A) / max(d + d * (d + 1) / 2 * max(q2, 1) / max(Xloc.shape[1], 1), 1))
        rows.append(rec)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[osg] model_comparison n={len(rows)}", flush=True)


def stage_normal_complement(root: Path, cfg: OrderStratConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "normal_complement_bounds.csv"
    if _done(Path(path), cfg.force):
        return
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    mix = pd.read_parquet(out / "mixed_scale_components.parquet") if (out / "mixed_scale_components.parquet").exists() else pd.DataFrame()
    rows = []
    for sid in ctx["use_sids"]:
        jp, bp = _j_path(out, cfg.model, sid, k_ref), _b_path(out, cfg.model, sid, k_ref)
        if not jp.exists() or not bp.exists():
            continue
        J = np.load(jp)["J"]
        Bpack = np.load(bp)
        R = min(cfg.R, J.shape[1])
        SR = J[:, :R]
        q2 = int(Bpack["q2"][0])
        UB = Bpack["UA"][:, : max(q2, 1)] if Bpack["UA"].shape[1] else Bpack["UA"]
        qN = intersection_rank(SR, UB, cos_min=0.7) if UB.shape[1] else 0
        qT = cfg.d_core
        qW = 0
        if len(mix):
            hit = mix[(mix.sample_id == sid) & (mix.series == "resid_E4")]
            if len(hit) and bool(hit.iloc[0].get("identifiable", False)) and float(hit.iloc[0].get("pi_thick", 0) or 0) >= 0.45:
                qW = 4
        qU = max(R - qT - qN - qW, 0)
        rows.append(
            {
                "sample_id": int(sid),
                "k": int(k_ref),
                "R": R,
                "q_T": qT,
                "q_N": qN,
                "q_W": qW,
                "q_U": qU,
                "q2": q2,
                "d1_minus": qT,
                "d1_plus": int(max(R - qN - qW, qT)),
                "d_le2": qT + q2,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[osg] normal_complement n={len(rows)}", flush=True)


def _eval_synth_pack(pack: dict, cfg: OrderStratConfig, thr: dict, device) -> dict[str, Any]:
    X, x0, neigh = pack["X"], pack["x0"], pack["neigh"]
    Xloc = X[neigh]
    k = len(neigh)
    d = min(cfg.d_core, 8 if cfg.smoke else cfg.d_core)
    Z = displacements(x0, Xloc, "log")
    J, _ev = nested_uncentred_svd(Z, max(d + 8, cfg.d_ref if not cfg.smoke else d + 4), device=device)
    th = angular_radii(x0, Xloc)
    A, B = radial_stratified_halves(th, pack.get("seed", 0))
    out = {
        "kind": pack["kind"],
        "true_d1": pack["true_d1"],
        "true_q2": pack["true_q2"],
        "median_q2": 0.0,
        "overlap_E4": float("nan"),
        "r2_quad_E4": float("nan"),
        "pi_quad": float("nan"),
        "pi_lin": float("nan"),
        "delta_M16_minus_M12q": float("nan"),
    }
    if min(len(A), len(B)) < d + 6 or J.shape[1] < d:
        return out
    fA, vA = _half_fit_indices(A, 1)
    chA, _, _ = fit_quad(Xloc, x0, J[:, :d], fA, vA, B, ridges=RIDGES, device=device)
    fB, vB = _half_fit_indices(B, 2)
    chB, _, _ = fit_quad(Xloc, x0, J[:, :d], fB, vB, A, ridges=RIDGES, device=device)
    if chA is None or chB is None:
        return out
    sA, sB = svd_quadratic_image(chA.BS_flat, d), svd_quadratic_image(chB.BS_flat, d)
    Uall = (Xloc - x0) @ J[:, :d]
    dS = _heldout_dS_curve(chA, Xloc, Uall, B, min(cfg.q_max, 6), d)
    from geometry.physics_activation_atlas.quadratic import quadratic_features as qf

    residA = qf(Uall[fA]) @ chA.BS_flat.T
    e_null = _null_energy(qf(Uall[fA]), residA, th[fA], max(4, cfg.n_null_draw // 2), 0)
    sel = select_q2(
        sA=sA["s"],
        sB=sB["s"],
        UA=sA["U"],
        UB=sB["U"],
        dS=dS[1 : min(cfg.q_max, 6) + 1],
        persist=np.ones(min(cfg.q_max, 6)),
        energy_null=e_null,
        thr=thr,
    )
    q2 = int(sel["q2"])
    out["median_q2"] = float(q2)
    if J.shape[1] >= d + 4 and sA["U"].shape[1] >= 1:
        out["overlap_E4"] = projector_overlap(J[:, d : d + 4], sA["U"][:, : min(4, sA["U"].shape[1])])
    U12 = Uall
    pred12 = normalize_rows(x0[None, :] + U12 @ J[:, :d].T)
    d16 = min(J.shape[1], d + 4)
    pred16 = normalize_rows(x0[None, :] + ((Xloc - x0) @ J[:, :d16]) @ J[:, :d16].T)
    chq = NestedChart(
        x0=chA.x0,
        J=chA.J,
        A_flat=chA.A_flat,
        BS_flat=truncate_bs_left(chA.BS_flat, d, q2),
        ridge_A=chA.ridge_A,
        ridge_BS=chA.ridge_BS,
        coord_scale=chA.coord_scale,
    )
    e12q = ambient_mse(chq.decode_TRS(U12[B]), Xloc[B])
    e16 = ambient_mse(pred16[B], Xloc[B])
    out["delta_M16_minus_M12q"] = float(e16 - e12q)
    out["M12_linear"] = ambient_mse(pred12[B], Xloc[B])
    return out


def stage_synthetic_calibration(root: Path, cfg: OrderStratConfig, ctx: dict) -> dict:
    out = cfg.resolved(root)
    path = out / "synthetic_calibration.csv"
    thr_path = out / "thresholds.json"
    if _done(Path(path), cfg.force) and _done(Path(thr_path), cfg.force):
        return json.loads(thr_path.read_text())
    thr = dict(DEFAULT_Q_THRESHOLDS)
    seeds = split_seeds(cfg.n_synth_cal, cfg.n_synth_eval)
    rows = []
    n, D, k_obs = (200, 32, 96) if cfg.smoke else (600, 48, 256)
    d_core = 6 if cfg.smoke else cfg.d_core
    for kind in SYNTH_KINDS:
        for seed in seeds["calibration_seeds"]:
            pack = make_order_synthetic(kind, n=n, D=D, seed=seed, k_obs=k_obs, d_core=d_core)
            pack["seed"] = seed
            row = _eval_synth_pack(pack, cfg, thr, ctx["device"])
            row["seed"] = seed
            row["split"] = "calibration"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    # freeze: require that curved_d12_q4 median q2 >= 2 and flat_d12 q2 <= 2 on calibration
    thr["frozen"] = True
    thr["cal_n"] = int(len(df))
    if len(df):
        c4 = df[df.kind == "curved_d12_q4"]
        flat = df[df.kind == "flat_d12"]
        if len(c4) and c4.median_q2.median() < 2:
            thr["dS_gain_min"] = float(min(float(thr["dS_gain_min"]), 1e-8))
            thr["mode_overlap_min"] = float(min(float(thr["mode_overlap_min"]), 0.35))
        if len(flat) and flat.median_q2.median() > 3:
            thr["mode_overlap_min"] = float(max(float(thr["mode_overlap_min"]), 0.55))
    thr_path.write_text(json.dumps(thr, indent=2))
    print(f"[osg] synthetic_calibration n={len(df)}", flush=True)
    return thr


def stage_synthetic_evaluation(root: Path, cfg: OrderStratConfig, ctx: dict, thr: dict) -> None:
    out = cfg.resolved(root)
    path = out / "synthetic_evaluation.csv"
    if _done(Path(path), cfg.force):
        return
    seeds = split_seeds(cfg.n_synth_cal, cfg.n_synth_eval)
    rows = []
    n, D, k_obs = (200, 32, 96) if cfg.smoke else (600, 48, 256)
    d_core = 6 if cfg.smoke else cfg.d_core
    for kind in SYNTH_KINDS:
        for seed in seeds["evaluation_seeds"]:
            pack = make_order_synthetic(kind, n=n, D=D, seed=seed, k_obs=k_obs, d_core=d_core)
            pack["seed"] = seed
            row = _eval_synth_pack(pack, cfg, thr, ctx["device"])
            row["seed"] = seed
            row["split"] = "evaluation"
            # confusion-style labels
            q2 = row["median_q2"]
            row["call_12q"] = bool(q2 >= 2 and (row["delta_M16_minus_M12q"] or 0) >= -1e-6)
            row["call_16"] = bool(q2 <= 1 and (row["delta_M16_minus_M12q"] or 0) < 0)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[osg] synthetic_evaluation n={len(rows)}", flush=True)


def stage_associations(root: Path, cfg: OrderStratConfig, ctx: dict) -> None:
    """Load OOF mag_r only after geometric ranks are frozen."""
    out = cfg.resolved(root)
    path = out / "probe_associations.csv"
    if _done(Path(path), cfg.force):
        return
    spec = out / "quadratic_spectrum.parquet"
    if not spec.exists():
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.read_parquet(spec)
    loc = df.drop_duplicates("sample_id")
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    geo = ctx["geo"][ctx["geo"].scale_k == k_ref][["sample_id", "local_r2"]]
    m = loc.merge(geo, on="sample_id", how="inner")
    rows = []
    cols = [c for c in ["q2", "cross_BF2", "A_K_dir", "A_K_H", "A_K_aniso", "sA"] if c in m.columns]
    pred = pd.read_parquet(out / "conditional_tail_prediction.parquet") if (out / "conditional_tail_prediction.parquet").exists() else pd.DataFrame()
    if len(pred):
        m = m.merge(pred[["sample_id", "resid_var_E4", "r2_E4"]], on="sample_id", how="left")
        cols += [c for c in ["resid_var_E4", "r2_E4"] if c in m.columns]
    for c in cols:
        if m[c].nunique(dropna=True) < 3:
            continue
        rho, p = spearmanr(m[c], m.local_r2, nan_policy="omit")
        rows.append({"metric": c, "rho_mag_r": float(rho), "p": float(p), "n": int(m[c].notna().sum()), "family_pass": bool(abs(float(rho)) >= 0.15 and p < 0.01)})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[osg] associations n={len(rows)}", flush=True)


def stage_replication(root: Path, cfg: OrderStratConfig, ctx0: dict, t0: float, thr: dict) -> None:
    out = cfg.resolved(root)
    path = out / "cross_model_order_dimensions.csv"
    if cfg.skip_replication or _done(Path(path), cfg.force):
        if cfg.skip_replication:
            print("[osg] replication skipped (ViT-B primary first)", flush=True)
        return
    rows = []
    locp = out / "normal_complement_bounds.csv"
    if locp.exists():
        nd = pd.read_csv(locp)
        qsum = pd.read_csv(out / "quadratic_rank_summary.csv") if (out / "quadratic_rank_summary.csv").exists() else pd.DataFrame()
        k_ref = cfg.primary_k
        qref = qsum[qsum.k == k_ref] if len(qsum) else pd.DataFrame()
        rows.append(
            {
                "model": cfg.model,
                "n": int(len(nd)),
                "d1_minus": float(nd.d1_minus.median()),
                "d1_plus": float(nd.d1_plus.median()),
                "q2": float(qref.iloc[0].median_q2) if len(qref) else float(nd.q2.median()),
                "d_le2": float(nd.d_le2.median()),
                "d_G": 12,
                "source": "primary_full",
            }
        )
    print("[osg] replication of comparison models is deferred until ViT-B report is complete", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _primary_label(out: Path) -> str:
    qsum = pd.read_csv(out / "quadratic_rank_summary.csv") if (out / "quadratic_rank_summary.csv").exists() else pd.DataFrame()
    ov = pd.read_parquet(out / "tail_quadratic_overlap.parquet") if (out / "tail_quadratic_overlap.parquet").exists() else pd.DataFrame()
    pred = pd.read_parquet(out / "conditional_tail_prediction.parquet") if (out / "conditional_tail_prediction.parquet").exists() else pd.DataFrame()
    mix = pd.read_parquet(out / "mixed_scale_components.parquet") if (out / "mixed_scale_components.parquet").exists() else pd.DataFrame()
    mc = pd.read_csv(out / "model_comparison.csv") if (out / "model_comparison.csv").exists() else pd.DataFrame()
    q2 = float(qsum.median_q2.iloc[0]) if len(qsum) else float("nan")
    overlap = float(ov.O_E4_B.median()) if len(ov) else float("nan")
    r2q = float(pred.r2_E4.median()) if len(pred) and "r2_E4" in pred.columns else float("nan")
    resid_lin = float(pred.resid_s0_E4.median()) if len(pred) and "resid_s0_E4" in pred.columns else float("nan")
    raw = mix[mix.series == "raw_E4"] if len(mix) else pd.DataFrame()
    pi_lin = float(raw.pi_lin.median()) if len(raw) and "pi_lin" in raw.columns else float("nan")
    pi_quad = float(raw.pi_quad.median()) if len(raw) and "pi_quad" in raw.columns else float("nan")
    pi_thick = float(raw.pi_thick.median()) if len(raw) and "pi_thick" in raw.columns else float("nan")
    mix_res = bool(len(raw) and raw.resolved.mean() > 0.3) if len(raw) and "resolved" in raw.columns else False
    delta = float(mc.delta_M16_minus_M12q.median()) if len(mc) and "delta_M16_minus_M12q" in mc.columns else float("nan")
    return classify_hypothesis(
        q2=q2,
        overlap_e4=overlap,
        r2_quad=r2q,
        residual_r2_linear=resid_lin if np.isfinite(resid_lin) else 0.0,
        pi_lin=pi_lin,
        pi_quad=pi_quad,
        pi_thick=pi_thick,
        m12_vs_m16=-delta if np.isfinite(delta) else float("nan"),
        mix_resolved=mix_res,
    )


def run(cfg: OrderStratConfig) -> dict[str, Any]:
    root = platonic_root()
    out = cfg.resolved(root)
    t0 = time.time()
    ctx = load_ctx(root, cfg)
    profile: dict[str, Any] = {"stages": {}, "completed": []}
    want = STAGES if cfg.stage == "all" else [s.strip() for s in cfg.stage.split(",")]
    if "all" in want:
        want = list(STAGES)
    run_set = set(want)
    if run_set & {"quadratic_rank", "tail_overlap", "conditional_tail", "mixed_scaling", "odd_even", "model_comparison"}:
        run_set.update(["prepare", "carrier"])
    if "quadratic_rank" in run_set:
        run_set.update(["carrier", "synthetic_calibration"])
    if run_set & {"analyze", "report"}:
        run_set.update(["prepare", "parity"])

    def mark(name: str, dt: float) -> None:
        profile["stages"][f"{name}_s"] = dt
        if name not in profile["completed"]:
            profile["completed"].append(name)
        (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))

    if "prepare" in run_set:
        t1 = time.time()
        print("[osg] stage=prepare", flush=True)
        stage_prepare(root, cfg, ctx)
        mark("prepare", time.time() - t1)

    parity: dict[str, Any] = {}
    if "parity" in run_set:
        t1 = time.time()
        print("[osg] stage=parity", flush=True)
        parity = stage_parity(root, cfg, ctx)
        mark("parity", time.time() - t1)
        if not parity.get("ok"):
            from .report import write_methods, write_report

            write_methods(out, cfg, ctx, parity, dict(DEFAULT_Q_THRESHOLDS))
            write_report(out, cfg, ctx, parity, {"primary": "order_stratification_unresolved", "parity_failed": True})
            raise RuntimeError("parity failed; see parity.json")
    elif (out / "parity.json").exists():
        parity = json.loads((out / "parity.json").read_text())

    if "carrier" in run_set:
        t1 = time.time()
        print("[osg] stage=carrier", flush=True)
        stage_carrier(root, cfg, ctx, t0)
        mark("carrier", time.time() - t1)

    thr = dict(DEFAULT_Q_THRESHOLDS)
    if "synthetic_calibration" in run_set:
        t1 = time.time()
        print("[osg] stage=synthetic_calibration", flush=True)
        thr = stage_synthetic_calibration(root, cfg, ctx)
        mark("synthetic_calibration", time.time() - t1)
    elif (out / "thresholds.json").exists():
        thr = json.loads((out / "thresholds.json").read_text())

    if "quadratic_rank" in run_set:
        t1 = time.time()
        print("[osg] stage=quadratic_rank", flush=True)
        stage_quadratic_rank(root, cfg, ctx, t0, thr)
        mark("quadratic_rank", time.time() - t1)

    if "tail_overlap" in run_set:
        t1 = time.time()
        print("[osg] stage=tail_overlap", flush=True)
        stage_tail_overlap(root, cfg, ctx)
        mark("tail_overlap", time.time() - t1)

    if "conditional_tail" in run_set:
        t1 = time.time()
        print("[osg] stage=conditional_tail", flush=True)
        stage_conditional_tail(root, cfg, ctx)
        mark("conditional_tail", time.time() - t1)

    if "mixed_scaling" in run_set:
        t1 = time.time()
        print("[osg] stage=mixed_scaling", flush=True)
        stage_mixed_scaling(root, cfg, ctx)
        mark("mixed_scaling", time.time() - t1)

    if "odd_even" in run_set:
        t1 = time.time()
        print("[osg] stage=odd_even", flush=True)
        stage_odd_even(root, cfg, ctx)
        mark("odd_even", time.time() - t1)

    if "model_comparison" in run_set:
        t1 = time.time()
        print("[osg] stage=model_comparison", flush=True)
        stage_model_comparison(root, cfg, ctx)
        mark("model_comparison", time.time() - t1)

    if "normal_complement" in run_set:
        t1 = time.time()
        print("[osg] stage=normal_complement", flush=True)
        stage_normal_complement(root, cfg, ctx)
        mark("normal_complement", time.time() - t1)

    if "synthetic_evaluation" in run_set:
        t1 = time.time()
        print("[osg] stage=synthetic_evaluation", flush=True)
        stage_synthetic_evaluation(root, cfg, ctx, thr)
        mark("synthetic_evaluation", time.time() - t1)

    if "associations" in run_set:
        t1 = time.time()
        print("[osg] stage=associations", flush=True)
        stage_associations(root, cfg, ctx)
        mark("associations", time.time() - t1)

    if "replication" in run_set:
        t1 = time.time()
        print("[osg] stage=replication", flush=True)
        stage_replication(root, cfg, ctx, t0, thr)
        mark("replication", time.time() - t1)

    if "analyze" in run_set or "report" in run_set or cfg.stage == "all":
        t1 = time.time()
        print("[osg] stage=analyze/report", flush=True)
        from .plots import write_figures
        from .report import write_methods, write_report

        labels = {"primary": _primary_label(out)}
        (out / "decision_labels.json").write_text(json.dumps(labels, indent=2))
        try:
            write_figures(out, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[osg] figures failed: {e}", flush=True)
        if not parity and (out / "parity.json").exists():
            parity = json.loads((out / "parity.json").read_text())
        write_methods(out, cfg, ctx, parity, thr)
        write_report(out, cfg, ctx, parity, labels)
        mark("analyze", time.time() - t1)
        mark("report", 0.0)

    profile["total_seconds"] = time.time() - t0
    (out / "runtime_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"[osg] done in {profile['total_seconds']:.1f}s completed={profile['completed']}", flush=True)
    return profile
