"""Carrier through odd/even stages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from geometry.physics_activation_atlas.effdim_curvature_metrics import (
    cross_metric_pair,
    metric_scalars,
)
from geometry.physics_activation_atlas.full_curvature_audit import RIDGES, fit_quad
from geometry.physics_activation_atlas.quadratic import quadratic_features
from geometry.physics_activation_atlas.sphere_normal_quadratic import NestedChart, chart_errors
from geometry.physics_activation_atlas.split_half_curvature_reliability import _half_fit_indices
from geometry.physics_activation_atlas.tangent_reliability import principal_angles
from geometry.physics_stable_tangent_dimension.dimension import paired_bootstrap_ci
from geometry.physics_stable_tangent_dimension.nested_pca import (
    nested_uncentred_svd,
    radial_stratified_halves,
)
from geometry.physics_stable_tangent_dimension.sphere_coords import angular_radii, rms_tangent_radius

from .algebra import (
    EPS,
    cross_frobenius,
    fit_quadratic_map,
    intersection_rank,
    mix_shares,
    mixed_scale_nnls,
    odd_even_displacements,
    pair_antipodes,
    pca_subspace,
    per_col_r2,
    predict_quadratic_map,
    projector_overlap,
    r2_score,
    svd_quadratic_image,
    truncate_bs_left,
    whiten_tangent,
)
from .pipeline import (
    OrderStratConfig,
    _b_path,
    _budget_ok,
    _done,
    _j_path,
    displacements,
    ensure_neigh,
)
from .rank import select_q2


def stage_carrier(root: Path, cfg: OrderStratConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    marker = out / "carrier_done.json"
    if _done(marker, cfg.force):
        return
    d_max = max([cfg.R, cfg.d_ref, *cfg.R_sens])
    X = ctx["X"]
    n_done = 0
    for i, sid in enumerate(ctx["use_sids"]):
        if not _budget_ok(t0, cfg, reserve=True):
            break
        ai = ctx["sid_to_ai"][int(sid)]
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        for k in ctx["ks"]:
            jp = _j_path(out, cfg.model, sid, k)
            if jp.exists() and not cfg.force:
                continue
            N = ensure_neigh(ctx, ai, k)
            Xloc = X[N].astype(np.float64)
            Z = displacements(x0, Xloc, cfg.coord)
            J, ev = nested_uncentred_svd(Z, d_max, device=ctx["device"])
            th = angular_radii(x0, Xloc)
            np.savez_compressed(jp, J=J, ev=ev, theta=th, rms=np.array([rms_tangent_radius(Z)]))
        n_done += 1
        if i % 32 == 0:
            print(f"[osg][carrier] {i}/{len(ctx['use_sids'])}", flush=True)
    marker.write_text(json.dumps({"n": n_done, "d_max": d_max, "ks": ctx["ks"]}))
    print(f"[osg] carrier n={n_done}", flush=True)


def _null_energy(Phi: np.ndarray, resid: np.ndarray, radii: np.ndarray, n_draw: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(resid)
    if n < 16 or Phi.shape[0] != n:
        return 0.0
    order = np.argsort(radii)
    bins = np.array_split(order, min(8, max(2, n // 16)))
    emax = []
    G = Phi.T @ Phi + 1e-3 * np.eye(Phi.shape[1])
    for _ in range(n_draw):
        perm = np.arange(n)
        for b in bins:
            if len(b) > 1:
                perm[b] = rng.permutation(b)
        try:
            B = np.linalg.solve(G, Phi.T @ resid[perm]).T
        except np.linalg.LinAlgError:
            continue
        s = np.linalg.svd(B, compute_uv=False)
        emax.append(float(s[0] ** 2) if len(s) else 0.0)
    return float(np.quantile(emax, 0.99)) if emax else 0.0


def _heldout_dS_curve(chart: NestedChart, Xloc: np.ndarray, U: np.ndarray, te: np.ndarray, q_max: int, d: int) -> np.ndarray:
    w = np.ones(len(Xloc))
    E_TR = chart_errors(chart, chart, Xloc, U, w, te)["E_TR"]
    out = np.full(q_max + 1, np.nan)
    for q in range(0, q_max + 1):
        ch = NestedChart(
            x0=chart.x0,
            J=chart.J,
            A_flat=chart.A_flat,
            BS_flat=truncate_bs_left(chart.BS_flat, d, q),
            ridge_A=chart.ridge_A,
            ridge_BS=chart.ridge_BS,
            coord_scale=chart.coord_scale,
        )
        out[q] = float(E_TR - chart_errors(ch, ch, Xloc, U, w, te)["E_TRS"])
    return out


def stage_quadratic_rank(root: Path, cfg: OrderStratConfig, ctx: dict, t0: float, thr: dict) -> None:
    out = cfg.resolved(root)
    path = out / "quadratic_spectrum.parquet"
    sum_path = out / "quadratic_rank_summary.csv"
    if _done(path, cfg.force) and _done(Path(sum_path), cfg.force):
        return
    X = ctx["X"]
    rows = []
    q_max, d = cfg.q_max, cfg.d_core
    for i, sid in enumerate(ctx["use_sids"]):
        if not _budget_ok(t0, cfg, reserve=True):
            break
        ai = ctx["sid_to_ai"][int(sid)]
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        for k in ctx["ks_quad"]:
            jp = _j_path(out, cfg.model, sid, k)
            if not jp.exists():
                continue
            z = np.load(jp)
            J = z["J"]
            if J.shape[1] < cfg.d_ref:
                continue
            N = ensure_neigh(ctx, ai, k)
            Xloc = X[N].astype(np.float64)
            th = z["theta"] if "theta" in z.files else angular_radii(x0, Xloc)
            A, Bidx = radial_stratified_halves(th, cfg.seed + ai + 13 * k)
            if min(len(A), len(Bidx)) < d + 8:
                continue
            fA, vA = _half_fit_indices(A, cfg.seed + 3 * ai)
            fB, vB = _half_fit_indices(Bidx, cfg.seed + 5 * ai + 1)
            Jc = J[:, :d]
            chA, _, infoA = fit_quad(Xloc, x0, Jc, fA, vA, Bidx, ridges=RIDGES, device=ctx["device"])
            chB, _, _infoB = fit_quad(Xloc, x0, Jc, fB, vB, A, ridges=RIDGES, device=ctx["device"])
            if chA is None or chB is None:
                continue
            sA, sB = svd_quadratic_image(chA.BS_flat, d), svd_quadratic_image(chB.BS_flat, d)
            UA, UB, svA, svB = sA["U"], sB["U"], sA["s"], sB["s"]
            Uall = (Xloc - x0) @ Jc
            dS = _heldout_dS_curve(chA, Xloc, Uall, Bidx, q_max, d)
            residA = quadratic_features(Uall[fA]) @ chA.BS_flat.T
            e_null = _null_energy(quadratic_features(Uall[fA]), residA, th[fA], cfg.n_null_draw, cfg.seed + ai)
            sel = select_q2(sA=svA, sB=svB, UA=UA, UB=UB, dS=dS[1 : q_max + 1], persist=np.ones(q_max), energy_null=e_null, thr=thr)
            q2 = int(sel["q2"])
            sc = metric_scalars(chA.BS_flat, d)
            cm = cross_metric_pair(chA.BS_flat, chB.BS_flat, d)
            np.savez_compressed(
                _b_path(out, cfg.model, sid, k),
                BSA=chA.BS_flat,
                BSB=chB.BS_flat,
                AA=chA.A_flat,
                AB=chB.A_flat,
                ridge_A=np.array([chA.ridge_A, chB.ridge_A]),
                ridge_BS=np.array([chA.ridge_BS, chB.ridge_BS]),
                sA=svA,
                sB=svB,
                UA=UA[:, : min(8, UA.shape[1])],
                UB=UB[:, : min(8, UB.shape[1])],
                q2=np.array([q2]),
            )
            nq = min(q_max, len(svA), len(svB))
            for q in range(nq):
                ov = projector_overlap(UA[:, : q + 1], UB[:, : q + 1]) if min(UA.shape[1], UB.shape[1]) > q else float("nan")
                rows.append(
                    {
                        "sample_id": int(sid),
                        "k": int(k),
                        "q": q + 1,
                        "sA": float(svA[q]),
                        "sB": float(svB[q]),
                        "overlap_prefix": ov,
                        "dS": float(dS[q + 1]) if q + 1 < len(dS) else float("nan"),
                        "dS_inc": float(sel["gains"][q]) if q < len(sel.get("gains", [])) else float("nan"),
                        "accepted": bool(sel["flags"][q]) if q < len(sel["flags"]) else False,
                        "q2": q2,
                        "energy_null": e_null,
                        "cross_BF2": cross_frobenius(chA.BS_flat, chB.BS_flat),
                        "dS_full": infoA.get("dS", float("nan")),
                        **{f"A_{kk}": vv for kk, vv in sc.items()},
                        **cm,
                    }
                )
        if i % 16 == 0:
            print(f"[osg][quad] {i}/{len(ctx['use_sids'])}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    sums = []
    if len(df):
        for k, g in df.groupby("k"):
            loc = g.drop_duplicates("sample_id")
            qv = loc.q2.to_numpy(float)
            med = paired_bootstrap_ci(qv, seed=cfg.seed)
            sums.append(
                {
                    "k": int(k),
                    "n": int(len(qv)),
                    "median_q2": med["point"],
                    "median_lo": med["lo"],
                    "median_hi": med["hi"],
                    "iqr": float(np.subtract(*np.quantile(qv, [0.75, 0.25]))),
                    **{f"p_ge_{q}": float(np.mean(qv >= q)) for q in range(1, cfg.q_max + 1)},
                }
            )
    pd.DataFrame(sums).to_csv(sum_path, index=False)
    print(f"[osg] quadratic_rank n={len(df)}", flush=True)


def _load_JB(out: Path, cfg: OrderStratConfig, sid: int, k: int):
    jp, bp = _j_path(out, cfg.model, sid, k), _b_path(out, cfg.model, sid, k)
    if not jp.exists() or not bp.exists():
        return None, None
    return np.load(jp), dict(np.load(bp))


def stage_tail_overlap(root: Path, cfg: OrderStratConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "tail_quadratic_overlap.parquet"
    if _done(path, cfg.force):
        return
    X = ctx["X"]
    rows = []
    d, R = cfg.d_core, cfg.R
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    for sid in ctx["use_sids"]:
        ai = ctx["sid_to_ai"][int(sid)]
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        zJ, Bpack = _load_JB(out, cfg, sid, k_ref)
        if zJ is None:
            continue
        J = zJ["J"]
        if J.shape[1] < cfg.d_ref:
            continue
        N = ensure_neigh(ctx, ai, k_ref)
        Xloc = X[N].astype(np.float64)
        Z = displacements(x0, Xloc, cfg.coord)
        th = angular_radii(x0, Xloc)
        A, B = radial_stratified_halves(th, cfg.seed + ai)
        JA, _ = nested_uncentred_svd(Z[A], max(R, cfg.d_ref), device=ctx["device"])
        JB, _ = nested_uncentred_svd(Z[B], max(R, cfg.d_ref), device=ctx["device"])
        q2 = int(Bpack["q2"][0])
        UA, UB = Bpack["UA"], Bpack["UB"]
        qU = min(max(q2, 4), UA.shape[1], UB.shape[1])
        E4A = JA[:, d:cfg.d_ref] if JA.shape[1] >= cfg.d_ref else JA[:, d:]
        E4B = JB[:, d:cfg.d_ref] if JB.shape[1] >= cfg.d_ref else JB[:, d:]
        ERA = JA[:, d:R] if JA.shape[1] >= min(R, JA.shape[1]) else JA[:, d:]
        ERB = JB[:, d:R] if JB.shape[1] >= min(R, JB.shape[1]) else JB[:, d:]
        o_ab = projector_overlap(E4A[:, : min(4, E4A.shape[1])], UB[:, :qU]) if E4A.shape[1] and qU else float("nan")
        o_ba = projector_overlap(E4B[:, : min(4, E4B.shape[1])], UA[:, :qU]) if E4B.shape[1] and qU else float("nan")
        ang = principal_angles(E4A[:, : min(4, E4A.shape[1])], UB[:, :qU]) if E4A.shape[1] and qU else np.zeros(0)
        rows.append(
            {
                "sample_id": int(sid),
                "k": int(k_ref),
                "q2": q2,
                "O_E4_B": 0.5 * (o_ab + o_ba) if np.isfinite(o_ab) and np.isfinite(o_ba) else float("nan"),
                "mean_cos_E4": float(np.mean(np.cos(ang))) if len(ang) else float("nan"),
                "intersect_E4": 0.5
                * (
                    intersection_rank(E4A[:, : min(4, E4A.shape[1])], UB[:, :qU])
                    + intersection_rank(E4B[:, : min(4, E4B.shape[1])], UA[:, :qU])
                )
                if qU
                else 0.0,
                "O_ER_B": 0.5
                * (
                    projector_overlap(ERA, UB[:, : min(UB.shape[1], max(ERA.shape[1], 1))])
                    + projector_overlap(ERB, UA[:, : min(UA.shape[1], max(ERB.shape[1], 1))])
                )
                if ERA.shape[1] and ERB.shape[1]
                else float("nan"),
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[osg] tail_overlap n={len(rows)}", flush=True)


def _conditional_one(Z, J, d, d_ref, R, A, B, ridge=1e-2) -> dict[str, float]:
    Jc = J[:, :d]
    E4 = J[:, d:d_ref] if J.shape[1] >= d_ref else J[:, d : d + 4]
    ER = J[:, d:R] if J.shape[1] >= min(R, J.shape[1]) else J[:, d:]
    Utr, sc = whiten_tangent(Z[A] @ Jc)
    Ute, _ = whiten_tangent(Z[B] @ Jc, sc)
    out: dict[str, float] = {}
    for name, E in (("E4", E4), ("ER", ER)):
        if E.shape[1] == 0:
            continue
        Ytr, Yte = Z[A] @ E, Z[B] @ E
        Yhat = predict_quadratic_map(Ute, fit_quadratic_map(Utr, Ytr, ridge))
        resid = Yte - Yhat
        tot = float(np.mean(np.sum(Yte * Yte, axis=1)))
        out[f"r2_{name}"] = r2_score(Yte, Yhat)
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(np.sum((Yte - Yhat) ** 2, axis=1)) / max(Yte.shape[1], 1)))
        out[f"pred_norm_{name}"] = float(np.sqrt(np.mean(np.sum(Yhat * Yhat, axis=1))))
        den = np.maximum(np.linalg.norm(Yte, axis=1) * np.linalg.norm(Yhat, axis=1), EPS)
        out[f"cos_{name}"] = float(np.nanmean(np.sum(Yte * Yhat, axis=1) / den))
        out[f"resid_var_{name}"] = float(np.mean(np.sum(resid * resid, axis=1)))
        out[f"resid_frac_{name}"] = float(out[f"resid_var_{name}"] / max(tot, EPS))
        out[f"r2_{name}_mean_dir"] = float(np.nanmean(per_col_r2(Yte, Yhat)))
        if len(resid) > E.shape[1] + 2:
            _, s, _ = np.linalg.svd(resid, full_matrices=False)
            out[f"resid_s0_{name}"] = float(s[0] ** 2 / max(len(resid), 1)) if len(s) else float("nan")
        else:
            out[f"resid_s0_{name}"] = float("nan")
    return out


def stage_conditional_tail(root: Path, cfg: OrderStratConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    pred_p = out / "conditional_tail_prediction.parquet"
    res_p = out / "conditional_tail_residual.parquet"
    if _done(pred_p, cfg.force) and _done(Path(res_p), cfg.force):
        return
    X = ctx["X"]
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    pred_rows, res_rows = [], []
    for sid in ctx["use_sids"]:
        ai = ctx["sid_to_ai"][int(sid)]
        jp = _j_path(out, cfg.model, sid, k_ref)
        if not jp.exists():
            continue
        J = np.load(jp)["J"]
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        N = ensure_neigh(ctx, ai, k_ref)
        Xloc = X[N].astype(np.float64)
        Z = displacements(x0, Xloc, cfg.coord)
        th = angular_radii(x0, Xloc)
        A, B = radial_stratified_halves(th, cfg.seed + ai)
        if min(len(A), len(B)) < cfg.d_core + 8:
            continue
        m = _conditional_one(Z, J, cfg.d_core, cfg.d_ref, cfg.R, A, B)
        m2 = _conditional_one(Z, J, cfg.d_core, cfg.d_ref, cfg.R, B, A)
        row = {"sample_id": int(sid), "k": int(k_ref)}
        for key in set(m) | set(m2):
            row[key] = 0.5 * (m.get(key, np.nan) + m2.get(key, np.nan))
        pred_rows.append(row)
        res_rows.append(
            {
                "sample_id": int(sid),
                "k": int(k_ref),
                "resid_frac_E4": row.get("resid_frac_E4", np.nan),
                "resid_s0_E4": row.get("resid_s0_E4", np.nan),
                "resid_frac_ER": row.get("resid_frac_ER", np.nan),
                "resid_s0_ER": row.get("resid_s0_ER", np.nan),
            }
        )
    pd.DataFrame(pred_rows).to_parquet(pred_p, index=False)
    pd.DataFrame(res_rows).to_parquet(res_p, index=False)
    print(f"[osg] conditional_tail n={len(pred_rows)}", flush=True)


def stage_mixed_scaling(root: Path, cfg: OrderStratConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "mixed_scale_components.parquet"
    if _done(path, cfg.force):
        return
    X = ctx["X"]
    rows = []
    for sid in ctx["use_sids"]:
        ai = ctx["sid_to_ai"][int(sid)]
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        recs = []
        for k in ctx["ks"]:
            jp = _j_path(out, cfg.model, sid, k)
            if not jp.exists():
                continue
            J = np.load(jp)["J"]
            N = ensure_neigh(ctx, ai, k)
            Z = displacements(x0, X[N].astype(np.float64), cfg.coord)
            if J.shape[1] < cfg.d_ref:
                continue
            E4 = J[:, cfg.d_core : cfg.d_ref]
            e_raw = float(np.mean(np.sum((Z @ E4) ** 2, axis=1)))
            pred_e = float("nan")
            bp = _b_path(out, cfg.model, sid, k)
            if bp.exists():
                Bpack = np.load(bp)
                Uw, _ = whiten_tangent(Z @ J[:, : cfg.d_core])
                Qn = quadratic_features(Uw) @ Bpack["BSA"].T
                pred_e = float(np.mean(np.sum((Qn @ E4) ** 2, axis=1)))
            recs.append(
                {
                    "r": rms_tangent_radius(Z),
                    "raw": e_raw,
                    "pred": pred_e,
                    "resid": e_raw - pred_e if np.isfinite(pred_e) else np.nan,
                }
            )
        if len(recs) < 4:
            continue
        rdf = pd.DataFrame(recs)
        for col, lab in (("raw", "raw_E4"), ("pred", "pred_E4"), ("resid", "resid_E4")):
            mix = mixed_scale_nnls(rdf.r.to_numpy(), rdf[col].to_numpy())
            r_ref = float(rdf.r.iloc[-1])
            sh = (
                mix_shares(mix["a"], mix["b"], mix["c"], r_ref)
                if mix["identifiable"]
                else {"pi_lin": np.nan, "pi_quad": np.nan, "pi_thick": np.nan}
            )
            rows.append({"sample_id": int(sid), "series": lab, **mix, **sh, "r_ref": r_ref})
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[osg] mixed_scaling n={len(rows)}", flush=True)


def stage_odd_even(root: Path, cfg: OrderStratConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "odd_even_diagnostics.parquet"
    if _done(path, cfg.force):
        return
    X = ctx["X"]
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    rows = []
    for sid in ctx["use_sids"]:
        ai = ctx["sid_to_ai"][int(sid)]
        jp = _j_path(out, cfg.model, sid, k_ref)
        if not jp.exists():
            continue
        J = np.load(jp)["J"]
        x0 = X[int(ctx["anchors_local"][ai])].astype(np.float64)
        N = ensure_neigh(ctx, ai, k_ref)
        Z = displacements(x0, X[N].astype(np.float64), cfg.coord)
        if J.shape[1] < cfg.d_ref:
            continue
        U = Z @ J[:, : cfg.d_core]
        pr = pair_antipodes(U, np.linalg.norm(U, axis=1))
        rec = {
            "sample_id": int(sid),
            "k": int(k_ref),
            "n_pairs": pr["n_pairs"],
            "pair_quality": pr["quality"],
            "resolved": bool(pr["n_pairs"] >= 24),
        }
        if pr["n_pairs"] >= 12:
            odd, even = odd_even_displacements(Z, pr["plus"], pr["minus"])
            Todd, Eeven = pca_subspace(odd, cfg.d_core), pca_subspace(even, 4)
            rec["O_odd_T12"] = projector_overlap(Todd, J[:, : cfg.d_core])
            rec["O_even_E4"] = projector_overlap(Eeven, J[:, cfg.d_core : cfg.d_ref])
            rec["O_odd_E4"] = projector_overlap(Todd[:, : min(4, Todd.shape[1])], J[:, cfg.d_core : cfg.d_ref])
            rec["O_even_T12"] = projector_overlap(Eeven, J[:, : cfg.d_core])
            bp = _b_path(out, cfg.model, sid, k_ref)
            if bp.exists():
                UB = np.load(bp)["UA"]
                rec["O_even_B"] = projector_overlap(Eeven, UB[:, : min(4, UB.shape[1])])
                rec["O_odd_B"] = projector_overlap(Todd[:, : min(4, Todd.shape[1])], UB[:, : min(4, UB.shape[1])])
        rows.append(rec)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[osg] odd_even n={len(rows)}", flush=True)
