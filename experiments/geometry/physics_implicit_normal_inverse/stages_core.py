"""Carrier through Gauss-map stages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from geometry.physics_stable_tangent_dimension.nested_pca import radial_stratified_halves
from geometry.physics_stable_tangent_dimension.sphere_coords import (
    angular_radii,
    parallel_transport_basis_yx,
    rms_tangent_radius,
    sphere_log_map,
)

from .algebra import EPS, intersection_rank, projector_overlap, qr_orthonormal, sampson_batch, unpack_h, weighted_phi
from .pipeline import (
    ImplicitNormalConfig,
    _budget_ok,
    _done,
    _j_ours,
    cache_path,
    carrier_coords,
    classify_fit,
    dimension_from_labels,
    implicit_q2_from_pack,
    load_or_compute_J,
    scaling_for_directions,
)
from .stages import _load_fit_cache, _local_pack, ensure_fit_cache


def stage_carrier(root: Path, cfg: ImplicitNormalConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    marker = out / "carrier_done.json"
    path = out / "carrier_diagnostics.parquet"
    if _done(marker, cfg.force) and _done(path, cfg.force):
        return
    rows = []
    n_done = 0
    Rs = sorted(set([cfg.R, *[r for r in cfg.R_sens if r >= 4]]))
    for sid in ctx["use_sids"]:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        for k in ctx["ks"]:
            x0, Xloc, Z, _ = _local_pack(ctx, cfg, sid, k)
            J, ev, _ = load_or_compute_J(out, ctx, cfg, sid, k, Z)
            dest = _j_ours(out, cfg, sid, k)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                th = angular_radii(x0, Xloc)
                np.savez_compressed(dest, J=J, ev=ev, theta=th, rms=np.array([rms_tangent_radius(Z)]))
            th = angular_radii(x0, Xloc)
            for R in Rs:
                if k != (cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])) and R != cfg.R:
                    continue
                if J.shape[1] < R:
                    continue
                Y, outside = carrier_coords(Z, J, R)
                e = np.asarray(ev, dtype=np.float64)
                frac = float(np.sum(e[:R]) / max(float(np.sum(e)), EPS)) if len(e) else float("nan")
                rows.append(
                    {
                        "sample_id": int(sid),
                        "k": int(k),
                        "R": int(R),
                        "n": int(Y.shape[0]),
                        "outside_energy": outside,
                        "nested_energy_frac": frac,
                        "theta_median": float(np.median(th)),
                        "rms": float(rms_tangent_radius(Z)),
                        "radial_dot_max": float(np.max(np.abs(Z @ x0))),
                    }
                )
        n_done += 1
        if n_done % 32 == 0:
            print(f"[ini] carrier {n_done}/{len(ctx['use_sids'])}", flush=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    marker.write_text(json.dumps({"n": n_done, "n_rows": len(rows)}))
    print(f"[ini] carrier n={n_done} rows={len(rows)}", flush=True)


def _iter_jobs(ctx: dict, cfg: ImplicitNormalConfig):
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    for sid in ctx["use_sids"]:
        for k in ctx["ks"]:
            yield sid, k, cfg.R
            if k == k_ref:
                for R in cfg.R_sens:
                    if R != cfg.R:
                        yield sid, k, int(R)


def stage_linear_constraints(root: Path, cfg: ImplicitNormalConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "linear_constraint_spectrum.parquet"
    if _done(path, cfg.force):
        return
    rows = []
    for sid, k, R in _iter_jobs(ctx, cfg):
        if not _budget_ok(t0, cfg, reserve=True):
            break
        fit = ensure_fit_cache(out, ctx, cfg, sid, k, R)
        if fit is None or not fit.get("ok"):
            continue
        for d in fit["dir_rows"]:
            rows.append({"sample_id": int(sid), "k": int(k), "R": int(R), **d})
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ini] linear_constraints n={len(rows)}", flush=True)


def stage_quadratic_constraints(root: Path, cfg: ImplicitNormalConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "quadratic_constraint_spectrum.parquet"
    metrics = out / "implicit_constraint_metrics.parquet"
    if _done(path, cfg.force) and _done(metrics, cfg.force):
        return
    spec_rows, met_rows = [], []
    n_done = 0
    for sid, k, R in _iter_jobs(ctx, cfg):
        if not _budget_ok(t0, cfg, reserve=True):
            break
        fit = ensure_fit_cache(out, ctx, cfg, sid, k, R)
        if fit is None or not fit.get("ok"):
            continue
        n_done += 1
        for d in fit["dir_rows"]:
            spec_rows.append(
                {
                    "sample_id": int(sid),
                    "k": int(k),
                    "R": int(R),
                    "lam": fit["lam"],
                    "df": fit["df"],
                    **d,
                }
            )
        for qrow in fit["q_rows"]:
            met_rows.append({"sample_id": int(sid), "k": int(k), "R": int(R), **qrow})
        if n_done % 64 == 0:
            print(f"[ini] quadratic caches {n_done}", flush=True)
    pd.DataFrame(spec_rows).to_parquet(path, index=False)
    pd.DataFrame(met_rows).to_parquet(metrics, index=False)
    print(f"[ini] quadratic_constraints spec={len(spec_rows)} qrows={len(met_rows)}", flush=True)


def stage_geometric_refine(root: Path, cfg: ImplicitNormalConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "geometric_refine.parquet"
    if _done(path, cfg.force):
        return
    from .pipeline import _refine

    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    rows = []
    for sid in ctx["use_sids"][: max(cfg.n_parity_anchors, 8)]:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        fit = ensure_fit_cache(out, ctx, cfg, sid, k_ref, cfg.R)
        if fit is None or not fit.get("ok"):
            continue
        x0, Xloc, Z, _ = _local_pack(ctx, cfg, sid, k_ref)
        J, _, _ = load_or_compute_J(out, ctx, cfg, sid, k_ref, Z)
        Y, _ = carrier_coords(Z, J, cfg.R)
        th = angular_radii(x0, Xloc)
        Aidx, Bidx = radial_stratified_halves(th, cfg.seed + 31 * int(sid) + k_ref)
        Phi = weighted_phi(Y)
        for q in [q for q in (4, 8) if q <= fit["UA"].shape[1]]:
            A = qr_orthonormal(fit["UA"][:, :q])
            Hs = np.stack([unpack_h(fit["h_pack"][j], cfg.R) for j in range(q)], axis=0)
            samp0 = float(np.mean(sampson_batch(Y[Bidx], A, Hs)))
            A2, Hs2 = _refine(Y[Aidx], A, Hs, Phi[Aidx], fit["lam"], cfg.n_refine_steps)
            samp1 = float(np.mean(sampson_batch(Y[Bidx], A2, Hs2)))
            rows.append(
                {
                    "sample_id": int(sid),
                    "k": int(k_ref),
                    "R": int(cfg.R),
                    "q": q,
                    "sampson_spectral": samp0,
                    "sampson_refined": samp1,
                    "improved": bool(samp1 < samp0 - 1e-12),
                    "overlap_spectral_refined": projector_overlap(A, A2),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ini] geometric_refine n={len(rows)}", flush=True)


def stage_constraint_scaling(root: Path, cfg: ImplicitNormalConfig, ctx: dict, t0: float) -> None:
    out = cfg.resolved(root)
    path = out / "constraint_scaling.parquet"
    if _done(path, cfg.force):
        return
    rows = []
    for sid in ctx["use_sids"]:
        if not _budget_ok(t0, cfg, reserve=True):
            break
        for k in ctx["ks"]:
            fit = ensure_fit_cache(out, ctx, cfg, sid, k, cfg.R)
            if fit is None or not fit.get("ok"):
                continue
            x0, Xloc, Z, _ = _local_pack(ctx, cfg, sid, k)
            J, _, _ = load_or_compute_J(out, ctx, cfg, sid, k, Z)
            Y, _ = carrier_coords(Z, J, cfg.R)
            th = angular_radii(x0, Xloc)
            for sc in scaling_for_directions(Y, th, fit["UA"], fit["h_pack"]):
                rows.append({"sample_id": int(sid), "k": int(k), "R": int(cfg.R), **sc})
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ini] constraint_scaling n={len(rows)}", flush=True)


def _persist_map(out: Path, ctx: dict, cfg: ImplicitNormalConfig) -> dict[tuple[int, int, int], float]:
    ks = list(ctx["ks"])
    persist: dict[tuple[int, int, int], float] = {}
    for sid in ctx["use_sids"]:
        fits = {}
        for k in ks:
            p = cache_path(out, cfg, sid, k, cfg.R)
            if p.exists():
                fits[k] = _load_fit_cache(p)
        for i, k in enumerate(ks):
            if k not in fits:
                continue
            m = fits[k]["UA"].shape[1]
            for j in range(m):
                ov = []
                for k2 in ks:
                    if abs(ks.index(k2) - i) != 1 or k2 not in fits:
                        continue
                    dots = np.abs(fits[k2]["UA"].T @ fits[k]["UA"][:, j])
                    ov.append(float(dots.max() ** 2) if dots.size else float("nan"))
                persist[(int(sid), int(k), j)] = float(np.nanmean(ov)) if ov else float("nan")
    return persist


def stage_normal_classification(root: Path, cfg: ImplicitNormalConfig, ctx: dict, thr: dict) -> None:
    out = cfg.resolved(root)
    path = out / "normal_classification.parquet"
    proj = out / "normal_projectors.parquet"
    if _done(path, cfg.force) and _done(proj, cfg.force):
        return
    scaling = (
        pd.read_parquet(out / "constraint_scaling.parquet")
        if (out / "constraint_scaling.parquet").exists()
        else pd.DataFrame()
    )
    persist = _persist_map(out, ctx, cfg)
    rows, prow = [], []
    for sid in ctx["use_sids"]:
        for k in ctx["ks"]:
            p = cache_path(out, cfg, sid, k, cfg.R)
            if not p.exists():
                continue
            fit = _load_fit_cache(p)
            if not fit.get("ok"):
                continue
            sc_sub = scaling[(scaling.sample_id == sid) & (scaling.k == k)] if len(scaling) else pd.DataFrame()
            sc_list = []
            for j in range(len(fit["dir_rows"])):
                hit = sc_sub[sc_sub.j == j] if len(sc_sub) and "j" in sc_sub.columns else pd.DataFrame()
                sc_list.append(hit.iloc[0].to_dict() if len(hit) else {})
            pers = [persist.get((int(sid), int(k), d["j"]), float("nan")) for d in fit["dir_rows"]]
            labels = classify_fit(fit, sc_list, pers, thr)
            bounds = dimension_from_labels(labels, cfg.R, fit["ev_lin"], thr)
            for d, lab, pv in zip(fit["dir_rows"], labels, pers):
                rows.append({"sample_id": int(sid), "k": int(k), "R": int(cfg.R), "label": lab, "persist": pv, **d})
            rec = {
                "sample_id": int(sid),
                "k": int(k),
                "R": int(cfg.R),
                "cN_minus": bounds["cN_minus"],
                "d1_plus": bounds["d1_plus"],
                "d1_minus": bounds["d1_minus"],
                "n_flat_prefix": bounds["n_flat_prefix"],
                "n_curv_prefix": bounds["n_curv_prefix"],
                "labels": json.dumps(labels),
            }
            prow.append(rec)
    pd.DataFrame(rows).to_parquet(path, index=False)
    pd.DataFrame(prow).to_parquet(proj, index=False)
    print(f"[ini] normal_classification n={len(rows)} anchors={len(prow)}", flush=True)


def stage_dimension_bounds(root: Path, cfg: ImplicitNormalConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "dimension_bounds.csv"
    if _done(path, cfg.force):
        return
    proj = out / "normal_projectors.parquet"
    if not proj.exists():
        pd.DataFrame().to_csv(path, index=False)
        return
    df = pd.read_parquet(proj)
    rows = []
    for k, g in df.groupby("k"):
        cNm = pd.to_numeric(g.cN_minus, errors="coerce")
        d1p = pd.to_numeric(g.d1_plus, errors="coerce")
        d1m = pd.to_numeric(g.d1_minus, errors="coerce")
        rows.append(
            {
                "k": int(k),
                "R": int(cfg.R),
                "n": int(len(g)),
                "median_cN_minus": float(cNm.median()),
                "iqr_cN_minus": float(cNm.quantile(0.75) - cNm.quantile(0.25)),
                "median_d1_plus": float(d1p.median()),
                "median_d1_minus": float(d1m.median()),
                "p_cN_ge_8": float((cNm >= 8).mean()),
                "p_cN_ge_4": float((cNm >= 4).mean()),
                "p_d1_le_12": float((d1p <= 12).mean()),
                "p_d1_le_16": float((d1p <= 16).mean()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[ini] dimension_bounds n={len(rows)}", flush=True)


def stage_implicit_curvature(root: Path, cfg: ImplicitNormalConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "implicit_curvature_rank.csv"
    if _done(path, cfg.force):
        return
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    rows = []
    for sid in ctx["use_sids"]:
        p = cache_path(out, cfg, sid, k_ref, cfg.R)
        if not p.exists():
            continue
        fit = _load_fit_cache(p)
        if not fit.get("ok"):
            continue
        for q in range(1, min(cfg.q_max, fit["UA"].shape[1]) + 1):
            info = implicit_q2_from_pack(fit["UA"], fit["h_pack"], q)
            rows.append(
                {
                    "sample_id": int(sid),
                    "k": int(k_ref),
                    "R": int(cfg.R),
                    "q": q,
                    "d1": cfg.R - q,
                    "q2": info["q2"],
                    "s0": float(info["s"][0]) if info["s"] else float("nan"),
                    "s1": float(info["s"][1]) if len(info["s"]) > 1 else float("nan"),
                    "Tdim": info.get("Tdim", cfg.R - q),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[ini] implicit_curvature n={len(rows)}", flush=True)


def stage_tail_analysis(root: Path, cfg: ImplicitNormalConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "tail_classification.parquet"
    if _done(path, cfg.force):
        return
    proj = out / "normal_projectors.parquet"
    if not proj.exists():
        pd.DataFrame().to_parquet(path, index=False)
        return
    pdf = pd.read_parquet(proj)
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    I = np.eye(cfg.R)
    E4 = I[:, cfg.d_core : cfg.d_ref]
    tail = I[:, cfg.d_core : cfg.R]
    rows = []
    for _, rec in pdf[pdf.k == k_ref].iterrows():
        sid = int(rec.sample_id)
        p = cache_path(out, cfg, sid, k_ref, cfg.R)
        if not p.exists():
            continue
        fit = _load_fit_cache(p)
        cN = int(rec.cN_minus)
        q_use = max(cN, 1)
        A = fit["UA"][:, : min(q_use, fit["UA"].shape[1])]
        PN = A @ A.T if A.shape[1] else np.zeros((cfg.R, cfg.R))
        e4_normal = float(np.trace(E4.T @ PN @ E4) / max(E4.shape[1], 1))
        rows.append(
            {
                "sample_id": sid,
                "k": int(k_ref),
                "cN_minus": cN,
                "overlap_E4": projector_overlap(A, E4) if A.shape[1] else float("nan"),
                "overlap_tail": projector_overlap(A, tail) if A.shape[1] else float("nan"),
                "intersect_rank_E4": intersection_rank(A, E4) if A.shape[1] else 0,
                "e4_normal_frac": e4_normal,
                "e4_nonnormal_frac": float(1.0 - e4_normal),
                "q4_overlap_E4": projector_overlap(fit["UA"][:, : min(4, fit["UA"].shape[1])], E4),
                "q8_overlap_E4": projector_overlap(fit["UA"][:, : min(8, fit["UA"].shape[1])], E4),
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[ini] tail_analysis n={len(rows)}", flush=True)


def stage_gauss_validation(root: Path, cfg: ImplicitNormalConfig, ctx: dict) -> None:
    out = cfg.resolved(root)
    path = out / "gauss_validation.csv"
    if _done(path, cfg.force):
        return
    k_ref = cfg.primary_k if cfg.primary_k in ctx["ks"] else max(ctx["ks"])
    sids = ctx["use_sids"][: cfg.n_gauss_anchors]
    X = ctx["X"]
    locs = []
    for sid in sids:
        ai = ctx["sid_to_ai"][int(sid)]
        locs.append(X[int(ctx["anchors_local"][ai])].astype(np.float64))
    Xa = np.stack(locs, axis=0)
    Xa = Xa / np.maximum(np.linalg.norm(Xa, axis=1, keepdims=True), EPS)
    frames = {}
    proj = out / "normal_projectors.parquet"
    pdf = pd.read_parquet(proj) if proj.exists() else pd.DataFrame()
    for sid in sids:
        p = cache_path(out, cfg, sid, k_ref, cfg.R)
        if not p.exists():
            continue
        fit = _load_fit_cache(p)
        x0, Xloc, Z, _ = _local_pack(ctx, cfg, sid, k_ref)
        J, _, _ = load_or_compute_J(out, ctx, cfg, sid, k_ref, Z)
        cN = 0
        if len(pdf):
            hit = pdf[(pdf.sample_id == sid) & (pdf.k == k_ref)]
            if len(hit):
                cN = int(hit.iloc[0].cN_minus)
        q = min(max(cN, 1), fit["UA"].shape[1])
        A = fit["UA"][:, :q]
        frames[int(sid)] = {
            "x": x0,
            "J": J[:, : cfg.R],
            "A": A,
            "Aamb": J[:, : cfg.R] @ A,
            "cN": cN,
            "Hs": np.stack([unpack_h(fit["h_pack"][j], cfg.R) for j in range(q)], axis=0) if q else None,
        }
    ids = [s for s in sids if int(s) in frames]
    rows = []
    for sid in ids:
        fi = frames[int(sid)]
        xi = fi["x"]
        dots = Xa @ (xi / max(np.linalg.norm(xi), EPS))
        order = np.argsort(-dots)
        ovs, weing = [], []
        n_ok = 0
        for jdx in order[1 : cfg.n_gauss_neighbors + 8]:
            if jdx >= len(sids):
                continue
            sj = sids[jdx]
            if int(sj) not in frames or int(sj) == int(sid):
                continue
            if n_ok >= cfg.n_gauss_neighbors:
                break
            fj = frames[int(sj)]
            Apt = parallel_transport_basis_yx(fj["x"], fi["x"], fj["Aamb"])
            A_in = fi["J"].T @ Apt
            if fi["A"].shape[1] == 0 or A_in.shape[1] == 0:
                continue
            qmin = min(A_in.shape[1], fi["A"].shape[1])
            A_in_q = qr_orthonormal(np.asarray(A_in[:, :qmin], dtype=np.float64))
            A_fi = fi["A"][:, :qmin]
            ovs.append(projector_overlap(A_in_q, A_fi))
            n_ok += 1
            v = sphere_log_map(fi["x"], fj["x"])
            vR = fi["J"].T @ v
            if fi["Hs"] is not None and qmin > 0 and fi["Hs"].shape[0] >= qmin:
                pred = np.stack([fi["Hs"][ℓ] @ vR for ℓ in range(qmin)], axis=1)
                obs = A_in[:, :qmin] - A_fi
                if pred.shape == obs.shape:
                    weing.append(float(np.sum(pred * obs) / max(np.linalg.norm(pred) * np.linalg.norm(obs), EPS)))
        rows.append(
            {
                "sample_id": int(sid),
                "k": int(k_ref),
                "cN": fi["cN"],
                "n_neighbors": n_ok,
                "median_overlap": float(np.nanmedian(ovs)) if ovs else float("nan"),
                "mean_overlap": float(np.nanmean(ovs)) if ovs else float("nan"),
                "weingarten_cos": float(np.nanmedian(weing)) if weing else float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[ini] gauss_validation n={len(rows)}", flush=True)
