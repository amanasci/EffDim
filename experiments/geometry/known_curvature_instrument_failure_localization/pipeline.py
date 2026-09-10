"""Bounded failure localization. Hard 45-minute wall. No new decoders. No new oracles."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from geometry.known_curvature_point_patch_fixture_audit.config import ANCHOR_HASH_SEED, FIXTURE_SEED
from geometry.known_curvature_point_patch_fixture_audit.estimator_d import split_indices
import torch

from geometry.known_curvature_point_patch_fixture_audit.estimator_q import fit_anchors_parallel, knn_indices
from geometry.known_curvature_point_patch_fixture_audit.fixtures import ambient_rotation, autodiff_geometry
from geometry.known_curvature_point_patch_fixture_audit.io_util import (
    assert_not_preserved,
    hash_stable_order,
    platonic_root,
    resolve_path,
    sha256_file16,
    write_df,
    write_json,
    write_text,
)
from geometry.known_curvature_point_patch_fixture_audit.pipeline import build_observed, select_anchors
from geometry.known_curvature_point_patch_fixture_audit.scoring import rank_cal

from .config import (
    D_LAT,
    FROZEN_HASH_PATHS,
    K_VALUES,
    N_ANCHORS,
    PRIMARY_CELLS,
    PRIMARY_N,
    RESERVE_WRITE_S,
    STRESS_CELLS,
    ExpConfig,
)
from .decision import decide
from .decoder_diag import d0_from_cached, load_decoder_seeds, seed_decomposition
from .q_ladder import (
    numpy_pca_frame,
    principal_angles,
    q1_exact,
    q2_exact_ridge,
    q3_pca,
    q4_pca_ridge,
    tensor_errors,
)
from .reports import maybe_ablation_plot, write_reports
from .tests_unit import run_unit_tests


def _cell_seed(tag: str) -> int:
    return FIXTURE_SEED + 17 * (sum(map(ord, tag)) % 10007)


def _remaining(t0: float, wall: float) -> float:
    return wall - RESERVE_WRITE_S - (time.time() - t0)


def _subset_ids(cached_ids: np.ndarray) -> np.ndarray:
    ordered = hash_stable_order(np.asarray(cached_ids, dtype=np.int64), ANCHOR_HASH_SEED)
    return ordered[:N_ANCHORS]


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    audit = resolve_path(root, cfg.audit_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []

    Q = ambient_rotation()
    print("[loc] unit tests", flush=True)
    tests = run_unit_tests(Q)
    write_json(out / "unit_tests.json", tests, force=True)
    if not tests["all_passed"]:
        write_text(out / "BLOCKER.md", "Unit tests failed.\n" + json.dumps(tests, indent=2), force=True)
        return {"blocked": True, "tests": tests}

    reuse = {rel: sha256_file16(root / rel) if (root / rel).exists() else "missing" for rel in FROZEN_HASH_PATHS}
    reuse["audit_dir"] = str(audit)
    write_json(out / "reuse_manifest.json", {"frozen_hashes": reuse, "n_anchors": N_ANCHORS, "k": list(K_VALUES)}, force=True)

    # ---- Phase 0: parity on 64-anchor subset of cached tables ----
    parity = {"cells": {}, "ok": True, "blocker": None}
    cached_tables = {}
    for name, sampling, noise, eta, tag in PRIMARY_CELLS + STRESS_CELLS:
        cell_dir = audit / "cells" / tag
        ap = cell_dir / "anchors.parquet"
        if not ap.exists():
            parity["ok"] = False
            parity["blocker"] = f"missing {tag}/anchors.parquet"
            break
        df = pd.read_parquet(ap)
        ids64 = _subset_ids(df["sample_id"].to_numpy())
        sub = df[df["sample_id"].isin(ids64)].copy()
        if len(sub) != N_ANCHORS:
            parity["ok"] = False
            parity["blocker"] = f"{tag}: expected {N_ANCHORS} aligned ids, got {len(sub)}"
            break
        cell = json.loads((cell_dir / "cell.json").read_text())
        rec = {
            "path": str(ap),
            "n_cached": int(len(df)),
            "n_subset": int(len(sub)),
            "anchor_ids": [int(x) for x in ids64],
            "has_T2": bool("K_dir_T2" in sub.columns and sub["K_dir_T2"].notna().sum() > 0),
            "n_T2": int(sub["K_dir_T2"].notna().sum()) if "K_dir_T2" in sub.columns else 0,
            "rho_Q_T1_64": rank_cal(sub["Q_K_dir_cross"], sub["K_dir_T1"] ** 2)["rho"] if "Q_K_dir_cross" in sub.columns else float("nan"),
            "rho_D_H_64": rank_cal(sub["H_norm_D"], sub["H_norm_T1"])["rho"] if "H_norm_D" in sub.columns else float("nan"),
            "rho_D_Kdir_64": rank_cal(sub["K_dir_D"], sub["K_dir_T1"])["rho"] if "K_dir_D" in sub.columns else float("nan"),
            "mean_Q_Kdir_64": float(np.nanmean(np.abs(sub["Q_K_dir_cross"]))) if "Q_K_dir_cross" in sub.columns else float("nan"),
            "cached_full_scores": {k: cell.get("scores", {}).get(k) for k in ("rho_D_H", "rho_D_T1", "rho_Q_T1", "rho_Q_T2", "F0_Q_Kdir", "F0_D_Kdir")},
        }
        if rec["has_T2"] and "Q_K_dir_cross" in sub.columns:
            m = sub["K_dir_T2"].notna()
            rec["rho_Q_T2_64"] = rank_cal(sub.loc[m, "Q_K_dir_cross"], sub.loc[m, "K_dir_T2"] ** 2)["rho"]
        cached_tables[tag] = {"df": sub, "ids": ids64, "cell": cell, "dir": cell_dir}
        parity["cells"][tag] = rec
    if not parity["ok"]:
        write_json(out / "parity.json", parity, force=True)
        write_text(out / "BLOCKER.md", f"sample_id alignment failed: {parity['blocker']}\n", force=True)
        return {"blocked": True, "parity": parity}

    # Expected audit signatures
    a4 = parity["cells"]["A_F4_S0_N0_n16384_k2048_eta0.00"]
    a0 = parity["cells"]["A_F0_S0_N0_n16384_k2048_eta0.00"]
    parity["checks"] = {
        "F0_Q_near_zero": bool(abs(a0.get("mean_Q_Kdir_64") or 1) < 1e-6 or abs((a0["cached_full_scores"] or {}).get("F0_Q_Kdir") or 1) < 1e-6),
        "F4_D_rhoH_full": (a4["cached_full_scores"] or {}).get("rho_D_H"),
        "F4_D_rhoK_full": (a4["cached_full_scores"] or {}).get("rho_D_T1"),
        "F4_Q_T2_weak": (a4.get("rho_Q_T2_64") is not None and (not np.isfinite(a4.get("rho_Q_T2_64")) or abs(a4.get("rho_Q_T2_64", 0)) < 0.5)),
    }
    write_json(out / "parity.json", {k: v for k, v in parity.items() if k != "cells"} | {"cells": {t: {kk: vv for kk, vv in rec.items()} for t, rec in parity["cells"].items()}}, force=True)

    # Cached decoder D0/D3 first (parquet only; no weights).
    dec_rows = []
    dec_summary = {}
    for name, sampling, noise, eta, tag in PRIMARY_CELLS:
        cached = cached_tables[tag]
        ddf = load_decoder_seeds(cached["dir"])
        cell = cached["cell"]
        if ddf is None:
            dec_summary[name] = {"available": False}
            skipped.append(f"D:{name}:no_seeds")
            continue
        ddf64 = ddf[ddf["sample_id"].isin(cached["ids"])]
        dec_summary[name] = seed_decomposition(ddf64, cell.get("recon"))
        d0 = d0_from_cached(cached["df"], cached["ids"])
        d0["fixture"] = name
        dec_rows.append(d0)
    if dec_rows:
        write_df(out / "decoder_ablation_table.csv", pd.concat(dec_rows, ignore_index=True), force=True)
    else:
        write_df(out / "decoder_ablation_table.csv", pd.DataFrame(), force=True)
    write_json(out / "decoder_seed_decomposition.json", dec_summary, force=True)
    stages_done = ["D0_D3_cached"]
    skipped.append("D1_true_projector:no_decoder_weights")
    skipped.append("D2_finite_difference:no_decoder_weights")

    q_rows = []
    design_rows = []
    tan_rows = []

    def add_q_row(stage, name, sampling, noise, eta, k, sid, fit, B_t1, g_t1, B_t2, g_t2, scalars, J_true=None):
        J_est = fit.get("J") if fit.get("ok") else None
        te_t1 = tensor_errors(fit.get("Hess"), fit.get("g"), B_t1, g_t1, J_est, J_true) if fit.get("ok") else {}
        te_t2 = tensor_errors(fit.get("Hess"), fit.get("g"), B_t2, g_t2, J_est, J_true) if fit.get("ok") and B_t2 is not None else {}
        kdir = float(fit["K_dir"]) if fit.get("ok") else float("nan")
        hn = float(fit["H_norm"]) if fit.get("ok") else float("nan")
        ktf = float(fit["K_tf"]) if fit.get("ok") else float("nan")
        row = {
            "stage": stage,
            "fixture": name,
            "sampling": sampling,
            "noise": noise,
            "eta": eta,
            "k": k,
            "sample_id": int(sid),
            "ok": bool(fit.get("ok")),
            "K_dir": kdir,
            "H_norm": hn,
            "K_tf": ktf,
            "K_dir_T1": scalars.get("K_dir_T1", float("nan")),
            "K_dir_T2": scalars.get("K_dir_T2", float("nan")),
            "K_dir_T3": scalars.get("K_dir_T3", float("nan")),
            "H_norm_T1": scalars.get("H_norm_T1", float("nan")),
            "H_norm_T2": scalars.get("H_norm_T2", float("nan")),
            "rel_Kdir_T1": float(abs(kdir - scalars.get("K_dir_T1", np.nan)) / max(abs(scalars.get("K_dir_T1", 0)) + abs(kdir), 1e-12)) if fit.get("ok") else float("nan"),
            "rel_Kdir_T2": float(abs(kdir - scalars.get("K_dir_T2", np.nan)) / max(abs(scalars.get("K_dir_T2", 0)) + abs(kdir), 1e-12)) if fit.get("ok") and np.isfinite(scalars.get("K_dir_T2", np.nan)) else float("nan"),
            **{f"t1_{k}": v for k, v in te_t1.items()},
            **{f"t2_{k}": v for k, v in te_t2.items()},
        }
        q_rows.append(row)
        if fit.get("ok") and fit.get("design"):
            design_rows.append({"stage": stage, "fixture": name, "k": k, "sample_id": int(sid), **fit["design"]})
        if fit.get("ok") and J_est is not None and J_true is not None and stage in ("Q3", "Q4", "Q5"):
            tan_rows.append({"stage": stage, "fixture": name, "k": k, "sample_id": int(sid), **principal_angles(J_est, J_true)})

    def process_condition(name, sampling, noise, eta, tag, ks, do_q5=True):
        if _remaining(t0, cfg.wall_s) < 30:
            skipped.append(f"{tag}:{ks}")
            return False
        cached = cached_tables[tag]
        ids = cached["ids"]
        df = cached["df"]
        seed = _cell_seed(tag)
        print(f"[loc] cloud {tag} seed={seed}", flush=True)
        cloud = build_observed(name, sampling, noise, PRIMARY_N, eta, seed, Q, r_med_hint=None, k_for_eta=2048)
        id_to_row = {int(s): i for i, s in enumerate(cloud["sample_ids"])}
        if any(int(s) not in id_to_row for s in ids):
            write_text(out / "BLOCKER.md", f"regenerated cloud missing sample_ids for {tag}\n", force=True)
            raise RuntimeError(f"sample_id mismatch {tag}")
        # cheap T1 tensors
        t1 = {}
        for sid in ids:
            row = id_to_row[int(sid)]
            t1[int(sid)] = autodiff_geometry(name, cloud["z"][row], Q)
        for k in ks:
            if _remaining(t0, cfg.wall_s) < 20:
                skipped.append(f"{tag}:k={k}")
                continue
            print(f"[loc] {tag} k={k} knn", flush=True)
            Xanc = np.stack([cloud["X"][id_to_row[int(s)]] for s in ids])
            neigh = knn_indices(cloud["X"], Xanc, k, device=torch.device("cpu"))
            print(f"[loc] {tag} k={k} knn done", flush=True)
            radii = np.array([np.median(np.linalg.norm(cloud["X"][neigh[i]] - Xanc[i], axis=1)) for i in range(len(ids))])
            q5_payloads = []
            cached_fits = []
            for i, sid in enumerate(ids):
                if _remaining(t0, cfg.wall_s) < 15:
                    skipped.append(f"{tag}:k={k}:partial")
                    break
                sid = int(sid)
                Yi = cloud["X"][neigh[i]]
                zi = cloud["z"][neigh[i]]
                z0 = cloud["z"][id_to_row[sid]]
                geo = t1[sid]
                Uex = zi - z0[None, :]
                scalars = df.loc[df.sample_id == sid].iloc[0].to_dict()
                B_t2 = geo["B"] if name in ("F0", "F1", "F2") else None
                g_t2 = geo["g"] if B_t2 is not None else None
                frame = numpy_pca_frame(Yi, D_LAT)
                fits = {
                    "Q1": q1_exact(Yi, geo["G"], geo["J"], U=Uex),
                    "Q2": q2_exact_ridge(Yi, geo["G"], geo["J"], U=Uex),
                    "Q3": q3_pca(Yi, frame=frame),
                    "Q4": q4_pca_ridge(Yi, frame=frame),
                }
                if i == 0 or (i + 1) % 8 == 0:
                    print(f"[loc] {tag} k={k} anchor {i+1}/{len(ids)}", flush=True)
                for st, ft in fits.items():
                    add_q_row(st, name, sampling, noise, eta, k, sid, ft, geo["B"], geo["g"], B_t2, g_t2, scalars, geo["J"])
                if do_q5:
                    q5_payloads.append({"Xloc": Yi, "d": D_LAT, "n_splits": 1, "seed": seed, "ai": i})
                    cached_fits.append((sid, geo, scalars, B_t2, g_t2))
            if do_q5 and q5_payloads and _remaining(t0, cfg.wall_s) > 20:
                print(f"[loc] {tag} k={k} Q5 n={len(q5_payloads)} workers={cfg.n_workers}", flush=True)
                q5s = fit_anchors_parallel(q5_payloads, cfg.n_workers)
                from geometry.known_curvature_point_patch_fixture_audit.geometry import curvature_from_B

                for (sid, geo, scalars, B_t2, g_t2), raw in zip(cached_fits, q5s):
                    if not raw.get("ok"):
                        add_q_row("Q5", name, sampling, noise, eta, k, sid, {"ok": False}, geo["B"], geo["g"], B_t2, g_t2, scalars, geo["J"])
                        continue
                    HA, HB = raw.get("Hess_A"), raw.get("Hess_B")
                    Hess = 0.5 * (HA + HB) if HA is not None and HB is not None else HA
                    J5 = raw["J"][:, :D_LAT]
                    g5 = J5.T @ J5
                    curv = curvature_from_B(Hess, g5)
                    ft = {"ok": True, "Hess": Hess, "J": J5, "x0": raw["x0"], "g": g5, "design": {"n_obs": int(raw["n_loc"]), "q": 136, "n_half": int(raw["n_loc"] // 2), "ridge": "frozen RIDGES A/B", "shrinkage": float("nan"), "design_rank": float("nan"), "cond": float("nan"), "edf": float("nan"), "resid_mse": float("nan")}, **curv}
                    add_q_row("Q5", name, sampling, noise, eta, k, sid, ft, geo["B"], geo["g"], B_t2, g_t2, scalars, geo["J"])
            stages_done.append(f"{tag}:k={k}")
            # store radii for shrinkage note
            design_rows.append({"stage": "radius", "fixture": name, "k": k, "sample_id": -1, "median_r": float(np.median(radii)), "n_obs": int(k), "q": 136})
        return True

    # Priority 1–3: primary S0/N0 k=2048 Q1–Q5 + later k=512
    try:
        for name, sampling, noise, eta, tag in PRIMARY_CELLS:
            process_condition(name, sampling, noise, eta, tag, ks=(2048,), do_q5=True)
        stages_done.append("primary_k2048")

        if _remaining(t0, cfg.wall_s) > 180:
            for name, sampling, noise, eta, tag in PRIMARY_CELLS:
                process_condition(name, sampling, noise, eta, tag, ks=(512,), do_q5=True)
            stages_done.append("primary_k512")
        else:
            skipped.append("k=512")

        if _remaining(t0, cfg.wall_s) > 180:
            for name, sampling, noise, eta, tag in STRESS_CELLS:
                process_condition(name, sampling, noise, eta, tag, ks=(2048,), do_q5=True)
            stages_done.append("stress_k2048")
        else:
            skipped.append("density_noise_cells")
    except RuntimeError as e:
        write_text(out / "BLOCKER.md", str(e) + "\n" + traceback.format_exc(), force=True)
        return {"blocked": True, "error": str(e)}

    qtab = pd.DataFrame(q_rows)
    write_df(out / "q_ablation_table.csv", qtab, force=True)
    write_df(out / "design_conditioning.csv", pd.DataFrame(design_rows), force=True)
    write_df(out / "tangent_alignment.csv", pd.DataFrame(tan_rows), force=True)

    def stage_metrics(stage, fixture=None, k=2048, sampling="S0", noise="N0"):
        m = qtab[(qtab.stage == stage) & (qtab.k == k) & (qtab.sampling == sampling) & (qtab.noise == noise)]
        if fixture:
            m = m[m.fixture == fixture]
        if not len(m):
            return {}
        outm = {
            "median_rel_Kdir_T1": float(np.nanmedian(m["rel_Kdir_T1"])),
            "median_rel_Kdir_T2": float(np.nanmedian(m["rel_Kdir_T2"])),
            "median_t1_rel_B": float(np.nanmedian(m["t1_rel_B"])) if "t1_rel_B" in m.columns else float("nan"),
            "F0_false_Kdir": float(np.nanmean(np.abs(m.loc[m.fixture == "F0", "K_dir"]))) if (m.fixture == "F0").any() else float("nan"),
        }
        f4 = m[m.fixture == "F4"]
        if len(f4) >= 8 and f4["K_dir_T2"].notna().sum() >= 8:
            outm["rho_Kdir_T2_F4"] = rank_cal(f4["K_dir"], f4["K_dir_T2"])["rho"]
            cal = rank_cal(f4["K_dir"], f4["K_dir_T2"])
            outm["cal_slope_T2_F4"] = cal.get("slope")
            outm["cal_R2_T2_F4"] = cal.get("R2")
        if len(f4) >= 8:
            outm["rho_Kdir_T1_F4"] = rank_cal(f4["K_dir"], f4["K_dir_T1"])["rho"]
        f1 = m[m.fixture == "F1"]
        if len(f1):
            outm["F1_tf_frac"] = float(np.nanmedian(f1["K_tf"] / np.clip(f1["K_dir"], 1e-12, None)))
        f2 = m[m.fixture == "F2"]
        if len(f2):
            outm["F2_H_frac"] = float(np.nanmedian(f2["H_norm"] / np.clip(np.sqrt(np.clip(f2["K_dir"], 0, None)), 1e-12, None)))
        return outm

    q1m = stage_metrics("Q1")
    q2m = stage_metrics("Q2")
    q3m = stage_metrics("Q3")
    q4m = stage_metrics("Q4")
    q5m = stage_metrics("Q5")

    def inc_rel(a, b, key="median_rel_Kdir_T2"):
        va, vb = a.get(key, np.nan), b.get(key, np.nan)
        if not (np.isfinite(va) and np.isfinite(vb)):
            va, vb = a.get("median_rel_Kdir_T1", np.nan), b.get("median_rel_Kdir_T1", np.nan)
        return float(vb - va) if np.isfinite(va) and np.isfinite(vb) else float("nan")

    def inc_rho(a, b):
        va, vb = a.get("rho_Kdir_T2_F4", np.nan), b.get("rho_Kdir_T2_F4", np.nan)
        if not (np.isfinite(va) and np.isfinite(vb)):
            va, vb = a.get("rho_Kdir_T1_F4", np.nan), b.get("rho_Kdir_T1_F4", np.nan)
        return float(va - vb) if np.isfinite(va) and np.isfinite(vb) else float("nan")

    increments = {
        "q1_to_q2_rel": inc_rel(q1m, q2m),
        "q1_to_q3_rel": inc_rel(q1m, q3m),
        "q3_to_q4_rel": inc_rel(q3m, q4m),
        "q4_to_q5_rel": inc_rel(q4m, q5m),
        "q1_to_q2_rho_drop": inc_rho(q1m, q2m),
        "q1_to_q3_rho_drop": inc_rho(q1m, q3m),
        "q4_to_q5_rho_drop": inc_rho(q4m, q5m),
    }

    # radius shrinkage
    rad = pd.DataFrame(design_rows)
    rad = rad[rad.stage == "radius"] if len(rad) and "stage" in rad.columns else pd.DataFrame()
    r2048 = float(rad.loc[rad.k == 2048, "median_r"].median()) if len(rad) and (rad.k == 2048).any() else float("nan")
    r512 = float(rad.loc[rad.k == 512, "median_r"].median()) if len(rad) and (rad.k == 512).any() else float("nan")
    expected = float((512 / 2048) ** (1.0 / D_LAT))
    q1_512 = stage_metrics("Q1", k=512)
    radius_note = {
        "median_r_2048": r2048,
        "median_r_512": r512,
        "ratio_512_over_2048": float(r512 / r2048) if r2048 and r2048 > 0 else float("nan"),
        "expected_k_to_d": expected,
        "q1_rel_T1_k2048": q1m.get("median_rel_Kdir_T1"),
        "q1_rel_T1_k512": q1_512.get("median_rel_Kdir_T1"),
        "note": "at d=16, k 2048→512 shrinks radius by only ~4%; each split has 1024 vs 256 obs vs q=136 coeffs. Not an asymptotic test.",
        "half_n_k2048": 1024,
        "half_n_k512": 256,
        "q": 136,
    }

    # density/noise table
    dn_rows = []
    for tag, lab in (
        ("A_F4_S0_N0_n16384_k2048_eta0.00", "S0N0"),
        ("B_F4_S1_N0_n16384_k2048_eta0.00", "S1_density"),
        ("C_F4_S0_N2_n16384_k2048_eta0.10", "N2_eta0.10"),
    ):
        for st in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            m = qtab[(qtab.stage == st) & (qtab.fixture == "F4") & (qtab.k == 2048)]
            if lab == "S1_density":
                m = m[m.sampling == "S1"]
            elif lab == "N2_eta0.10":
                m = m[m.noise == "N2"]
            else:
                m = m[(m.sampling == "S0") & (m.noise == "N0")]
            if not len(m):
                continue
            dn_rows.append(
                {
                    "condition": lab,
                    "stage": st,
                    "median_rel_T1": float(np.nanmedian(m.rel_Kdir_T1)),
                    "median_rel_T2": float(np.nanmedian(m.rel_Kdir_T2)),
                    "rho_T1": rank_cal(m.K_dir, m.K_dir_T1)["rho"],
                    "rho_T2": rank_cal(m.K_dir, m.K_dir_T2)["rho"] if m.K_dir_T2.notna().sum() >= 8 else float("nan"),
                    "T2_available": bool(m.K_dir_T2.notna().sum() >= 8),
                }
            )
    write_df(out / "density_noise_diagnostics.csv", pd.DataFrame(dn_rows), force=True)

    # T2 vs T1 on F4 from cached scalars
    f4 = cached_tables["A_F4_S0_N0_n16384_k2048_eta0.00"]["df"]
    t2t1 = {"median_rel_Kdir": float("nan"), "available": False}
    if "K_dir_T2" in f4.columns and f4["K_dir_T2"].notna().sum() >= 8:
        t2t1 = {
            "median_rel_Kdir": float(np.nanmedian(np.abs(f4["K_dir_T2"] - f4["K_dir_T1"]) / np.clip(np.abs(f4["K_dir_T1"]) + np.abs(f4["K_dir_T2"]), 1e-12, None))),
            "available": True,
            "n": int(f4["K_dir_T2"].notna().sum()),
        }

    seed_k = float(np.nanmean([dec_summary.get(fx, {}).get("spearman_Kdir", {}).get("mean_rho", np.nan) for fx in ("F0", "F1", "F2", "F4")]))

    bundle = {
        "q1_recovers_t2": bool(q1m.get("median_rel_Kdir_T2", 1) <= 0.25 or (q1m.get("rho_Kdir_T2_F4") or 0) >= 0.7 or q1m.get("median_rel_Kdir_T1", 1) <= 0.25),
        "q1": q1m,
        "q2": q2m,
        "q3": q3m,
        "q4": q4m,
        "q5": q5m,
        "increments": increments,
        "patch_differs_from_t1": bool(t2t1.get("median_rel_Kdir", 0) >= 0.30),
        "t2_vs_t1": t2t1,
        "d1_available": False,
        "d2_fd_agrees": False,
        "d_false_survives_true_projector": False,
        "decoder": {"spearman_Kdir": seed_k},
        "primary_complete": "primary_k2048" in stages_done,
    }
    decision = decide(bundle)
    write_json(out / "decision.json", decision, force=True)

    runtime = time.time() - t0
    summary = {
        "decision": decision["label"],
        "reason": decision["reason"],
        "tests_ok": tests["all_passed"],
        "n_tests": tests["n_tests"],
        "n_passed": tests["n_passed"],
        "runtime_s": runtime,
        "stages_done": stages_done,
        "skipped": skipped,
        "q1": q1m,
        "q2": q2m,
        "q3": q3m,
        "q4": q4m,
        "q5": q5m,
        "increments": increments,
        "radius": radius_note,
        "t2_vs_t1_F4": t2t1,
        "decoder": {k: {"spearman_Kdir": v.get("spearman_Kdir", {}).get("mean_rho") if isinstance(v, dict) else None, "spearman_H": v.get("spearman_H", {}).get("mean_rho") if isinstance(v, dict) else None, "recon_r2_range": v.get("recon_r2_range"), "limitation": v.get("limitation")} for k, v in dec_summary.items()},
        "n_q_rows": int(len(qtab)),
        "output_dir": str(out),
    }
    write_json(out / "summary.json", summary, force=True)
    write_reports(out, summary=summary, decision=decision, tests=tests)
    if _remaining(t0, cfg.wall_s) > 5:
        (out / "figures").mkdir(exist_ok=True)
        maybe_ablation_plot(out, qtab)

    required_done = "primary_k2048" in stages_done and "primary_k512" in stages_done and "stress_k2048" in stages_done
    if tests["all_passed"] and required_done:
        write_json(out / "COMPLETE.json", {"ok": True, "decision": decision["label"], "runtime_s": runtime, "n_q_rows": int(len(qtab))}, force=True)
    else:
        write_json(
            out / "TIME_CAP_STOP.json",
            {
                "ok": bool("primary_k2048" in stages_done),
                "primary_complete": "primary_k2048" in stages_done,
                "decision": decision["label"],
                "skipped": skipped,
                "stages_done": stages_done,
                "runtime_s": runtime,
            },
            force=True,
        )
    print(f"[loc] done label={decision['label']} t={runtime:.1f}s skipped={skipped}", flush=True)
    return summary
