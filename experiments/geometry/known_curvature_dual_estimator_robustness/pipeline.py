"""Bounded 12-cell dual-estimator robustness. No manuscript edits. No D=768 training."""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_EXP = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[3]
_NB = _REPO / "notebooks"
for p in (_EXP, _NB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from .config import (
    CELLS,
    CURVATURE_EXPERIMENTS_SHA,
    D_AMB,
    D_LAT,
    DATA_SEED,
    ExpConfig,
    FIXTURE_VALIDITY_AUDIT_SHA,
    K_PRIMARY,
    K_SECONDARY,
    MAX_AES,
    MAX_EPOCHS,
    N_ANCHORS,
    OUT_REL,
    RESERVE_WRITE_S,
    TRAIN_CFG,
    TRAIN_CFG_SEED,
    TORCH_INIT_SEED,
    WALL_S,
)
from .decoder import encode, estimate_D_full, estimate_D_residual, r2_centered, train_decoder, var_explained
from .fixtures import GENERATORS_NP, ambient_rotation_d28, truth_at
from .quadratic import oracle_T2_matched, oracle_T3_uniform, q_tensors, run_Q
from .reports import write_all
from .sampling import apply_noise, make_anchors, sample_training
from .scoring import axes_vec, compare_aligned, spearman_safe
from .tests_unit import run_unit_tests


def _remaining(t0, wall):
    return wall - RESERVE_WRITE_S - (time.time() - t0)


def _dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _reuse_manifest() -> dict:
    return {
        "decoder_D": {
            "source": "outputs/geometry/pointwise_decoder_curvature_reproduction + notebooks/pu_manifold/decoder_curvature.py",
            "curvature_experiments_sha": CURVATURE_EXPERIMENTS_SHA,
            "H": "g^{ab} II_ab unnormalized",
            "II": "(I-P_T) D^2 F",
            "differentiate": "raw model.decode for D-full; F/||F|| for D-residual",
            "train": TRAIN_CFG,
            "torch_init": TORCH_INIT_SEED,
            "epochs": MAX_EPOCHS,
            "architecture": "PlainAutoEncoder hidden (250,250,250) silu",
        },
        "quadratic_Q": {
            "source": "experiments/geometry/known_curvature_point_patch_fixture_audit/estimator_q.py",
            "fit": "nested_pca_frame + fit_quad RIDGES grid + split-half",
            "n_splits": 3,
            "k_primary": K_PRIMARY,
            "no_clamp": True,
            "T2_this_experiment": "matched neighbourhood, clean Y, exact latents (NOT prior T2=uniform-volume)",
            "T3_this_experiment": "Sobol uniform latent ball, max 2048 (prior T2-like volume design without sqrt(det g) weights)",
            "convention_map": "Prior audit T2=uniform volume, T3=sampling-weighted. This experiment T2=empirical neighbourhood, T3=uniform reference.",
        },
        "fixtures": {
            "source": "known_curvature_point_patch_fixture_audit F0/F1/F2/F4 generators",
            "change": "pad/rotate to D=28 with orthonormal_qr seed 20260907; frozen 768 Q.npy unused",
            "centers": "F4_CENTERS imported unchanged from audit config",
        },
        "fixture_validity_audit_sha": FIXTURE_VALIDITY_AUDIT_SHA,
        "compatibility_changes": [
            "D=28 rotation via seeded QR (768×768 matrix inapplicable)",
            "Decoder D-residual differentiates through normalization (not historical primary)",
            "T2/T3 labels follow this brief, not the prior audit names",
        ],
    }


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    out = (_REPO / cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "reuse_manifest.json", _reuse_manifest())
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": cfg.device,
        "cuda": bool(torch.cuda.is_available()),
        "curvature_experiments_sha": CURVATURE_EXPERIMENTS_SHA,
        "d": D_LAT,
        "D": D_AMB,
        "n_anchors": N_ANCHORS,
        "k": K_PRIMARY,
        "epochs": MAX_EPOCHS,
    }
    _dump(out / "environment.json", env)

    tests = run_unit_tests()
    _dump(out / "unit_test_results.json", tests)
    _dump(out / "truth_validation.json", {"unit_tests": tests["all_passed"], "rows": tests["rows"]})
    if not tests["all_passed"]:
        _dump(out / "COMPLETE.json", {"status": "blocked", "reason": "unit tests failed", "tests": tests})
        write_all(out, blocked=True, tests=tests)
        return {"blocked": True}

    Qrot = ambient_rotation_d28()
    fixtures = ("F0", "F1", "F2", "F4")
    anchors = {nm: make_anchors(nm, Qrot) for nm in fixtures}
    truth = {}
    for nm in fixtures:
        rows = []
        for i in range(N_ANCHORS):
            rows.append(truth_at(nm, anchors[nm]["z"][i], Qrot))
        truth[nm] = rows

    _dump(
        out / "fixture_manifest.json",
        {"fixtures": list(fixtures), "d": D_LAT, "D": D_AMB, "n_anchors": N_ANCHORS, "rotation": "orthonormal_qr(28, 20260907)"},
    )
    _dump(
        out / "sampling_manifest.json",
        {"S0": "Haar uniform n=5000", "S1": "Haar uniform n=1500", "S2": "w=exp(β s(u)) n=5000 β=log(10)/2", "S3": "same weight n=1500"},
    )
    _dump(
        out / "noise_manifest.json",
        {
            "scale": "s_x = median ||x-mean(x)|| on clean training cloud",
            "N0": "clean",
            "N1": "sphere-normal RMS 0.01 s_x, renormalize",
            "N2": "sphere-normal RMS 0.05 s_x, renormalize",
            "N3": "isotropic RMS 0.05 s_x, renormalize",
            "N4": "heteroscedastic 0.01–0.05 s_x by density, renormalize",
        },
    )

    skipped = []
    stages = ["unit_tests"]
    per_anchor = []
    d_full_rows, d_res_rows, q_t2_rows, q_t3_rows, q_pw_rows, scal_rows = [], [], [], [], [], []
    n_ae = 0
    run_k512 = False

    def persist():
        if per_anchor:
            pd.DataFrame(per_anchor).to_parquet(out / "per_anchor_metrics.parquet", index=False)
        for name, rows in (
            ("decoder_full_metrics.csv", d_full_rows),
            ("decoder_residual_metrics.csv", d_res_rows),
            ("quadratic_T2_metrics.csv", q_t2_rows),
            ("quadratic_T3_metrics.csv", q_t3_rows),
            ("quadratic_pointwise_metrics.csv", q_pw_rows),
            ("intrinsic_curvature_metrics.csv", scal_rows),
        ):
            pd.DataFrame(rows).to_csv(out / name, index=False)
        _dump(out / "runtime.json", {"runtime_s": time.time() - t0, "stages": stages, "skipped": skipped, "n_ae": n_ae})

    try:
        for fi, sa, no in CELLS:
            if n_ae >= MAX_AES:
                skipped.append(f"{fi}/{sa}/{no}:max_aes")
                continue
            if _remaining(t0, cfg.wall_s) < 45:
                skipped.append(f"{fi}/{sa}/{no}:wall")
                continue
            print(f"[dual] {fi} {sa} {no}", flush=True)
            cell = f"{fi}_{sa}_{no}"
            anc = anchors[fi]
            tr = sample_training(fi, sa, Qrot)
            nz = apply_noise(fi, no, tr, Qrot)
            X_obs = nz["X_obs"]
            model, info, t_train = train_decoder(X_obs)
            n_ae += 1
            x_obs_t = torch.as_tensor(X_obs, dtype=torch.float64)
            with torch.no_grad():
                y_obs = model(x_obs_t)["y"].cpu().numpy()
            ve_obs = var_explained(x_obs_t, torch.as_tensor(y_obs))
            r2_obs = r2_centered(X_obs, y_obs)
            z_anc = encode(model, anc["X"])
            with torch.no_grad():
                y_anc = model.decode(z_anc).cpu().numpy()
            rec_anc = float(np.mean(np.sum((y_anc - anc["X"]) ** 2, axis=1)))
            X_clean_train = tr["X_clean"]
            with torch.no_grad():
                y_cl = model(torch.as_tensor(X_clean_train, dtype=torch.float64))["y"].cpu().numpy()
            rec_clean = r2_centered(X_clean_train, y_cl)

            t1 = time.time()
            dfull = estimate_D_full(model, z_anc)
            dres = estimate_D_residual(model, z_anc)
            t_d = time.time() - t1

            HE_est = np.stack([r["H_E"] for r in dfull["rows"]])
            HE_true = np.stack([r["H_E"] for r in truth[fi]])
            HS_est = np.stack([r["H_S"] for r in dres["rows"]])
            HS_true = np.stack([r["H_S"] for r in truth[fi]])
            axE = axes_vec(HE_est, HE_true)
            axS = axes_vec(HS_est, HS_true)
            scal_e = np.array([r["Scal"] for r in dres["rows"]])
            scal_t = np.array([r["Scal"] for r in truth[fi]])
            e_res = float(np.mean([r["energy_B_S"] for r in dres["rows"]]))
            e_full_true = float(np.mean([r["energy_II_E"] for r in truth[fi]]))

            t_cmp = []
            for i in range(N_ANCHORS):
                cE = compare_aligned(dfull["rows"][i], truth[fi][i], kind="full")
                cS = compare_aligned(dres["rows"][i], truth[fi][i], kind="residual")
                t_cmp.append((cE, cS))

            d_full_rows.append({"cell": cell, "fixture": fi, "sampling": sa, "noise": no, **axE, "median_tensor_cos": float(np.nanmedian([c[0]["tensor_cosine"] for c in t_cmp])), "t_train_s": t_train, "t_eval_s": t_d, "var_explained": ve_obs, "r2_obs": r2_obs, "r2_clean_cloud": rec_clean, "anchor_mse": rec_anc, "epochs": info.get("epochs_run")})
            d_res_rows.append({"cell": cell, "fixture": fi, "sampling": sa, "noise": no, **axS, "median_tensor_cos": float(np.nanmedian([c[1]["tensor_cosine"] for c in t_cmp])), "scal_rho": spearman_safe(scal_e, scal_t), "f_res_mean": float(np.mean([r["f_res"] for r in dres["rows"]])), "energy_B_S_mean": e_res, "energy_II_E_true_mean": e_full_true, "energy_frac": e_res / max(e_full_true, 1e-18)})

            tq0 = time.time()
            qfits = run_Q(X_obs, anc["X"], K_PRIMARY, DATA_SEED, cfg.n_workers, cfg.q_device)
            t_q = time.time() - tq0

            q_t2_cos, q_t3_cos, q_pw_cos = [], [], []
            q_t2_ratio, q_pw_ratio = [], []
            scal_q, scal_t2, scal_t3, scal_pw = [], [], [], []
            kh_est, kh_t2 = [], []
            for i, fit in enumerate(qfits):
                qt = q_tensors(fit)
                tr_i = truth[fi][i]
                t2 = {"ok": False}
                t3 = {"ok": False}
                if qt.get("ok") and fit.get("ok"):
                    nidx = fit["neigh_idx"]
                    t2 = oracle_T2_matched(fi, Qrot, tr["z"], nidx, anc["z"][i])
                    rlat = float(np.median(np.linalg.norm(tr["z"][nidx] - anc["z"][i], axis=1)))
                    t3 = oracle_T3_uniform(fi, Qrot, anc["z"][i], max(rlat, 1e-4), seed=1000 + i)
                    c_t2 = compare_aligned(qt["tgt"], t2, kind="residual")
                    c_t3 = compare_aligned(t2, t3, kind="residual")  # measure dependence: T2 vs T3
                    c_pw = compare_aligned(qt["tgt"], tr_i, kind="residual")
                    c_t3pw = compare_aligned(t3, tr_i, kind="residual")
                    q_t2_cos.append(c_t2["tensor_cosine"])
                    q_t2_ratio.append(c_t2["tensor_ratio"])
                    q_t3_cos.append(c_t3["tensor_cosine"])
                    q_pw_cos.append(c_pw["tensor_cosine"])
                    q_pw_ratio.append(c_pw["tensor_ratio"])
                    scal_q.append(qt["Scal_cross"])
                    scal_t2.append(t2["Scal"])
                    scal_t3.append(t3["Scal"])
                    scal_pw.append(tr_i["Scal"])
                    kh_est.append(qt["K_H_cross"])
                    kh_t2.append(float(np.dot(t2["H_S_avg"], t2["H_S_avg"])))
                    per_anchor.append(
                        {
                            "cell": cell,
                            "fixture": fi,
                            "sampling": sa,
                            "noise": no,
                            "anchor_i": i,
                            "sample_id": int(anc["sample_id"][i]),
                            "w": float(anc["w"][i]),
                            "H_E_est": float(np.linalg.norm(dfull["rows"][i]["H_E"])),
                            "H_E_true": float(tr_i["H_E_norm"]),
                            "H_S_est": float(np.linalg.norm(dres["rows"][i]["H_S"])),
                            "H_S_true": float(tr_i["H_S_norm"]),
                            "Scal_D": float(dres["rows"][i]["Scal"]),
                            "Scal_true": float(tr_i["Scal"]),
                            "Scal_Q": qt["Scal_cross"],
                            "Q_ok": True,
                            "t2_cos": c_t2["tensor_cosine"],
                            "t3_vs_t2_cos": c_t3["tensor_cosine"],
                            "pw_cos": c_pw["tensor_cosine"],
                            "t3_vs_pw_cos": c_t3pw["tensor_cosine"],
                            "K_H_cross": qt["K_H_cross"],
                            "K_dir_cross": qt["K_dir_cross"],
                            "split_cos": qt["split_cos"],
                            "q_radius": fit.get("radius_median"),
                            "proj_err_D": t_cmp[i][0]["proj_err"],
                            "cond_g_D": float(dfull["cond_g"][i]),
                            "density_w": float(anc["w"][i]),
                        }
                    )
                else:
                    per_anchor.append({"cell": cell, "fixture": fi, "sampling": sa, "noise": no, "anchor_i": i, "sample_id": int(anc["sample_id"][i]), "Q_ok": False})

            def med(xs):
                return float(np.nanmedian(xs)) if xs else float("nan")

            q_t2_rows.append({"cell": cell, "fixture": fi, "sampling": sa, "noise": no, "median_tensor_cos": med(q_t2_cos), "median_tensor_ratio": med(q_t2_ratio), "scal_rho": spearman_safe(scal_q, scal_t2), "t_q_s": t_q, "k": K_PRIMARY})
            q_t3_rows.append({"cell": cell, "fixture": fi, "sampling": sa, "noise": no, "median_T2_T3_cos": med(q_t3_cos), "scal_T2_T3_rho": spearman_safe(scal_t2, scal_t3)})
            q_pw_rows.append({"cell": cell, "fixture": fi, "sampling": sa, "noise": no, "median_tensor_cos": med(q_pw_cos), "median_tensor_ratio": med(q_pw_ratio), "scal_rho": spearman_safe(scal_q, scal_pw), "K_H_rho": spearman_safe(kh_est, kh_t2)})
            scal_rows.append({"cell": cell, "fixture": fi, "sampling": sa, "noise": no, "D_scal_rho": spearman_safe(scal_e, scal_t), "Q_scal_T2_rho": spearman_safe(scal_q, scal_t2), "Q_scal_pw_rho": spearman_safe(scal_q, scal_pw)})
            stages.append(cell)
            persist()
            print(f"[dual] {cell} Dρ={axE['rho']:.3f} Dresρ={axS['rho']:.3f} QT2cos={med(q_t2_cos):.3f} t={t_train:.1f}+{t_d:.1f}+{t_q:.1f}s", flush=True)

            if (fi, sa, no) == ("F4", "S0", "N0") and not cfg.skip_k512 and _remaining(t0, cfg.wall_s) > 600:
                run_k512 = True
                q512 = run_Q(X_obs, anc["X"], K_SECONDARY, DATA_SEED, cfg.n_workers, cfg.q_device)
                _dump(out / "k512_note.json", {"ran": True, "n_ok": sum(1 for f in q512 if f.get("ok"))})
            elif (fi, sa, no) == ("F4", "S0", "N0"):
                skipped.append("k512:time")

    except Exception as exc:
        skipped.append(f"exception:{type(exc).__name__}:{exc}")
        print(traceback.format_exc(), flush=True)

    persist()
    from .decision import decide

    decision = decide(d_full_rows, d_res_rows, q_t2_rows, q_t3_rows, q_pw_rows, scal_rows, skipped)
    _dump(out / "decision.json", decision)
    _dump(out / "summary.json", decision)
    runtime_s = time.time() - t0
    _dump(out / "runtime.json", {"runtime_s": runtime_s, "wall_s": cfg.wall_s, "stages": stages, "skipped": skipped, "n_ae": n_ae, "k512": run_k512})
    status = "complete" if n_ae >= 12 and not any("wall" in s for s in skipped) else ("complete_with_resource_cap" if n_ae else "blocked")
    if n_ae >= 10:
        status = "complete" if n_ae == 12 and not any(":wall" in s for s in skipped) else "complete_with_resource_cap"
    _dump(out / "COMPLETE.json", {"status": status, "decision": decision, "runtime_s": runtime_s, "tests": tests["all_passed"], "n_ae": n_ae, "skipped": skipped})
    write_all(out, blocked=False, tests=tests, decision=decision, d_full=d_full_rows, d_res=d_res_rows, q_t2=q_t2_rows, q_pw=q_pw_rows, skipped=skipped, runtime_s=runtime_s)
    print(f"[dual] done label={decision.get('summary_label')} t={runtime_s:.1f}s n_ae={n_ae}", flush=True)
    return decision
