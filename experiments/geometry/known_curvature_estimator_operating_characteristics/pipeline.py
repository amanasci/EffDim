"""Bounded operating-characteristics audit. No manuscript edits. Max 8 new AEs."""

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
    BOOT_SEED,
    CAL_SPLIT_SEED,
    CELLS,
    CONDITION_LABELS,
    CURVATURE_EXPERIMENTS_SHA,
    D_AMB,
    D_LAT,
    DATA_SEED,
    DECODER_INIT_SEEDS,
    DRAW_SEEDS,
    ExpConfig,
    K_PRIMARY,
    MAX_NEW_AES,
    N_ANCHORS,
    N_BOOT,
    OPERATING_F4_CELLS,
    OUT_REL,
    REPEAT_CONDITIONS,
    REPEAT_FIXTURE,
    RESERVE_WRITE_S,
    STRUCTURAL_CELLS,
    WALL_S,
)
from .decision import decide
from .decoder_seeded import encode, estimate_D_full, estimate_D_residual, r2_centered, train_decoder_seeded, var_explained
from .figures import write_figures
from .metrics import (
    attenuation_ceiling,
    band_ceil,
    band_rank,
    band_rel,
    bootstrap_stat,
    calibration_report,
    calibration_split,
    delta_rho,
    dynamic_range,
    pairwise_ordering_accuracy,
    quartile_discrimination,
    _ranks,
    rank_bundle,
    residualize_linear,
    retained_ratio,
    spearman_safe,
)
from .oracles_regen import _design_cond, regenerate_oracles
from .reports import write_reports
from .reuse import load_prior, reproduction_headlines, reuse_manifest
from .scalar_oracles import matched_q_scalars, residualized_full_magnitude
from .tests_unit import parquet_csv_parity, run_unit_tests

from geometry.known_curvature_dual_estimator_robustness.fixtures import ambient_rotation_d28, truth_at
from geometry.known_curvature_dual_estimator_robustness.quadratic import oracle_T2_matched, oracle_T3_uniform, q_tensors, run_Q
from geometry.known_curvature_dual_estimator_robustness.sampling import apply_noise, make_anchors, sample_training
from geometry.known_curvature_point_patch_fixture_audit.estimator_q import knn_indices


def _remaining(t0, wall):
    return wall - RESERVE_WRITE_S - (time.time() - t0)


def _dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


REQUIRED_TABLES = (
    "operating_curves.csv",
    "scalar_rank_recovery.csv",
    "tensor_recovery.csv",
    "quartile_discrimination.csv",
    "pairwise_ordering.csv",
    "calibration.csv",
    "dynamic_range.csv",
    "repeat_reliability.csv",
    "reliability_ceiling.csv",
    "density_stratified.csv",
    "noise_scale_comparison.csv",
    "null_baselines.csv",
)


def _touch_required(out: Path, per_anchor: pd.DataFrame | None = None) -> None:
    for name in REQUIRED_TABLES:
        p = out / name
        if not p.exists():
            pd.DataFrame().to_csv(p, index=False)
    pq = out / "per_anchor_metrics.parquet"
    if not pq.exists():
        (per_anchor if per_anchor is not None else pd.DataFrame()).to_parquet(pq, index=False)


def _cell_sub(df, cell):
    return df[df["cell"] == cell].sort_values("anchor_i")


def _bundle_row(estimator, target, cell, est, truth, *, analytic_constant=False, extra=None):
    b = rank_bundle(est, truth, analytic_constant=analytic_constant)
    q = quartile_discrimination(est, truth)
    row = {
        "estimator": estimator,
        "target": target,
        "cell": cell,
        "fixture": cell.split("_")[0],
        "sampling": cell.split("_")[1] if cell.count("_") >= 2 else "",
        "noise": cell.split("_")[2] if cell.count("_") >= 2 else "",
        "condition": CONDITION_LABELS.get(cell, cell),
        **{k: b[k] for k in b},
        **{f"q_{k}": q[k] for k in ("top_precision", "top_recall", "top_jaccard", "p_high_above_low", "roc_auc_top_bottom", "chance_precision", "chance_roc_auc", "n_high", "n_low")},
    }
    if extra:
        row.update(extra)
    return row


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    out = (_REPO / cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = ["start"]
    skipped = []
    n_ae = 0
    new_cells: list[str] = []

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
        "max_new_aes": MAX_NEW_AES,
        "wall_s": cfg.wall_s,
    }
    _dump(out / "environment.json", env)

    tests = run_unit_tests()
    _dump(out / "unit_test_results.json", tests)
    if not tests["all_passed"]:
        runtime = {"runtime_s": time.time() - t0, "wall_s": cfg.wall_s, "n_ae": 0, "stages": stages, "skipped": skipped, "blocked": "unit_tests"}
        _dump(out / "runtime.json", runtime)
        _dump(out / "COMPLETE.json", {"status": "blocked", "reason": "unit tests failed", "tests": tests, "runtime_s": runtime["runtime_s"]})
        _dump(out / "decision.json", {"blocked": True, "reason": "unit_tests"})
        _dump(out / "summary.json", {"blocked": True, "reason": "unit_tests"})
        _dump(out / "parity.json", {"ok": False, "reason": "unit_tests"})
        _dump(out / "reuse_manifest.json", {"blocked": True})
        _touch_required(out)
        write_reports(out, decision={"blocked": True}, parity={"ok": False}, runtime=runtime, summary={"headline_md": "blocked on unit tests"})
        return {"blocked": True, "reason": "unit_tests"}

    prior = load_prior()
    _dump(out / "reuse_manifest.json", reuse_manifest(prior))
    stages.append("reuse")

    repro_h = reproduction_headlines(prior["repro_cells"])
    pa_parity = parquet_csv_parity(prior["per_anchor"], prior["d_res"], prior["d_full"], prior["q_t2"], prior["q_t3"])
    parity = {
        "ok": bool(pa_parity["ok"] and repro_h["cubic_ok"] and repro_h["ridge_ok"]),
        "reproduction": repro_h,
        "per_anchor": pa_parity,
        "prior_dual_label": prior["dual_decision"].get("summary_label"),
    }
    _dump(out / "parity.json", parity)
    tests2 = run_unit_tests(reused=pa_parity)
    _dump(out / "unit_test_results.json", tests2)
    stages.append("parity")

    if not parity["ok"]:
        runtime = {
            "runtime_s": time.time() - t0,
            "wall_s": cfg.wall_s,
            "n_ae": 0,
            "stages": stages,
            "skipped": skipped + ["new_computation_stopped:parity"],
            "reused_cells": list(prior["per_anchor"]["cell"].unique()),
            "new_cells": [],
            "stopped_before_cap": True,
        }
        _dump(out / "runtime.json", runtime)
        _dump(out / "summary.json", {"blocked": True, "reason": "parity", "parity": parity})
        _dump(out / "decision.json", {"blocked": True, "reason": "parity", "prior_exact_recovery_label": prior["dual_decision"].get("summary_label")})
        _dump(
            out / "COMPLETE.json",
            {"status": "blocked", "reason": "parity failed; no new computation", "parity": parity, "runtime_s": runtime["runtime_s"]},
        )
        _touch_required(out, prior["per_anchor"])
        write_reports(
            out,
            decision={"blocked": True, "prior_exact_recovery_label": prior["dual_decision"].get("summary_label")},
            parity=parity,
            runtime=runtime,
            summary={"headline_md": "Parity with reused audits failed. New decoder fits were not started."},
        )
        (out / "REPORT.md").write_text(
            (out / "REPORT.md").read_text() + "\n\n## Blocker\nExact headline parity failed. See parity.json.\n"
        )
        return {"blocked": True, "reason": "parity", "parity": parity}

    pa = prior["per_anchor"].copy()
    pa["H_E_resid_est"] = residualized_full_magnitude(pa["H_E_est"].to_numpy(), D_LAT)
    pa["H_E_resid_true"] = residualized_full_magnitude(pa["H_E_true"].to_numpy(), D_LAT)

    oracle_df = pd.DataFrame()
    if not cfg.skip_oracles and _remaining(t0, cfg.wall_s) > 90:
        print("[oc] regenerating T2/T3 scalar oracles", flush=True)
        try:
            oracle_df = regenerate_oracles()
            oracle_df.to_csv(out / "t2_t3_scalar_oracles.csv", index=False)
            stages.append("oracles")
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"oracles:{type(exc).__name__}:{exc}")
            print(traceback.format_exc(), flush=True)
    else:
        skipped.append("oracles:skipped")

    if not oracle_df.empty:
        pa["sample_id"] = pa["sample_id"].astype(np.int64)
        oracle_df["sample_id"] = oracle_df["sample_id"].astype(np.int64)
        pa = pa.merge(
            oracle_df[
                [
                    "cell",
                    "sample_id",
                    "K_H_T2",
                    "K_dir_T2",
                    "K_H_T3",
                    "K_dir_T3",
                    "K_H_pw",
                    "K_dir_pw",
                    "q_n_eff",
                    "q_design_cond",
                    "radius_lat",
                    "s_x",
                    "rms_eps",
                ]
            ],
            on=["cell", "sample_id"],
            how="left",
        )

    cal_mask = calibration_split(N_ANCHORS, CAL_SPLIT_SEED)

    rank_rows = []
    quart_rows = []
    pair_rows = []
    cal_rows = []
    dyn_rows = []
    curve_rows = []
    tensor_rows = []
    null_rows = []

    pairs = [
        ("D-full", "H_E_norm", "H_E_est", "H_E_true", True),
        ("D-full residualized", "H_S_from_HE", "H_E_resid_est", "H_E_resid_true", False),
        ("D-residual", "H_S_norm", "H_S_est", "H_S_true", False),
        ("D-residual Scal", "Scal", "Scal_D", "Scal_true", False),
        ("Q K_H vs T2", "K_H_T2", "K_H_cross", "K_H_T2", False),
        ("Q K_dir vs T2", "K_dir_T2", "K_dir_cross", "K_dir_T2", False),
        ("Q K_H vs T3", "K_H_T3", "K_H_cross", "K_H_T3", False),
        ("Q K_H vs pointwise", "K_H_pw", "K_H_cross", "K_H_pw", False),
        ("Q K_H vs pointwise H_S2", "H_S2_pw", "K_H_cross", None, False),
    ]

    clean_rho = {}
    for cell in list(OPERATING_F4_CELLS) + list(STRUCTURAL_CELLS):
        sub = _cell_sub(pa, cell)
        if sub.empty:
            continue
        analytic_HE = cell.startswith(("F0", "F1", "F2"))
        for est_name, target, ecol, tcol, force_const in pairs:
            if tcol is None:
                truth = (sub["H_S_true"].to_numpy() / float(D_LAT)) ** 2
            elif tcol not in sub.columns:
                continue
            else:
                truth = sub[tcol].to_numpy()
            if ecol not in sub.columns:
                continue
            est = sub[ecol].to_numpy()
            const = bool(force_const and analytic_HE)
            row = _bundle_row(est_name, target, cell, est, truth, analytic_constant=const)
            rank_rows.append(row)
            quart_rows.append({k: row[k] for k in row if k in ("estimator", "target", "cell", "fixture", "condition") or k.startswith("q_")})
            pair_rows.append(
                {
                    "estimator": est_name,
                    "target": target,
                    "cell": cell,
                    "pairwise_acc": row["pairwise_acc"],
                    "chance": 0.5,
                    "pair_band": row["pair_band"],
                }
            )
            dyn_rows.append(
                {
                    "estimator": est_name,
                    "target": target,
                    "cell": cell,
                    **{k: row[k] for k in ("n", "mean", "sd", "median_abs", "iqr", "cv", "r_iqr", "tol", "analytic_constant", "rank_target_degenerate")},
                }
            )
            if cell.startswith("F4"):
                key = (est_name, target)
                if cell == "F4_S0_N0":
                    clean_rho[key] = row["spearman"]
                    # global calibration factor from clean F4, frozen anchor half
                    factor_row = calibration_report(est, truth, cal_mask)
                    factor_row.update({"estimator": est_name, "target": target, "cell": cell, "scope": "fit_on_clean_F4"})
                    cal_rows.append(factor_row)
                    global_factor = factor_row["global_scale_factor"]
                else:
                    global_factor = None
                    for cr in cal_rows:
                        if cr.get("estimator") == est_name and cr.get("target") == target and cr.get("cell") == "F4_S0_N0":
                            global_factor = cr["global_scale_factor"]
                    cal_eval = calibration_report(est, truth, cal_mask)
                    cal_eval.update({"estimator": est_name, "target": target, "cell": cell, "scope": "apply_clean_factor", "global_scale_factor_applied": global_factor})
                    cal_rows.append(cal_eval)
                rho_c = clean_rho.get(key, float("nan"))
                curve_rows.append(
                    {
                        "estimator_target": est_name,
                        "target": target,
                        "cell": cell,
                        "fixture": "F4",
                        "condition": CONDITION_LABELS.get(cell, cell),
                        "spearman": row["spearman"],
                        "kendall_tau": row["kendall_tau"],
                        "pairwise_acc": row["pairwise_acc"],
                        "r_retained": retained_ratio(row["spearman"], rho_c) if cell != "F4_S0_N0" else 1.0,
                        "delta_rho": delta_rho(row["spearman"], rho_c) if cell != "F4_S0_N0" else 0.0,
                        "rank_target_degenerate": row["rank_target_degenerate"],
                        "rank_band": row["rank_band"],
                    }
                )

            # null baselines
            dens = sub["density_w"].to_numpy() if "density_w" in sub.columns else None
            rad = sub["q_radius"].to_numpy() if "q_radius" in sub.columns else None
            rng_n = np.random.default_rng(BOOT_SEED + (sum(map(ord, cell)) % 10000))
            shuf = [spearman_safe(rng_n.permutation(est), truth) for _ in range(40)]
            null_rows.append(
                {
                    "estimator": est_name,
                    "target": target,
                    "cell": cell,
                    "rho_estimator": row["spearman"],
                    "rho_random_mean": float(np.nanmean(shuf)),
                    "rho_density": spearman_safe(dens, truth) if dens is not None else float("nan"),
                    "rho_radius": spearman_safe(rad, truth) if rad is not None else float("nan"),
                    "rho_clean_reference": clean_rho.get((est_name, target), float("nan")),
                }
            )
            if dens is not None and rad is not None and est_name.startswith("Q") and np.isfinite(est).all():
                e_res = residualize_linear(_ranks(est), _ranks(dens), _ranks(rad))
                t_res = residualize_linear(_ranks(truth), _ranks(dens), _ranks(rad))
                null_rows[-1]["rho_after_density_radius"] = spearman_safe(e_res, t_res)

    # tensor table from reused cell CSVs
    for _, r in prior["d_full"].iterrows():
        tensor_rows.append({"estimator": "D-full", "cell": r["cell"], "median_tensor_cos": r.get("median_tensor_cos"), "median_H_cosine": r.get("median_cosine"), "rho_scalar": r.get("rho"), "constant_truth": r.get("constant_truth")})
    for _, r in prior["d_res"].iterrows():
        tensor_rows.append({"estimator": "D-residual", "cell": r["cell"], "median_tensor_cos": r.get("median_tensor_cos"), "median_H_cosine": r.get("median_cosine"), "rho_scalar": r.get("rho"), "scal_rho": r.get("scal_rho"), "energy_frac": r.get("energy_frac")})
    for _, r in prior["q_t2"].iterrows():
        tensor_rows.append({"estimator": "Q vs T2", "cell": r["cell"], "median_tensor_cos": r.get("median_tensor_cos"), "scal_rho": r.get("scal_rho")})
    for _, r in prior["q_t3"].iterrows():
        tensor_rows.append({"estimator": "T2 vs T3", "cell": r["cell"], "median_tensor_cos": r.get("median_T2_T3_cos"), "scal_rho": r.get("scal_T2_T3_rho")})
    for _, r in prior["q_pw"].iterrows():
        tensor_rows.append({"estimator": "Q vs pointwise", "cell": r["cell"], "median_tensor_cos": r.get("median_tensor_cos"), "K_H_rho": r.get("K_H_rho"), "scal_rho": r.get("scal_rho")})

    # F0 / F2 structural
    f0 = _cell_sub(pa, "F0_S0_N0")
    f2 = _cell_sub(pa, "F2_S0_N0")
    structural = {
        "f0_dres_energy_frac": float(prior["d_res"].loc[prior["d_res"]["cell"] == "F0_S0_N0", "energy_frac"].iloc[0]),
        "f0_dfull_cosine": float(prior["d_full"].loc[prior["d_full"]["cell"] == "F0_S0_N0", "median_cosine"].iloc[0]),
        "f2_dres_hs_mean": float(f2["H_S_est"].mean()) if not f2.empty else float("nan"),
        "f2_dres_tensor_cos": float(prior["d_res"].loc[prior["d_res"]["cell"] == "F2_S0_N0", "median_tensor_cos"].iloc[0]),
        "f2_note": "F2 has ~zero residual mean curvature and nonzero trace-free bending; D-residual H_S should stay small while tensor cosine can be nonzero.",
    }

    # density stratified on non-uniform F4
    dens_rows = []
    for cell in ("F4_S2_N0", "F4_S3_N4"):
        sub = _cell_sub(pa, cell)
        if sub.empty:
            continue
        w = sub["density_w"].to_numpy()
        try:
            qid = pd.qcut(w, 5, labels=False, duplicates="drop")
        except ValueError:
            qid = np.zeros(len(sub), dtype=int)
        sub = sub.copy()
        sub["quintile"] = np.asarray(qid) + 1
        specs = [
            ("D-full residualized", sub["H_E_resid_est"], sub["H_E_resid_true"]),
            ("D-residual", sub["H_S_est"], sub["H_S_true"]),
        ]
        if "K_H_T2" in sub.columns:
            specs.append(("Q K_H vs T2", sub["K_H_cross"], sub["K_H_T2"]))
        for qn, g in sub.groupby("quintile"):
            for name, ests, trs in specs:
                est = np.asarray(ests.loc[g.index] if hasattr(ests, "loc") else ests[g.index], dtype=np.float64)
                tru = np.asarray(trs.loc[g.index] if hasattr(trs, "loc") else trs[g.index], dtype=np.float64)
                # groupby index from original sub
                est = g["H_E_resid_est"].to_numpy() if name.startswith("D-full") else (g["H_S_est"].to_numpy() if name.startswith("D-residual") else g["K_H_cross"].to_numpy())
                tru = g["H_E_resid_true"].to_numpy() if name.startswith("D-full") else (g["H_S_true"].to_numpy() if name.startswith("D-residual") else g["K_H_T2"].to_numpy() if "K_H_T2" in g.columns else np.full(len(g), np.nan))
                dens_rows.append(
                    {
                        "cell": cell,
                        "estimator": name,
                        "quintile": int(qn),
                        "n": int(len(g)),
                        "spearman": spearman_safe(est, tru),
                        "pearson": float(np.corrcoef(est, tru)[0, 1]) if np.std(est) > 1e-15 and np.std(tru) > 1e-15 and np.isfinite(est).all() and np.isfinite(tru).all() else float("nan"),
                        "rel_abs_err": float(np.median(np.abs(est - tru) / np.maximum(np.abs(tru), 1e-12))) if np.isfinite(tru).all() else float("nan"),
                        "pairwise_acc": pairwise_ordering_accuracy(est, tru),
                        "q_n_eff": float(g["q_n_eff"].median()) if "q_n_eff" in g.columns else float(K_PRIMARY),
                        "q_radius": float(g["q_radius"].median()),
                        "q_design_cond": float(g["q_design_cond"].median()) if "q_design_cond" in g.columns else float("nan"),
                        "d_cond_g": float(g["cond_g_D"].median()),
                        "mean_density_w": float(g["density_w"].mean()),
                    }
                )

    # repeats
    repeat_rows = []
    rel_rows = []
    ceil_rows = []
    if not cfg.skip_repeats and _remaining(t0, cfg.wall_s) > 180:
        print("[oc] F4 repeat panel", flush=True)
        try:
            repeat_rows, n_ae, new_cells, skipped = _run_repeats(cfg, t0, skipped, n_ae)
            stages.append("repeats")
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"repeats:{type(exc).__name__}:{exc}")
            print(traceback.format_exc(), flush=True)
    else:
        skipped.append("repeats:skipped")

    repeat_df = pd.DataFrame(repeat_rows)
    if not repeat_df.empty:
        rel_rows, ceil_rows = _reliability_from_repeats(repeat_df)

    # noise scale
    noise_rows = _noise_scale(prior, oracle_df, pa)

    # persist tables
    rank_df = pd.DataFrame(rank_rows)
    curve_df = pd.DataFrame(curve_rows)
    dens_df = pd.DataFrame(dens_rows)
    ceil_df = pd.DataFrame(ceil_rows)
    rel_df = pd.DataFrame(rel_rows)

    curve_df.to_csv(out / "operating_curves.csv", index=False)
    rank_df.to_csv(out / "scalar_rank_recovery.csv", index=False)
    pd.DataFrame(tensor_rows).to_csv(out / "tensor_recovery.csv", index=False)
    pd.DataFrame(quart_rows).to_csv(out / "quartile_discrimination.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "pairwise_ordering.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(dyn_rows).to_csv(out / "dynamic_range.csv", index=False)
    rel_df.to_csv(out / "repeat_reliability.csv", index=False)
    ceil_df.to_csv(out / "reliability_ceiling.csv", index=False)
    dens_df.to_csv(out / "density_stratified.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(out / "noise_scale_comparison.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out / "null_baselines.csv", index=False)
    pa.to_parquet(out / "per_anchor_metrics.parquet", index=False)
    if not repeat_df.empty:
        repeat_df.to_parquet(out / "repeat_per_anchor.parquet", index=False)

    # decision inputs
    def _rho(est, target, cell):
        hit = rank_df[(rank_df["estimator"] == est) & (rank_df["target"] == target) & (rank_df["cell"] == cell)]
        return float(hit["spearman"].iloc[0]) if len(hit) else float("nan")

    def _deg(est, target, cell):
        hit = rank_df[(rank_df["estimator"] == est) & (rank_df["target"] == target) & (rank_df["cell"] == cell)]
        return bool(hit["rank_target_degenerate"].iloc[0]) if len(hit) else True

    q_t2_clean = _rho("Q K_H vs T2", "K_H_T2", "F4_S0_N0")
    q_t2_stress = _rho("Q K_H vs T2", "K_H_T2", "F4_S3_N4")
    q_t3_clean = _rho("Q K_H vs T3", "K_H_T3", "F4_S0_N0")
    q_pw_clean = _rho("Q K_H vs pointwise", "K_H_pw", "F4_S0_N0")
    t2_t3_cos = float(prior["q_t3"].loc[prior["q_t3"]["cell"] == "F4_S0_N0", "median_T2_T3_cos"].iloc[0])
    q_t2_cos = float(prior["q_t2"].loc[prior["q_t2"]["cell"] == "F4_S0_N0", "median_tensor_cos"].iloc[0])

    def _rel_get(est, kind):
        if rel_df.empty:
            return float("nan")
        hit = rel_df[(rel_df["estimator"] == est) & (rel_df["kind"] == kind)]
        return float(hit["r_rel"].iloc[0]) if len(hit) else float("nan")

    def _ceil_get(est):
        if ceil_df.empty:
            return {"f_ceiling": float("nan"), "r_rel": float("nan"), "rho_truth": float("nan")}
        hit = ceil_df[ceil_df["estimator"] == est]
        if hit.empty:
            return {"f_ceiling": float("nan"), "r_rel": float("nan"), "rho_truth": float("nan")}
        r = hit.iloc[0]
        return {"f_ceiling": float(r.get("f_ceiling", np.nan)), "r_rel": float(r.get("r_rel", np.nan)), "rho_truth": float(r.get("rho_truth", np.nan))}

    dres_clean = float(prior["d_res"].loc[prior["d_res"]["cell"] == "F4_S0_N0", "rho"].iloc[0])
    dres_stress = float(prior["d_res"].loc[prior["d_res"]["cell"] == "F4_S3_N4", "rho"].iloc[0])
    dfull_cos = float(prior["d_full"].loc[prior["d_full"]["cell"] == "F4_S0_N0", "median_cosine"].iloc[0])
    sparse_q = dens_df[dens_df["quintile"] == 1] if not dens_df.empty else pd.DataFrame()

    dims = {
        "d_full_clean_vector_recovery": band_rank(dfull_cos),
        "d_full_rank_informative": "degenerate_target" if _deg("D-full", "H_E_norm", "F4_S0_N0") else band_rank(_rho("D-full", "H_E_norm", "F4_S0_N0")),
        "d_residual_clean_rank_recovery": band_rank(dres_clean),
        "d_residual_stress_rank_recovery": band_rank(dres_stress),
        "q_t2_scalar_rank_recovery": band_rank(q_t2_clean, degenerate=_deg("Q K_H vs T2", "K_H_T2", "F4_S0_N0") if "K_H_T2" in pa.columns else False),
        "q_t2_tensor_recovery": band_rank(q_t2_cos),
        "q_t3_geometric_rank_recovery": band_rank(q_t3_clean),
        "q_pointwise_rank_recovery": band_rank(q_pw_clean),
        "q_sampling_measure_dependence": "strong" if t2_t3_cos < 0.90 else "weak",
        "d_sampling_reliability": band_rel(_rel_get("D-residual", "sampling")),
        "q_sampling_reliability": band_rel(_rel_get("Q K_H", "sampling")),
        "d_fraction_of_ceiling": band_ceil(_ceil_get("D-residual")["f_ceiling"]),
        "q_fraction_of_ceiling": band_ceil(_ceil_get("Q K_H")["f_ceiling"]),
    }

    d_full_ctx = {
        "sphere_rank_degenerate": _deg("D-full", "H_E_norm", "F4_S0_N0"),
        "cubic_rho": repro_h["d_full_cubic_rho"],
        "ridge_rho": repro_h["d_full_ridge_rho"],
        "clean_vector_cosine": dfull_cos,
        "residualized_f4_clean_rho": _rho("D-full residualized", "H_S_from_HE", "F4_S0_N0"),
        "residualized_f4_stress_rho": _rho("D-full residualized", "H_S_from_HE", "F4_S3_N4"),
        "unresolved": False,
    }
    q_ctx = {
        "t2_kh_clean_rho": q_t2_clean,
        "t2_kh_stress_rho": q_t2_stress,
        "t2_tensor_clean": q_t2_cos,
        "t2_t3_cos": t2_t3_cos,
        "sampling_measure_dependence": t2_t3_cos < 0.90,
        "sampling_reliability": _rel_get("Q K_H", "sampling"),
        "unresolved": not np.isfinite(q_t2_clean),
    }
    dres_ctx = {
        "f4_clean_rho": dres_clean,
        "f4_stress_rho": dres_stress,
        "f4_scal_rho": float(prior["d_res"].loc[prior["d_res"]["cell"] == "F4_S0_N0", "scal_rho"].iloc[0]),
        "unresolved": False,
    }
    decision = decide({"d_full": d_full_ctx, "q": q_ctx, "d_res": dres_ctx, "dimensions": dims})
    _dump(out / "decision.json", decision)

    figs = write_figures(out, curve_df, ceil_df, dens_df)

    runtime_s = time.time() - t0
    runtime = {
        "runtime_s": runtime_s,
        "wall_s": cfg.wall_s,
        "n_ae": n_ae,
        "max_new_aes": MAX_NEW_AES,
        "stages": stages + skipped,
        "skipped": skipped,
        "reused_cells": sorted(pa["cell"].unique().tolist()),
        "new_cells": new_cells,
        "stopped_before_cap": runtime_s < cfg.wall_s,
        "figures": figs,
    }
    _dump(out / "runtime.json", runtime)

    headline_md = _headline_md(
        repro_h, prior, d_full_ctx, q_ctx, dres_ctx, dims, rel_df, ceil_df, dens_df, noise_rows, decision, runtime
    )
    summary = {
        "parity_ok": True,
        "d_full_label": decision["d_full_summary_label"],
        "q_label": decision["q_summary_label"],
        "d_residual_label": decision["d_residual_summary_label"],
        "prior_label": decision["prior_exact_recovery_label"],
        "dimensions": dims,
        "structural": structural,
        "headline_md": headline_md,
        "dimensions_md": "\n".join(f"- {k}: **{v}**" for k, v in dims.items()),
        "manuscript_note": "Keep exact-recovery and operating-utility paragraphs separate.",
        "paths": {n: str(out / n) for n in (
            "summary.json", "decision.json", "REPORT.md", "COMPLETE.json"
        )},
    }
    _dump(out / "summary.json", summary)
    write_reports(out, decision=decision, parity=parity, runtime=runtime, summary=summary)

    status = "complete"
    if n_ae < MAX_NEW_AES and "repeats:skipped" in skipped:
        status = "complete_with_resource_cap"
    if any("repeats:" in s and "skipped" not in s for s in skipped):
        status = "complete_with_resource_cap"
    _dump(
        out / "COMPLETE.json",
        {
            "status": status,
            "runtime_s": runtime_s,
            "n_ae": n_ae,
            "parity_ok": True,
            "tests": tests2["all_passed"],
            "decision_labels": {
                "d_full": decision["d_full_summary_label"],
                "q": decision["q_summary_label"],
                "d_residual": decision["d_residual_summary_label"],
                "prior_unchanged": decision["prior_exact_recovery_label"],
            },
            "skipped": skipped,
        },
    )
    print(f"[oc] done t={runtime_s:.1f}s n_ae={n_ae} D={decision['d_full_summary_label']} Q={decision['q_summary_label']}", flush=True)
    return decision


def _run_repeats(cfg, t0, skipped, n_ae):
    Qrot = ambient_rotation_d28()
    anc = make_anchors(REPEAT_FIXTURE, Qrot)
    truth = [truth_at(REPEAT_FIXTURE, anc["z"][i], Qrot) for i in range(N_ANCHORS)]
    HS_true = np.array([t["H_S_norm"] for t in truth])
    HE_true = np.array([t["H_E_norm"] for t in truth])
    rows = []
    new_cells = []
    for sa, no in REPEAT_CONDITIONS:
        for draw_name, dseed in DRAW_SEEDS.items():
            if _remaining(t0, cfg.wall_s) < 90:
                skipped.append(f"repeats:{sa}_{no}_{draw_name}:wall")
                return rows, n_ae, new_cells, skipped
            print(f"[oc] repeat {sa} {no} draw={draw_name}", flush=True)
            tr = sample_training(REPEAT_FIXTURE, sa, Qrot, seed=int(dseed))
            nz = apply_noise(REPEAT_FIXTURE, no, tr, Qrot, seed=int(dseed))
            X_obs = nz["X_obs"]
            # Q once per draw
            qfits = run_Q(X_obs, anc["X"], K_PRIMARY, DATA_SEED, cfg.n_workers, cfg.q_device)
            idx = knn_indices(X_obs, anc["X"], K_PRIMARY, device=None)
            q_ok = []
            for i, fit in enumerate(qfits):
                qt = q_tensors(fit)
                nidx = idx[i]
                t2 = oracle_T2_matched(REPEAT_FIXTURE, Qrot, tr["z"], nidx, anc["z"][i]) if qt.get("ok") else None
                s2 = matched_q_scalars(t2["B_S"], t2["g"]) if t2 is not None else {}
                U = tr["z"][nidx] - anc["z"][i][None, :]
                q_ok.append(
                    {
                        "K_H_cross": qt.get("K_H_cross", float("nan")) if qt.get("ok") else float("nan"),
                        "K_dir_cross": qt.get("K_dir_cross", float("nan")) if qt.get("ok") else float("nan"),
                        "K_H_T2": s2.get("K_H_star", float("nan")),
                        "K_dir_T2": s2.get("K_dir_star", float("nan")),
                        "q_radius": fit.get("radius_median", float("nan")),
                        "q_n_eff": int(fit.get("n_loc") or len(nidx)),
                        "q_design_cond": _design_cond(U),
                        "Q_ok": bool(qt.get("ok")),
                    }
                )
            for init_seed in DECODER_INIT_SEEDS:
                if n_ae >= MAX_NEW_AES:
                    skipped.append("repeats:max_aes")
                    return rows, n_ae, new_cells, skipped
                if _remaining(t0, cfg.wall_s) < 60:
                    skipped.append("repeats:wall")
                    return rows, n_ae, new_cells, skipped
                model, info, t_train = train_decoder_seeded(X_obs, int(init_seed), device=cfg.device)
                n_ae += 1
                z_anc = encode(model, anc["X"])
                with torch.no_grad():
                    y_anc = model.decode(z_anc).cpu().numpy()
                rec_anc = np.sum((y_anc - anc["X"]) ** 2, axis=1)
                dfull = estimate_D_full(model, z_anc)
                dres = estimate_D_residual(model, z_anc)
                HE = np.array([np.linalg.norm(r["H_E"]) for r in dfull["rows"]])
                HS = np.array([np.linalg.norm(r["H_S"]) for r in dres["rows"]])
                tag = f"F4_{sa}_{no}_draw{draw_name}_seed{init_seed}"
                new_cells.append(tag)
                print(f"[oc] trained {tag} t={t_train:.1f}s n_ae={n_ae}", flush=True)
                for i in range(N_ANCHORS):
                    rows.append(
                        {
                            "panel": "repeat",
                            "cell": f"F4_{sa}_{no}",
                            "fixture": "F4",
                            "sampling": sa,
                            "noise": no,
                            "draw": draw_name,
                            "init_seed": int(init_seed),
                            "data_seed": int(dseed),
                            "anchor_i": i,
                            "sample_id": int(anc["sample_id"][i]),
                            "H_E_est": float(HE[i]),
                            "H_E_true": float(HE_true[i]),
                            "H_S_est": float(HS[i]),
                            "H_S_true": float(HS_true[i]),
                            "H_E_resid_est": float(residualized_full_magnitude([HE[i]], D_LAT)[0]),
                            "H_E_resid_true": float(residualized_full_magnitude([HE_true[i]], D_LAT)[0]),
                            "K_H_cross": q_ok[i]["K_H_cross"],
                            "K_dir_cross": q_ok[i]["K_dir_cross"],
                            "K_H_T2": q_ok[i]["K_H_T2"],
                            "K_dir_T2": q_ok[i]["K_dir_T2"],
                            "q_radius": q_ok[i]["q_radius"],
                            "q_n_eff": q_ok[i]["q_n_eff"],
                            "q_design_cond": q_ok[i]["q_design_cond"],
                            "anchor_mse": float(rec_anc[i]),
                            "cond_g_D": float(dfull["cond_g"][i]),
                            "density_w": float(anc["w"][i]),
                            "t_train_s": t_train,
                            "epochs": info.get("epochs_run"),
                        }
                    )
                del model
    return rows, n_ae, new_cells, skipped


def _reliability_from_repeats(df: pd.DataFrame):
    rel_rows = []
    ceil_rows = []
    rng = np.random.default_rng(BOOT_SEED)
    for (sa, no), gcond in df.groupby(["sampling", "noise"]):
        cond = f"F4_{sa}_{no}"
        # Q: one value per draw (duplicate across init seeds) — use seed 0 rows
        q = gcond[gcond["init_seed"] == DECODER_INIT_SEEDS[0]]
        qa = q[q["draw"] == "A"].sort_values("anchor_i")
        qb = q[q["draw"] == "B"].sort_values("anchor_i")
        if len(qa) == N_ANCHORS and len(qb) == N_ANCHORS:
            for col, est_name, tcol in (
                ("K_H_cross", "Q K_H", "K_H_T2"),
                ("K_dir_cross", "Q K_dir", "K_dir_T2"),
            ):
                r_rel = spearman_safe(qa[col], qb[col])
                rho_t = spearman_safe(qa[col], qa[tcol])
                ac = attenuation_ceiling(r_rel, rho_t)
                boot_rel = bootstrap_stat(spearman_safe, [qa[col].to_numpy(), qb[col].to_numpy()], rng)
                boot_t = bootstrap_stat(spearman_safe, [qa[col].to_numpy(), qa[tcol].to_numpy()], rng)

                def _fceil(a, b, t):
                    rr = spearman_safe(a, b)
                    rt = spearman_safe(a, t)
                    return attenuation_ceiling(rr, rt)["f_ceiling"]

                boot_f = bootstrap_stat(_fceil, [qa[col].to_numpy(), qb[col].to_numpy(), qa[tcol].to_numpy()], rng)
                rel_rows.append({"estimator": est_name, "kind": "sampling", "cell": cond, "r_rel": r_rel, "r_rel_lo": boot_rel["lo"], "r_rel_hi": boot_rel["hi"], "band": band_rel(r_rel)})
                ceil_rows.append(
                    {
                        "estimator": est_name,
                        "cell": cond,
                        "rho_truth": rho_t,
                        "rho_truth_lo": boot_t["lo"],
                        "rho_truth_hi": boot_t["hi"],
                        **ac,
                        "f_lo": boot_f["lo"],
                        "f_hi": boot_f["hi"],
                    }
                )
        # D: init vs sampling vs total
        for col, est_name, tcol in (
            ("H_S_est", "D-residual", "H_S_true"),
            ("H_E_resid_est", "D-full residualized", "H_E_resid_true"),
            ("H_E_est", "D-full", "H_E_true"),
        ):
            def take(draw, seed):
                s = gcond[(gcond["draw"] == draw) & (gcond["init_seed"] == seed)].sort_values("anchor_i")
                return s

            a0, a1, b0, b1 = take("A", 0), take("A", 1), take("B", 0), take("B", 1)
            if len(a0) != N_ANCHORS:
                continue
            pairs = {
                "init": (a0, a1) if len(a1) == N_ANCHORS else None,
                "sampling": (a0, b0) if len(b0) == N_ANCHORS else None,
                "total": (a0, b1) if len(b1) == N_ANCHORS else None,
            }
            for kind, pair in pairs.items():
                if pair is None:
                    continue
                x, y = pair
                r_rel = spearman_safe(x[col], y[col])
                boot_rel = bootstrap_stat(spearman_safe, [x[col].to_numpy(), y[col].to_numpy()], rng)
                rel_rows.append({"estimator": est_name, "kind": kind, "cell": cond, "r_rel": r_rel, "r_rel_lo": boot_rel["lo"], "r_rel_hi": boot_rel["hi"], "band": band_rel(r_rel)})
            rho_t = spearman_safe(a0[col], a0[tcol])
            r_samp = next((r["r_rel"] for r in rel_rows if r["estimator"] == est_name and r["kind"] == "sampling" and r["cell"] == cond), float("nan"))
            ac = attenuation_ceiling(r_samp, rho_t)
            boot_t = bootstrap_stat(spearman_safe, [a0[col].to_numpy(), a0[tcol].to_numpy()], rng)
            ceil_rows.append(
                {
                    "estimator": est_name,
                    "cell": cond,
                    "rho_truth": rho_t,
                    "rho_truth_lo": boot_t["lo"],
                    "rho_truth_hi": boot_t["hi"],
                    **ac,
                }
            )
    return rel_rows, ceil_rows


def _noise_scale(prior, oracle_df, pa) -> list[dict]:
    rows = []
    # synthetic
    if not oracle_df.empty:
        for cell, g in oracle_df.groupby("cell"):
            sx = float(g["s_x"].median())
            rms = float(g["rms_eps"].median())
            rel = rms / max(sx, 1e-12)
            rows.append(
                {
                    "domain": "synthetic",
                    "cell": cell,
                    "quantity": "observation_rms_over_s_x",
                    "value": rel,
                    "s_x": sx,
                    "rms_eps": rms,
                    "note": "injected noise / cloud scale; not identified as decoder residual",
                }
            )
    dfull = prior["d_full"]
    for _, r in dfull.iterrows():
        mse = float(r.get("anchor_mse", np.nan))
        rows.append(
            {
                "domain": "synthetic",
                "cell": r["cell"],
                "quantity": "decoder_anchor_rmse",
                "value": float(np.sqrt(max(mse, 0.0))),
                "note": "D=28 reconstruction RMSE at anchors; estimator residual scale comparator",
            }
        )
    # ViT-B existing
    phys = prior.get("phys_train") or {}
    recons = []
    if isinstance(phys, dict):
        for k, v in phys.items():
            if isinstance(v, dict) and "final_recon" in v:
                recons.append(float(v["final_recon"]))
            if k == "seeds" and isinstance(v, list):
                for s in v:
                    if isinstance(s, dict) and "final_recon" in s:
                        recons.append(float(s["final_recon"]))
        # nested lists
        for v in phys.values():
            if isinstance(v, list):
                for s in v:
                    if isinstance(s, dict) and "final_recon" in s:
                        recons.append(float(s["final_recon"]))
    recon = float(np.median(recons)) if recons else 0.0453
    vit_rmse = float(np.sqrt(max(recon, 0.0)))
    rows.append(
        {
            "domain": "vit_base",
            "cell": "physics_D_residual_run",
            "quantity": "decoder_recon_rmse_over_unit_signal",
            "value": vit_rmse,
            "final_recon_mse": recon,
            "note": "empirical estimator residual scale comparator only; not observational noise",
        }
    )
    fcr = prior.get("fcr")
    q_ratio = float("nan")
    if isinstance(fcr, pd.DataFrame):
        cols = {c.lower(): c for c in fcr.columns}
        rad = None
        resid = None
        for cand in ("radius", "q_radius", "rho", "neigh_radius", "radius_median"):
            if cand in cols:
                rad = fcr[cols[cand]]
                break
        for cand in ("quad_resid", "residual", "fit_rmse", "recon", "mse"):
            if cand in cols:
                resid = fcr[cols[cand]]
                break
        if rad is not None and resid is not None:
            q_ratio = float(np.nanmedian(np.abs(resid) / np.maximum(np.abs(rad), 1e-12)))
            rows.append({"domain": "vit_base", "quantity": "quadratic_residual_over_radius", "value": q_ratio, "note": "from existing FCR vit_base parquet"})
        else:
            rows.append({"domain": "vit_base", "quantity": "quadratic_residual_over_radius", "value": float("nan"), "note": "unresolved: no valid residual/radius columns in existing ViT-B artifacts", "columns": list(fcr.columns)[:40]})
    else:
        rows.append({"domain": "vit_base", "quantity": "quadratic_residual_over_radius", "value": float("nan"), "note": "unresolved: FCR parquet missing or unreadable"})
    rows.append({"domain": "vit_base", "quantity": "augmentation_embedding_variation", "value": float("nan"), "note": "unresolved: no existing augmentation-induced embedding variation artifact used"})

    # classify synthetic conditions vs vit decoder residual
    for cell, rms_frac in (
        ("F4_S0_N1", 0.01),
        ("F4_S0_N2", 0.05),
        ("F4_S0_N3", 0.05),
        ("F4_S3_N4", 0.03),
    ):
        cls = "below_observed_estimator_residual_scale"
        if rms_frac > 1.5 * vit_rmse:
            cls = "above_observed_estimator_residual_scale"
        elif abs(rms_frac - vit_rmse) / max(vit_rmse, 1e-6) < 0.5:
            cls = "comparable_to_observed_estimator_residual_scale"
        rows.append({"domain": "classification", "cell": cell, "synthetic_rms_over_s_x": rms_frac, "vit_decoder_rmse": vit_rmse, "class": cls, "note": "reconstruction residual is an empirical scale comparator only"})
    return rows


def _headline_md(repro, prior, d_full, q, dres, dims, rel_df, ceil_df, dens_df, noise_rows, decision, runtime):
    sparse = "n/a"
    if dens_df is not None and not dens_df.empty:
        s = dens_df[dens_df["quintile"] == 1]
        if not s.empty:
            sparse = "; ".join(f"{r.estimator} ρ={r.spearman:.3f}" for r in s.itertuples())
    return f"""
- D-full cubic/ridge reuse: ρ(R2)={repro['d_full_cubic_rho']:.3f}, ρ(R3)={repro['d_full_ridge_rho']:.3f}
- D-full F4 clean vector cosine={d_full['clean_vector_cosine']:.3f}; sphere rank degenerate={d_full['sphere_rank_degenerate']}
- D-full residualized F4 clean ρ={d_full['residualized_f4_clean_rho']}
- D-residual F4 clean ρ={dres['f4_clean_rho']:.3f}; stress ρ={dres['f4_stress_rho']:.3f}; Scal ρ={dres['f4_scal_rho']:.3f}
- Q T2 scalar K_H ρ={q['t2_kh_clean_rho']}; T2 tensor cos={q['t2_tensor_clean']:.3f}; T2–T3 cos={q['t2_t3_cos']:.3f}
- Repeat reliability / ceiling: see repeat_reliability.csv and reliability_ceiling.csv
- Sparsest density quintile: {sparse}
- Labels: D-full `{decision['d_full_summary_label']}`; Q `{decision['q_summary_label']}`; D-residual `{decision['d_residual_summary_label']}`
- Prior exact-recovery label unchanged: `{decision['prior_exact_recovery_label']}`
- Runtime {runtime.get('runtime_s')} s; new AEs {runtime.get('n_ae')}
"""
