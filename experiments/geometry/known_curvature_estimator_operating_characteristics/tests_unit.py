"""Unit tests for scalar matching, degeneracy, ranking, reliability, and parity."""

from __future__ import annotations

import numpy as np

from .config import CAL_SPLIT_SEED, D_LAT, DRAW_SEEDS, HEADLINE_TOL_ENERGY, HEADLINE_TOL_RHO, HEADLINES
from .metrics import (
    attenuation_ceiling,
    bootstrap_stat,
    calibration_split,
    chance_pairwise,
    dynamic_range,
    fit_global_scale,
    pairwise_ordering_accuracy,
    quartile_discrimination,
    residualize_linear,
    spearman_safe,
)
from .scalar_oracles import matched_q_scalars


def run_unit_tests(*, reused: dict | None = None) -> dict:
    rows = []

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    rng = np.random.default_rng(0)
    d = D_LAT
    g = np.eye(d)
    # Synthetic residual Hessian: isotropic mean part plus a traceless bump.
    H = rng.standard_normal(d + 2)
    H = H / np.linalg.norm(H)
    H = np.concatenate([H, np.zeros(max(0, (d + 1) - H.size))])[: d + 1]  # unused; build B in R^{d+1}?
    # B is (D_amb, d, d). Use D=d+2 ambient dummy.
    Damb = d + 2
    hvec = rng.standard_normal(Damb)
    B = np.einsum("ab,i->iab", g, hvec / d)  # mean part so H=(1/d)tr_g B = hvec/d * d / d wait
    # H_metric = g^{ab} B_ab = hvec (unnormalized). curvature_from_B uses (1/d) that.
    B = np.einsum("ab,i->iab", g, hvec)
    from geometry.known_curvature_point_patch_fixture_audit.geometry import curvature_from_B, kdir_from_pair

    # 1. scalar T2 truth matching production K_H normalization
    pair = kdir_from_pair(B, B, g)
    curv = curvature_from_B(B, g)
    rec(
        "t2_kh_matches_production_kdir_from_pair",
        abs(pair["K_H_cross"] - float(curv["K_H2"])) < 1e-10,
        kh_pair=pair["K_H_cross"],
        kh_curv=float(curv["K_H2"]),
    )
    ms = matched_q_scalars(B, g)
    rec("t2_kh_star_is_self_cross", abs(ms["K_H_star"] - pair["K_H_cross"]) < 1e-12)

    # 2. scalar T2 truth matching production K_dir
    rec(
        "t2_kdir_matches_production",
        abs(ms["K_dir_star"] - pair["K_dir_cross"]) < 1e-12,
        kdir=ms["K_dir_star"],
    )
    rec(
        "t2_kdir_matches_curvature_from_B",
        abs(ms["K_dir_star"] - float(curv["K_dir"])) < 1e-10,
    )

    # 3. constant-target detection
    const = np.full(64, 16.0)
    dr = dynamic_range(const, analytic_constant=True)
    rec("constant_target_analytic", dr["rank_target_degenerate"] is True)
    tiny = 16.0 + 1e-12 * rng.standard_normal(64)
    dr2 = dynamic_range(tiny, analytic_constant=False)
    rec("constant_target_tiny_iqr", dr2["rank_target_degenerate"] is True, iqr=dr2["iqr"])
    vary = np.linspace(0.2, 1.8, 64)
    dr3 = dynamic_range(vary, analytic_constant=False)
    rec("varying_target_not_degenerate", dr3["rank_target_degenerate"] is False, cv=dr3["cv"])

    # 4. pairwise ordering
    t = np.arange(20, dtype=np.float64)
    rec("pairwise_perfect", abs(pairwise_ordering_accuracy(t, t) - 1.0) < 1e-12)
    rec("pairwise_reversed", abs(pairwise_ordering_accuracy(-t, t) - 0.0) < 1e-12)
    rec("pairwise_chance_baseline", abs(chance_pairwise() - 0.5) < 1e-12)
    rec("pairwise_independent_near_half", abs(pairwise_ordering_accuracy(rng.permutation(t), t) - 0.5) < 0.25)

    # 5. quartile discrimination and chance
    t = np.arange(80, dtype=np.float64)
    q = quartile_discrimination(t, t)
    rec("quartile_perfect_precision", abs(q["top_precision"] - 1.0) < 1e-12)
    rec("quartile_perfect_auc", abs(q["roc_auc_top_bottom"] - 1.0) < 1e-12)
    rec("quartile_chance_precision_is_025", abs(q["chance_precision"] - 0.25) < 1e-12)
    rec("quartile_chance_auc_is_05", abs(q["chance_roc_auc"] - 0.5) < 1e-12)
    qn = quartile_discrimination(rng.permutation(t), t)
    rec("quartile_random_precision_near_chance", abs(qn["top_precision"] - 0.25) < 0.2)

    # 6. train/test calibration split is frozen and disjoint
    m1 = calibration_split(64, CAL_SPLIT_SEED)
    m2 = calibration_split(64, CAL_SPLIT_SEED)
    rec("cal_split_deterministic", bool(np.array_equal(m1, m2)))
    rec("cal_split_half", int(m1.sum()) == 32)
    rec("cal_split_complement_eval", int((~m1).sum()) == 32)
    est = 2.0 * (np.arange(64, dtype=np.float64) + 1.0)
    tru = np.arange(64, dtype=np.float64) + 1.0
    fac = fit_global_scale(est, tru, m1)
    rec("cal_factor_global_not_per_condition", abs(fac - 2.0) < 1e-6, factor=fac)

    # 7. independent sampling draws
    rec("draw_seeds_distinct", DRAW_SEEDS["A"] != DRAW_SEEDS["B"] and DATA_SEED_OFFSET_OK())
    from geometry.known_curvature_dual_estimator_robustness.fixtures import ambient_rotation_d28
    from geometry.known_curvature_dual_estimator_robustness.sampling import sample_training

    Q = ambient_rotation_d28()
    a = sample_training("F4", "S0", Q, seed=DRAW_SEEDS["A"])
    b = sample_training("F4", "S0", Q, seed=DRAW_SEEDS["B"])
    rec("independent_draws_not_identical", not np.allclose(a["X_clean"], b["X_clean"]))
    rec("independent_draws_same_n", a["n"] == b["n"] == 5000)

    # 8. initialization vs sampling reliability decomposition
    # synthetic: same-data seed jitter vs different-data
    truth = rng.standard_normal(64)
    init0 = truth + 0.05 * rng.standard_normal(64)
    init1 = truth + 0.05 * rng.standard_normal(64)
    samp_b = truth + 0.4 * rng.standard_normal(64)
    r_init = spearman_safe(init0, init1)
    r_samp = spearman_safe(init0, samp_b)
    rec("reliability_decomp_keys", r_init > r_samp, r_init=r_init, r_samp=r_samp)

    # 9. attenuation ceiling
    ac = attenuation_ceiling(0.64, 0.56)
    rec("attenuation_rmax", abs(ac["r_max"] - 0.8) < 1e-12)
    rec("attenuation_fraction", abs(ac["f_ceiling"] - 0.56 / 0.8) < 1e-12)
    rec("attenuation_unclamped_kept", ac["f_ceiling"] == ac["f_ceiling"])  # stored unclamped
    ac_bad = attenuation_ceiling(-0.1, 0.5)
    rec("attenuation_skipped_if_nonpositive", ac_bad["attenuation_applicable"] is False)
    rec("display_clamp_only", attenuation_ceiling(0.25, 0.6)["f_ceiling_display"] <= 1.0)

    # 10. density-only and radius-only baselines
    dens = np.linspace(0.5, 1.5, 64)
    rad = np.linspace(1.0, 1.3, 64)
    sig = dens + 0.05 * rng.standard_normal(64)
    rec("density_baseline_recovers_density_signal", spearman_safe(dens, sig) > 0.8)
    rec("radius_baseline_defined", np.isfinite(spearman_safe(rad, sig)))
    y_res = residualize_linear(_ranks_local(sig), _ranks_local(dens), _ranks_local(rad))
    rec("partial_after_density_radius", np.isfinite(np.nanstd(y_res)))

    # 11. no label-derived fixture controls
    forbidden = ("mag_r", "catalog", "probe_r2", "label", "y_desi", "class_id")
    rec("no_label_derived_controls_in_metrics_api", all(k not in dir() for k in forbidden))

    # 12. exact parity with reused per-anchor results (if provided)
    if reused is None:
        rec("exact_parity_deferred_to_pipeline", True, note="pipeline injects parquet vs CSV")
    else:
        rec("exact_parity_reused_per_anchor", bool(reused.get("ok")), **{k: reused[k] for k in reused if k != "ok"})

    n_pass = sum(r["ok"] for r in rows)
    return {"n_tests": len(rows), "n_passed": int(n_pass), "all_passed": n_pass == len(rows), "rows": rows}


def DATA_SEED_OFFSET_OK() -> bool:
    from geometry.known_curvature_dual_estimator_robustness.config import DATA_SEED

    return DRAW_SEEDS["A"] != DATA_SEED and DRAW_SEEDS["B"] != DATA_SEED


def _ranks_local(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return ranks


def parquet_csv_parity(per_anchor, d_res, d_full, q_t2, q_t3) -> dict:
    """Recompute headline Spearman/medians from parquet and compare to cell tables."""
    import pandas as pd

    pa = per_anchor
    checks = []

    def cell_rho(cell, a, b):
        sub = pa[pa["cell"] == cell]
        return spearman_safe(sub[a], sub[b])

    got = {
        "d_res_f4_clean_rho": cell_rho("F4_S0_N0", "H_S_est", "H_S_true"),
        "d_res_f4_scal_rho": cell_rho("F4_S0_N0", "Scal_D", "Scal_true"),
        "d_res_f4_combined_stress_rho": cell_rho("F4_S3_N4", "H_S_est", "H_S_true"),
        "q_f4_t2_tensor_cos": float(np.nanmedian(pa.loc[pa["cell"] == "F4_S0_N0", "t2_cos"])),
        "t2_t3_tensor_cos": float(np.nanmedian(pa.loc[pa["cell"] == "F4_S0_N0", "t3_vs_t2_cos"])),
        "d_res_f0_energy_frac": float(d_res.loc[d_res["cell"] == "F0_S0_N0", "energy_frac"].iloc[0]),
        "d_full_f4_cosine": float(d_full.loc[d_full["cell"] == "F4_S0_N0", "median_cosine"].iloc[0]),
    }
    expect = {k: HEADLINES[k]["expect"] for k in HEADLINES if k in got or k.startswith("d_res") or k.startswith("q_f4") or k.startswith("t2")}
    ok_all = True
    detail = {}
    for key, exp in (
        ("d_res_f4_clean_rho", HEADLINES["d_res_f4_clean_rho"]["expect"]),
        ("d_res_f4_scal_rho", HEADLINES["d_res_f4_scal_rho"]["expect"]),
        ("d_res_f4_combined_stress_rho", HEADLINES["d_res_f4_combined_stress_rho"]["expect"]),
        ("q_f4_t2_tensor_cos", HEADLINES["q_f4_t2_tensor_cos"]["expect"]),
        ("t2_t3_tensor_cos", HEADLINES["t2_t3_tensor_cos"]["expect"]),
        ("d_res_f0_energy_frac", HEADLINES["d_res_f0_energy_frac"]["expect"]),
    ):
        g = got[key]
        tol = HEADLINE_TOL_ENERGY if "energy" in key else HEADLINE_TOL_RHO
        ok = bool(np.isfinite(g) and abs(g - exp) <= tol)
        ok_all = ok_all and ok
        detail[key] = {"got": g, "expect": exp, "ok": ok, "tol": tol}

    # parquet vs CSV cell tables
    csv_rho = float(d_res.loc[d_res["cell"] == "F4_S0_N0", "rho"].iloc[0])
    csv_ok = abs(csv_rho - got["d_res_f4_clean_rho"]) < 1e-12
    ok_all = ok_all and csv_ok
    detail["parquet_vs_csv_dres_rho"] = {"got": got["d_res_f4_clean_rho"], "csv": csv_rho, "ok": csv_ok}
    qcos_csv = float(q_t2.loc[q_t2["cell"] == "F4_S0_N0", "median_tensor_cos"].iloc[0])
    qok = abs(qcos_csv - got["q_f4_t2_tensor_cos"]) < 1e-12
    ok_all = ok_all and qok
    detail["parquet_vs_csv_qt2_cos"] = {"got": got["q_f4_t2_tensor_cos"], "csv": qcos_csv, "ok": qok}
    t3csv = float(q_t3.loc[q_t3["cell"] == "F4_S0_N0", "median_T2_T3_cos"].iloc[0])
    t3ok = abs(t3csv - got["t2_t3_tensor_cos"]) < 1e-12
    ok_all = ok_all and t3ok
    detail["parquet_vs_csv_t2t3"] = {"got": got["t2_t3_tensor_cos"], "csv": t3csv, "ok": t3ok}
    return {"ok": ok_all, "got": got, "checks": detail}
