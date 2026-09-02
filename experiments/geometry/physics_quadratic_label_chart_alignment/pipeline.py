"""Orchestrate quadratic-label chart alignment experiment."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix, freedman_lane_y

from .alignment import alignment_AB, fit_uq_gamma_oof, random_gamma_null
from .config import PRIMARY_D, PROBE_ALPHA, ExpConfig
from .data import build_reuse_manifest, kh_row, load_bundle, load_chart, tangent_coords
from .decision import decide
from .features import bs_prod_to_frob, verify_n_quad
from .figures import write_figures
from .io_util import assert_not_preserved, p_mc, platonic_root, resolve_path, write_df, write_json
from .models import fit_A_flat_fast, mae, mse, oof_ambient_P, oof_predict_model, r2
from .parity import run_parity
from .synthetic import run_synthetics


def _process_anchor(payload: dict) -> tuple[dict, dict]:
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_k] = "1"

    sid = int(payload["sid"])
    Xloc = payload["Xloc"]
    yloc = payload["yloc"]
    floc = payload["floc"]
    g_pred = payload["g_pred"]
    chart = payload["chart"]
    U = tangent_coords(Xloc, chart["x0"], chart["J"])

    A_flat = fit_A_flat_fast(Xloc, chart["x0"], chart["J"], alpha=0.1)
    if A_flat is not None:
        Q_prod = chart["J"] @ A_flat + chart["BS_mean_prod"]
        Q_frob = bs_prod_to_frob(Q_prod, PRIMARY_D)
        QT_frob = bs_prod_to_frob(chart["J"] @ A_flat, PRIMARY_D)
    else:
        Q_frob = QT_frob = None

    yL, _ = oof_predict_model(U, yloc, floc, kind="L")
    yUQ, _ = oof_predict_model(U, yloc, floc, kind="UQ")
    yBS, _ = oof_predict_model(U, yloc, floc, kind="BS", BS_frob=chart["BS_mean_frob"])
    if Q_frob is not None:
        yFQ, _ = oof_predict_model(U, yloc, floc, kind="FQ", Q_frob=Q_frob)
        yFQm, _ = oof_predict_model(U, yloc, floc, kind="FQ", Q_frob=QT_frob)
    else:
        yFQ = yFQm = np.full(len(yloc), np.nan)

    yP = oof_ambient_P(Xloc, yloc, floc, alpha=PROBE_ALPHA)

    mse_L, mse_UQ = mse(yloc, yL), mse(yloc, yUQ)
    mse_BS, mse_FQ = mse(yloc, yBS), mse(yloc, yFQ)
    mse_FQm = mse(yloc, yFQm)
    mse_G, mse_P = mse(yloc, g_pred), mse(yloc, yP)

    gamma, stab = fit_uq_gamma_oof(U, yloc, floc)
    aB = alignment_AB(gamma, chart["BS_mean_frob"])
    aBA = alignment_AB(gamma, chart["BS_A_frob"])
    aBB = alignment_AB(gamma, chart["BS_B_frob"])

    row = {
        **payload["kh"],
        "mse_L": mse_L,
        "mse_UQ": mse_UQ,
        "mse_BS": mse_BS,
        "mse_FQ": mse_FQ,
        "mse_FQ_minus_BS": mse_FQm,
        "mse_G": mse_G,
        "mse_P": mse_P,
        "delta_Q": mse_L - mse_UQ,
        "delta_BS": mse_L - mse_BS,
        "delta_FQ": mse_L - mse_FQ,
        "delta_normal": (mse_FQm - mse_FQ) if np.isfinite(mse_FQm) and np.isfinite(mse_FQ) else float("nan"),
        "dMSE_G_to_P": mse_G - mse_P,
        "mae_L": mae(yloc, yL),
        "mae_UQ": mae(yloc, yUQ),
        "r2_L": r2(yloc, yL),
        "r2_UQ": r2(yloc, yUQ),
        "A_B": aB,
        "A_B_A": aBA,
        "A_B_B": aBB,
        "gamma_fold_cosine": stab,
        "n_eval": int(np.sum(np.isfinite(yloc) & np.isfinite(yL))),
    }
    align = {
        "sample_id": sid,
        "A_B": aB,
        "A_B_A": aBA,
        "A_B_B": aBB,
        "gamma_fold_cosine": stab,
        "A_B_agree": float(1.0 - abs(aBA - aBB) / max(abs(aBA) + abs(aBB), 1e-8))
        if np.isfinite(aBA) and np.isfinite(aBB)
        else float("nan"),
    }
    return row, align


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    root = platonic_root()
    out = resolve_path(root, cfg.output_dir)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "CONFIG.json", asdict(cfg), force=cfg.force)

    verify_n_quad()
    bundle = load_bundle(cfg)
    man = build_reuse_manifest(bundle)
    write_json(out / "reuse_manifest.json", man, force=cfg.force)

    parity = run_parity(bundle, cfg, out)
    if not parity.get("ok"):
        write_json(out / "BLOCKER.json", {"reason": "parity_failed", "parity": parity}, force=True)
        raise RuntimeError("Phase 0 parity failed — refusing scientific inference")

    print("[qlca] synthetics…", flush=True)
    synth = run_synthetics(seed=cfg.seed)
    write_json(out / "synthetic_results.json", synth, force=cfg.force)
    print(f"[qlca] synthetics ok={synth.get('ok')}", flush=True)

    X = bundle["X"]
    y = bundle["y"]
    yhat_g = bundle["yhat"]
    fold_all = bundle["fold"]
    neigh = bundle["neigh"]
    ndc = bundle["ndc"]

    payloads = []
    for sid in bundle["sids"]:
        ai = bundle["sid_to_ai"][int(sid)]
        idx = np.asarray(neigh[ai], dtype=int)
        chart = load_chart(ndc, int(sid))
        payloads.append(
            {
                "sid": int(sid),
                "Xloc": np.asarray(X[idx], dtype=np.float64),
                "yloc": y[idx].copy(),
                "floc": fold_all[idx].copy(),
                "g_pred": yhat_g[idx].copy(),
                "chart": chart,
                "seed": cfg.seed,
                "kh": kh_row(bundle, int(sid)),
            }
        )

    rows: list[dict] = []
    align_rows: list[dict] = []
    n_workers = 1 if cfg.smoke else min(8, max(1, (os.cpu_count() or 8) // 2))
    print(f"[qlca] fitting {len(payloads)} anchors with {n_workers} workers", flush=True)

    if n_workers == 1:
        for i, p in enumerate(payloads):
            row, al = _process_anchor(p)
            rows.append(row)
            align_rows.append(al)
            if (i + 1) % 8 == 0 or i == 0:
                print(f"[qlca] anchor {i+1}/{len(payloads)} sid={p['sid']}", flush=True)
                write_df(out / "anchor_risks_partial.csv", pd.DataFrame(rows), force=True)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_process_anchor, p): p["sid"] for p in payloads}
            for fut in as_completed(futs):
                row, al = fut.result()
                rows.append(row)
                align_rows.append(al)
                done += 1
                if done % 8 == 0 or done == 1:
                    print(f"[qlca] completed {done}/{len(payloads)} (last sid={futs[fut]})", flush=True)
                    write_df(out / "anchor_risks_partial.csv", pd.DataFrame(rows), force=True)

    anchor = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    write_df(out / "anchor_risks.csv", anchor, force=cfg.force)
    write_df(out / "chart_alignment.csv", pd.DataFrame(align_rows).sort_values("sample_id"), force=cfg.force)

    print("[qlca] primary inference…", flush=True)
    primary = _primary_tests(anchor, cfg)
    write_json(out / "primary_inference.json", primary, force=cfg.force)

    secondary = _secondary(anchor, cfg)
    write_json(out / "secondary_inference.json", secondary, force=cfg.force)

    null_meds = []
    for sid in bundle["sids"][: min(64, len(bundle["sids"]))]:
        ch = load_chart(ndc, int(sid))
        null = random_gamma_null(ch["BS_mean_frob"], n=20, seed=cfg.seed + int(sid))
        null_meds.append(float(np.nanmedian(null)))
    align_summary = {
        "A_B_median": float(np.nanmedian(anchor.A_B)),
        "A_B_null_median": float(np.nanmedian(null_meds)) if null_meds else 1.0,
        "gamma_fold_cosine_median": float(np.nanmedian(anchor.gamma_fold_cosine)),
        "frac_stable": float(np.mean(anchor.gamma_fold_cosine.fillna(0) >= 0.5)),
    }
    write_json(out / "alignment_summary.json", align_summary, force=cfg.force)

    unstable = bool(align_summary["frac_stable"] < 0.25 and primary.get("median_delta_Q", 0) > 0)
    decision = decide(
        primary=primary,
        secondary=secondary,
        alignment=align_summary,
        synth_ok=bool(synth.get("ok")),
        unstable=unstable,
    )
    write_json(out / "decision.json", decision, force=cfg.force)

    write_figures(out, anchor, primary)
    _write_methods(out)
    _write_report(out, parity, primary, secondary, align_summary, decision, synth, t0, cfg)

    summary = {
        "label": decision["label"],
        "median_delta_Q": primary.get("median_delta_Q"),
        "rho_KH_delta_Q": primary.get("rho_KH_delta_Q"),
        "median_delta_BS": secondary.get("median_delta_BS"),
        "seconds": time.time() - t0,
        "n_anchors": len(anchor),
        "smoke": cfg.smoke,
        "n_workers": n_workers,
    }
    write_json(out / "summary.json", summary, force=cfg.force)

    if not cfg.smoke and parity.get("ok"):
        import resource

        write_json(
            out / "COMPLETE.json",
            {
                "ok": True,
                "label": decision["label"],
                "seconds": time.time() - t0,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "n_perm": cfg.n_perm_eff(),
                "n_boot": cfg.n_boot_eff(),
                "n_workers": n_workers,
            },
            force=cfg.force,
        )
    print(f"[qlca] done label={decision['label']} s={time.time()-t0:.1f}", flush=True)
    return decision


def _primary_tests(anchor: pd.DataFrame, cfg: ExpConfig) -> dict:
    dq = anchor.delta_Q.to_numpy(float)
    m = np.isfinite(dq)
    dq = dq[m]
    med = float(np.median(dq)) if len(dq) else float("nan")
    rng = np.random.default_rng(cfg.seed)
    B = cfg.n_boot_eff()
    boots = np.empty(B)
    for b in range(B):
        samp = rng.choice(dq, size=len(dq), replace=True)
        boots[b] = float(np.median(samp))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p_pos = float(np.mean(dq > 0)) if len(dq) else float("nan")

    Z = control_matrix(anchor)
    assoc = associate(anchor.K_H_cross.to_numpy(float), anchor.delta_Q.to_numpy(float), Z)
    n_perm = cfg.n_perm_eff()
    yv = anchor.delta_Q.to_numpy(float)
    xv = anchor.K_H_cross.to_numpy(float)
    null = np.empty(n_perm)
    rng2 = np.random.default_rng(cfg.seed + 7)
    for b in range(n_perm):
        yp = freedman_lane_y(yv, Z, rng2)
        null[b] = associate(xv, yp, Z)["controlled"]
    obs = float(assoc["controlled"])
    b_count = int(np.sum(null >= obs)) if obs >= 0 else int(np.sum(null <= obs))
    p2 = p_mc(b_count, n_perm)

    b_neg = int(np.sum(boots <= 0))
    p1 = p_mc(b_neg, B)
    ps = sorted([(p1, "median_delta_Q"), (p2, "rho_KH_delta_Q")])
    holm = {name: min(1.0, p * (2 - i)) for i, (p, name) in enumerate(ps)}
    holm_pass = holm["median_delta_Q"] <= 0.05 and holm["rho_KH_delta_Q"] <= 0.05

    return {
        "median_delta_Q": med,
        "mean_delta_Q": float(np.mean(dq)) if len(dq) else float("nan"),
        "delta_Q_ci_lo": float(lo),
        "delta_Q_ci_hi": float(hi),
        "frac_positive_delta_Q": p_pos,
        "p_mc_median_delta_Q": p1,
        "rho_KH_delta_Q": obs,
        "rho_KH_delta_Q_raw": float(assoc["raw"]),
        "rho_KH_delta_Q_p_mc": p2,
        "n_perm": n_perm,
        "n_boot": B,
        "holm": holm,
        "holm_both_pass": holm_pass,
        "n": int(m.sum()),
    }


def _secondary(anchor: pd.DataFrame, cfg: ExpConfig) -> dict:
    out: dict = {}
    for col, key in [
        ("delta_BS", "median_delta_BS"),
        ("delta_FQ", "median_delta_FQ"),
        ("delta_normal", "median_delta_normal"),
    ]:
        v = anchor[col].to_numpy(float) if col in anchor else np.array([])
        v = v[np.isfinite(v)]
        out[key] = float(np.median(v)) if len(v) else float("nan")
        out[key.replace("median", "mean")] = float(np.mean(v)) if len(v) else float("nan")

    m = np.isfinite(anchor.delta_Q) & np.isfinite(anchor.delta_BS) & (anchor.delta_Q > 0)
    if m.sum() >= 8:
        out["frac_UQ_captured_by_BS"] = float(
            np.median(np.clip(anchor.delta_BS[m] / anchor.delta_Q[m], -1, 2))
        )
    else:
        out["frac_UQ_captured_by_BS"] = float("nan")

    Z = control_matrix(anchor)
    for col, name in [("delta_BS", "rho_KH_delta_BS"), ("delta_FQ", "rho_KH_delta_FQ")]:
        if col in anchor.columns:
            out[name] = associate(anchor.K_H_cross.to_numpy(float), anchor[col].to_numpy(float), Z)[
                "controlled"
            ]

    if "dMSE_G_to_P" in anchor.columns:
        base = associate(anchor.K_H_cross.to_numpy(float), anchor.dMSE_G_to_P.to_numpy(float), Z)
        from scipy.stats import rankdata

        mm = np.isfinite(anchor.dMSE_G_to_P) & np.isfinite(anchor.delta_Q) & np.isfinite(anchor.K_H_cross)
        if mm.sum() >= 20:
            yy = rankdata(anchor.dMSE_G_to_P[mm])
            dq = rankdata(anchor.delta_Q[mm])
            A = np.column_stack([np.ones(int(mm.sum())), dq - dq.mean()])
            b, *_ = np.linalg.lstsq(A, yy - yy.mean(), rcond=None)
            resid = yy - yy.mean() - A @ b
            sub = anchor.loc[mm].copy()
            sub["_resid"] = resid
            adj = associate(sub.K_H_cross.to_numpy(float), sub._resid.to_numpy(float), control_matrix(sub))
            out["rho_KH_dMSE_GP"] = base["controlled"]
            out["rho_KH_dMSE_GP_adj_deltaQ"] = adj["controlled"]
    return out


def _write_methods(out: Path) -> None:
    (out / "METHODS.md").write_text(
        """# METHODS — quadratic label chart alignment

## Frozen geometry
- Charts: NDC `H_vectors` (`J16`, `BS16_A/B`, `x0`); FQ uses fast `A_flat` + stored `B^S`
- `K_H_cross` from CPRS; MM folds/neighbours/OOF

## Models
Gram-cached nested CV; Frobenius φ₂ (q=136 at d=16); L/UQ/BS/FQ; G/P α=100 parity.

## Primary
median Δ_Q>0 (B=2000 bootstrap); ρ_ctl(K_H,Δ_Q)>0 (B=10000 FL); Holm.
"""
    )


def _write_report(out, parity, primary, secondary, align, decision, synth, t0, cfg) -> None:
    (out / "REPORT.md").write_text(
        f"""# REPORT — quadratic label chart alignment

## Decision
`{decision['label']}`

## Parity
ok={parity.get('ok')} ρ(R²)={parity['rho_r2_G']['controlled']:.3f} ρ(MSE)={parity['rho_mse_G']['controlled']:.3f} ρ(ΔMSE)={parity['rho_dMSE_GP']['controlled']:.3f}

## Primary
median Δ_Q={primary.get('median_delta_Q')} CI [{primary.get('delta_Q_ci_lo')}, {primary.get('delta_Q_ci_hi')}]
ρ_ctl(K_H, Δ_Q)={primary.get('rho_KH_delta_Q')} p_MC={primary.get('rho_KH_delta_Q_p_mc')} Holm={primary.get('holm_both_pass')}

## Secondary
Δ_BS={secondary.get('median_delta_BS')} Δ_FQ={secondary.get('median_delta_FQ')} frac_BS={secondary.get('frac_UQ_captured_by_BS')}

## Alignment
A_B={align.get('A_B_median')} null={align.get('A_B_null_median')} γ-stab={align.get('gamma_fold_cosine_median')}

## Synthetics
ok={synth.get('ok')}

Runtime {time.time()-t0:.1f}s smoke={cfg.smoke}
"""
    )
