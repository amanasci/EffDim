"""ViT-B Hessian split, component alignment, and held-out IQ/TQ/UQ-v2/BSH/BSTF."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from geometry.physics_quadratic_label_chart_alignment.alignment import (
    alignment_AB,
    fit_uq_gamma_oof,
)
from geometry.physics_quadratic_label_chart_alignment.config import PRIMARY_D
from geometry.physics_quadratic_label_chart_alignment.config import ExpConfig as QlcaCfg
from geometry.physics_quadratic_label_chart_alignment.data import load_bundle, load_chart, tangent_coords
from geometry.physics_quadratic_label_chart_alignment.features import Gamma_from_gamma
from geometry.physics_quadratic_label_chart_alignment.models import mse

from .config import STAB_MIN, ExpConfig
from .decompose import bh_btf_frob, induced_energy, project_gamma_iso_tf, split_Gamma
from .io_util import write_df
from .probes import oof_component_model, verify_quadratic_span

_STAB = STAB_MIN


def random_gamma_null_fast(BS_frob: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Haar / Gaussian γ on the Frobenius sphere; BtB is computed once."""
    B = np.asarray(BS_frob, dtype=np.float64)
    BtB = B.T @ B
    tr = float(np.trace(BtB))
    q = B.shape[1]
    if tr < 1e-18 or n <= 0:
        return np.full(max(n, 0), np.nan)
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, q))
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    vals = np.einsum("ij,jk,ik->i", G, BtB, G)
    return q * vals / tr


def _fold_component_stability(gammas: list[np.ndarray], d: int) -> dict[str, float]:
    if len(gammas) < 2:
        return {"stab_H": float("nan"), "stab_TF": float("nan"), "stab_full": float("nan")}
    Hs, TFs = [], []
    for g in gammas:
        gH, gTF = project_gamma_iso_tf(g, d)
        Hs.append(gH)
        TFs.append(gTF)

    def _med_cos(xs):
        cos = []
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                a, b = xs[i], xs[j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-12 and nb > 1e-12:
                    cos.append(float(a @ b / (na * nb)))
        return float(np.median(cos)) if cos else float("nan")

    return {"stab_H": _med_cos(Hs), "stab_TF": _med_cos(TFs), "stab_full": _med_cos(gammas)}


def _foldwise_gammas(U, y, fold) -> list[np.ndarray]:
    from geometry.physics_quadratic_label_chart_alignment.models import _design_UQ, _ridge_block, _scalar_rms

    gammas = []
    for f in sorted(set(fold.tolist())):
        tr = fold != f
        if tr.sum() < 32:
            continue
        s = max(_scalar_rms(U[tr]), 1e-8)
        Xtr = _design_UQ(U[tr] / s)
        w, b, info = _ridge_block(Xtr, y[tr], n_lin=PRIMARY_D, alpha_lin=100.0, alpha_quad=1000.0)
        if not info.get("ok"):
            continue
        gammas.append(w[PRIMARY_D:] / (s * s))
    return gammas


def process_anchor(payload: dict) -> dict:
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_k] = "1"
    sid = int(payload["sid"])
    Xloc = payload["Xloc"]
    yloc = payload["yloc"]
    floc = payload["floc"]
    chart = payload["chart"]
    n_haar = int(payload["n_haar"])
    skip_probes = bool(payload.get("skip_probes", False))
    U = tangent_coords(Xloc, chart["x0"], chart["J"])
    d = PRIMARY_D

    gamma, stab_full = fit_uq_gamma_oof(U, yloc, floc)
    if np.all(np.isfinite(gamma)):
        G = Gamma_from_gamma(gamma, d)
        GH, GTF = split_Gamma(G)
        gnorm2 = float(np.sum(G * G))
        frac_H = float(np.sum(GH * GH) / gnorm2) if gnorm2 > 1e-18 else float("nan")
        frac_TF = float(np.sum(GTF * GTF) / gnorm2) if gnorm2 > 1e-18 else float("nan")
        tr_tf = float(np.trace(GTF))
        inner = float(np.sum(GH * GTF))
    else:
        frac_H = frac_TF = tr_tf = inner = float("nan")

    folds_g = _foldwise_gammas(U, yloc, floc)
    st = _fold_component_stability(folds_g, d)

    BH_A, BTF_A = bh_btf_frob(chart["BS_A_prod"], d)
    BH_B, BTF_B = bh_btf_frob(chart["BS_B_prod"], d)
    BH = 0.5 * (BH_A + BH_B)
    BTF = 0.5 * (BTF_A + BTF_B)
    BS = chart["BS_mean_frob"]

    aB = alignment_AB(gamma, BS)
    aH = alignment_AB(gamma, BH)
    aTF = alignment_AB(gamma, BTF)
    aH_A, aH_B = alignment_AB(gamma, BH_A), alignment_AB(gamma, BH_B)
    aTF_A, aTF_B = alignment_AB(gamma, BTF_A), alignment_AB(gamma, BTF_B)
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if np.all(np.isfinite(g)):
        cross = float(2.0 * g @ (BH.T @ BTF) @ g)
        eH = induced_energy(g, BH)
        eTF = induced_energy(g, BTF)
        eB = induced_energy(g, BS)
    else:
        cross = eH = eTF = eB = float("nan")

    null_H = random_gamma_null_fast(BH, n_haar, seed=int(payload["seed"]) + 11 * sid)
    null_TF = random_gamma_null_fast(BTF, n_haar, seed=int(payload["seed"]) + 13 * sid)
    null_B = random_gamma_null_fast(BS, n_haar, seed=int(payload["seed"]) + 17 * sid)

    row = {
        "sample_id": sid,
        "gamma_fold_cosine": stab_full,
        "gamma_H_fold_cosine": st["stab_H"],
        "gamma_TF_fold_cosine": st["stab_TF"],
        "frac_Gamma_H": frac_H,
        "frac_Gamma_TF": frac_TF,
        "tr_GTF": tr_tf,
        "inner_GH_GTF": inner,
        "A_B": aB,
        "A_H": aH,
        "A_TF": aTF,
        "A_H_A": aH_A,
        "A_H_B": aH_B,
        "A_TF_A": aTF_A,
        "A_TF_B": aTF_B,
        "A_H_null_median": float(np.nanmedian(null_H)),
        "A_TF_null_median": float(np.nanmedian(null_TF)),
        "A_B_null_median": float(np.nanmedian(null_B)),
        "A_H_null_p95": float(np.nanpercentile(null_H, 95)),
        "A_TF_null_p95": float(np.nanpercentile(null_TF, 95)),
        "A_B_null_p95": float(np.nanpercentile(null_B, 95)),
        "induced_EH": eH,
        "induced_ETF": eTF,
        "induced_EB": eB,
        "induced_cross": cross,
        "stable_full": bool(np.isfinite(stab_full) and stab_full >= _STAB),
        "stable_H": bool(np.isfinite(st["stab_H"]) and st["stab_H"] >= _STAB),
        "stable_TF": bool(np.isfinite(st["stab_TF"]) and st["stab_TF"] >= _STAB),
    }

    if skip_probes:
        return row

    yL, dL = oof_component_model(U, yloc, floc, kind="L")
    yIQ, dIQ = oof_component_model(U, yloc, floc, kind="IQ", omit_quad=True)
    yTQ, dTQ = oof_component_model(U, yloc, floc, kind="TQ", omit_quad=True)
    yUQ2, dUQ2 = oof_component_model(U, yloc, floc, kind="UQ2", omit_quad=True)
    yBSH, dBSH = oof_component_model(U, yloc, floc, kind="BSH", BS_frob=BH, cap=48)
    yBSTF, dBSTF = oof_component_model(U, yloc, floc, kind="BSTF", BS_frob=BTF, cap=48)
    yBS, dBS = oof_component_model(U, yloc, floc, kind="BS", BS_frob=BS, cap=48)

    mse_L = mse(yloc, yL)
    mse_IQ, mse_TQ, mse_UQ2 = mse(yloc, yIQ), mse(yloc, yTQ), mse(yloc, yUQ2)
    mse_BSH, mse_BSTF, mse_BS = mse(yloc, yBSH), mse(yloc, yBSTF), mse(yloc, yBS)
    row.update(
        {
            "mse_L": mse_L,
            "mse_IQ": mse_IQ,
            "mse_TQ": mse_TQ,
            "mse_UQ2": mse_UQ2,
            "mse_BSH": mse_BSH,
            "mse_BSTF": mse_BSTF,
            "mse_BS_v2": mse_BS,
            "delta_IQ": mse_L - mse_IQ,
            "delta_TQ": mse_L - mse_TQ,
            "delta_UQ2": mse_L - mse_UQ2,
            "delta_BSH": mse_L - mse_BSH,
            "delta_BSTF": mse_L - mse_BSTF,
            "delta_BS_v2": mse_L - mse_BS,
            "mse_IQ_minus_UQ2": mse_IQ - mse_UQ2,
            "mse_TQ_minus_UQ2": mse_TQ - mse_UQ2,
            "rank_BSH": dBSH.get("algebraic_rank"),
            "rank_BSTF": dBSTF.get("algebraic_rank"),
            "rank_BS": dBS.get("algebraic_rank"),
            "n_comp_BSH": dBSH.get("n_comp"),
            "n_comp_BSTF": dBSTF.get("n_comp"),
            "n_comp_BS": dBS.get("n_comp"),
            "energy_BSH": dBSH.get("energy_captured"),
            "energy_BSTF": dBSTF.get("energy_captured"),
            "energy_BS": dBS.get("energy_captured"),
            "mode_C_trace_BSH": dBSH.get("median_mode_C_trace"),
            "mode_C_trace_BSTF": dBSTF.get("median_mode_C_trace"),
            "mode_C_trace_BS": dBS.get("median_mode_C_trace"),
            "edf_IQ": dIQ.get("median_edf"),
            "edf_TQ": dTQ.get("median_edf"),
            "edf_UQ2": dUQ2.get("median_edf"),
            "alpha_quad_IQ": dIQ.get("median_alpha_quad"),
            "alpha_quad_TQ": dTQ.get("median_alpha_quad"),
            "alpha_quad_UQ2": dUQ2.get("median_alpha_quad"),
            "frac_omit_UQ2": dUQ2.get("frac_omit_quad"),
            "pred_corr_BSH_IQ": float(
                np.corrcoef(
                    yBSH[np.isfinite(yBSH) & np.isfinite(yIQ)],
                    yIQ[np.isfinite(yBSH) & np.isfinite(yIQ)],
                )[0, 1]
            )
            if np.sum(np.isfinite(yBSH) & np.isfinite(yIQ)) > 8
            else float("nan"),
        }
    )
    # energy-rule ranks (no extra label fits)
    for name, Bf, prefix in (("BSH", BH, "BSH"), ("BSTF", BTF, "BSTF"), ("BS", BS, "BS")):
        S = np.linalg.svd(Bf, compute_uv=False)
        energy = np.cumsum(S * S) / max(float(np.sum(S * S)), 1e-18)
        for frac, tag in ((0.90, "90"), (0.95, "95"), (0.99, "99")):
            r = int(np.searchsorted(energy, frac) + 1)
            row[f"n_comp_{prefix}_{tag}"] = int(max(1, min(r, S.size)))
    _ = name
    return row


def run_vitb_pass(shared: dict, cfg: ExpConfig, out, sids: list[int]) -> pd.DataFrame:
    span = verify_quadratic_span(PRIMARY_D)
    if span["n_tf"] != 135 or span["span_err"] > 1e-8:
        raise RuntimeError(f"IQ/TQ span failed: {span}")

    qcfg = QlcaCfg(n_anchors_override=512)
    bundle = load_bundle(qcfg)
    use = [s for s in sids if int(s) in bundle["sid_to_ai"]]
    payloads = []
    X, y, fold_all, neigh = bundle["X"], bundle["y"], bundle["fold"], bundle["neigh"]
    for sid in use:
        ai = bundle["sid_to_ai"][int(sid)]
        idx = np.asarray(neigh[ai], dtype=int)
        payloads.append(
            {
                "sid": int(sid),
                "Xloc": np.asarray(X[idx], dtype=np.float64),
                "yloc": y[idx].copy(),
                "floc": fold_all[idx].copy(),
                "chart": load_chart(bundle["ndc"], int(sid)),
                "seed": cfg.seed,
                "n_haar": cfg.n_haar_eff(),
                "skip_probes": cfg.skip_probes,
            }
        )
    rows = []
    n_workers = 1 if cfg.smoke else min(int(cfg.n_workers), max(1, (os.cpu_count() or 8) // 2))
    print(f"[ccpd] ViT-B pass {len(payloads)} anchors, {n_workers} workers", flush=True)
    if n_workers == 1:
        for i, p in enumerate(payloads):
            rows.append(process_anchor(p))
            if (i + 1) % 8 == 0 or i == 0:
                print(f"[ccpd] vitb {i+1}/{len(payloads)} sid={p['sid']}", flush=True)
                write_df(out / "tables" / "vitb_component_partial.parquet", pd.DataFrame(rows), force=True)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_anchor, p): p["sid"] for p in payloads}
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % 8 == 0 or done == 1:
                    print(f"[ccpd] vitb completed {done}/{len(payloads)}", flush=True)
                    write_df(out / "tables" / "vitb_component_partial.parquet", pd.DataFrame(rows), force=True)
    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    write_df(out / "tables" / "vitb_component_pass.parquet", df, force=cfg.force)
    return df
