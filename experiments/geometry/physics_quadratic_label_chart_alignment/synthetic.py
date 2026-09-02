"""Deterministic synthetic validation of L/UQ/BS recovery (fast fixed-α path)."""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry.physics_local_probe_adaptation.ridge import ridge_fit_intercept, ridge_predict

from .config import (
    PRIMARY_D,
    SYNTH_ALIGN_MIN_DELTA,
    SYNTH_LIN_MAX_DELTA,
    SYNTH_ORTH_BS_MAX_FRAC,
    SYNTH_SHUFFLE_MAX_ABS_RHO,
)
from .features import gamma_from_Gamma, phi2_frob
from .models import _bs_basis, _design_constrained, _design_L, _design_UQ, _ridge_block, mse


def _oof_fixed(U, y, fold, kind, BS=None, al=10.0, aq=100.0):
    """Fast OOF with fixed penalties (synthetics only)."""
    yhat = np.full(len(y), np.nan)
    n_lin = U.shape[1]
    basis = None
    if kind == "BS" and BS is not None:
        basis, _, _ = _bs_basis(BS)
    for f in sorted(set(fold.tolist())):
        te = fold == f
        tr = ~te
        if tr.sum() < 16 or te.sum() < 4:
            continue
        if kind == "L":
            Xtr, Xte = _design_L(U[tr]), _design_L(U[te])
            aq_ = 1.0
        elif kind == "UQ":
            Xtr, Xte = _design_UQ(U[tr]), _design_UQ(U[te])
            aq_ = aq
        else:
            Xtr = _design_constrained(U[tr], BS, basis)
            Xte = _design_constrained(U[te], BS, basis)
            aq_ = aq
        w, b, info = _ridge_block(Xtr, y[tr], n_lin=n_lin, alpha_lin=al, alpha_quad=aq_)
        if info.get("ok"):
            yhat[te] = ridge_predict(Xte, w, b)
    return yhat


def _make_chart(rng: np.random.Generator, n: int = 400, d: int = PRIMARY_D, D: int = 64):
    U = rng.normal(size=(n, d))
    q = d * (d + 1) // 2
    BS = rng.normal(size=(D, q)) * 0.1
    fold = np.tile(np.arange(5), n // 5 + 1)[:n]
    return U, BS, fold


def run_synthetics(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    U, BS, fold = _make_chart(rng)
    n, d = U.shape
    a1 = rng.normal(size=d)

    y_lin = U @ a1 + rng.normal(scale=0.05, size=n)
    d_lin = mse(y_lin, _oof_fixed(U, y_lin, fold, "L")) - mse(y_lin, _oof_fixed(U, y_lin, fold, "UQ"))

    c = rng.normal(size=BS.shape[0])
    gamma = BS.T @ c
    y_al = U @ a1 + phi2_frob(U) @ gamma + rng.normal(scale=0.05, size=n)
    yL = _oof_fixed(U, y_al, fold, "L")
    yUQ = _oof_fixed(U, y_al, fold, "UQ")
    yBS = _oof_fixed(U, y_al, fold, "BS", BS=BS)
    d_uq_al = mse(y_al, yL) - mse(y_al, yUQ)
    d_bs_al = mse(y_al, yL) - mse(y_al, yBS)

    G = rng.normal(size=(d, d))
    G = 0.5 * (G + G.T)
    g = gamma_from_Gamma(G)
    _, _, Vt = np.linalg.svd(BS, full_matrices=False)
    r = min(Vt.shape[0], 32)
    P = Vt[:r].T @ Vt[:r]
    g_orth = g - P @ g
    y_or = U @ a1 + phi2_frob(U) @ g_orth + rng.normal(scale=0.05, size=n)
    yL = _oof_fixed(U, y_or, fold, "L")
    yUQ = _oof_fixed(U, y_or, fold, "UQ")
    yBS = _oof_fixed(U, y_or, fold, "BS", BS=BS)
    d_uq_or = mse(y_or, yL) - mse(y_or, yUQ)
    d_bs_or = mse(y_or, yL) - mse(y_or, yBS)

    d_shs = []
    for _ in range(3):
        y_sh = rng.permutation(y_al)
        d_shs.append(
            mse(y_sh, _oof_fixed(U, y_sh, fold, "L")) - mse(y_sh, _oof_fixed(U, y_sh, fold, "UQ"))
        )
    d_sh = float(np.median(d_shs))

    gates = {
        "linear_uq_near_zero": abs(d_lin) <= SYNTH_LIN_MAX_DELTA,
        "aligned_uq_positive": d_uq_al >= SYNTH_ALIGN_MIN_DELTA,
        "aligned_bs_positive": d_bs_al >= SYNTH_ALIGN_MIN_DELTA * 0.5,
        "orth_bs_weak": (d_bs_or <= SYNTH_ORTH_BS_MAX_FRAC * max(d_uq_or, 1e-8)) if d_uq_or > 0 else True,
        "shuffle_small": abs(d_sh) <= max(0.1, 0.05 * max(abs(d_uq_al), 1e-8)),
    }
    return {
        "ok": all(gates.values()),
        "gates": gates,
        "deltas": {
            "linear_dQ": d_lin,
            "aligned_dUQ": d_uq_al,
            "aligned_dBS": d_bs_al,
            "orth_dUQ": d_uq_or,
            "orth_dBS": d_bs_or,
            "shuffle_dQ": d_sh,
        },
        "thresholds": {
            "SYNTH_LIN_MAX_DELTA": SYNTH_LIN_MAX_DELTA,
            "SYNTH_ALIGN_MIN_DELTA": SYNTH_ALIGN_MIN_DELTA,
            "SYNTH_ORTH_BS_MAX_FRAC": SYNTH_ORTH_BS_MAX_FRAC,
            "SYNTH_SHUFFLE_MAX_ABS_RHO": SYNTH_SHUFFLE_MAX_ABS_RHO,
        },
    }
