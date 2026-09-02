"""Geometry-only truncated BS refits (labels never choose the rank)."""

from __future__ import annotations

import numpy as np

from geometry.physics_local_probe_adaptation.ridge import ridge_predict
from geometry.physics_quadratic_label_chart_alignment.config import LIN_GRID, MIN_TEST, MIN_TRAIN, PRIMARY_D, QUAD_GRID
from geometry.physics_quadratic_label_chart_alignment.features import phi2_frob
from geometry.physics_quadratic_label_chart_alignment.models import (
    _nested_select,
    _ridge_block,
    _scalar_rms,
    mse,
)

from .config import ENERGY_FRACS, ORIGINAL_N_COMP_CAP
from .rank import energy_rank, original_retained_rank, reachable_fraction


def truncation_ranks(S: np.ndarray) -> dict[str, int]:
    out = {
        "e90": energy_rank(S, ENERGY_FRACS[0]),
        "e95": energy_rank(S, ENERGY_FRACS[1]),
        "e99": energy_rank(S, ENERGY_FRACS[2]),
        "original_rule": original_retained_rank(S),
    }
    return {k: int(max(1, min(v, S.size, S.size))) for k, v in out.items()}


def oof_bs_truncated(
    U: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    BS_frob: np.ndarray,
    n_comp: int,
) -> tuple[np.ndarray, dict]:
    """Same outer folds / train-only RMS / nested block ridge as frozen BS, with fixed r."""
    n = len(y)
    yhat = np.full(n, np.nan)
    U_svd, S, _ = np.linalg.svd(np.asarray(BS_frob, dtype=np.float64), full_matrices=False)
    r = int(max(1, min(int(n_comp), U_svd.shape[1])))
    # Intentionally no extra cap of 48 except when the caller passes original_rule.
    basis = U_svd[:, :r]
    diags = []
    n_lin = PRIMARY_D
    for f in sorted(set(fold.tolist())):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            continue
        s = max(_scalar_rms(U[tr]), 1e-8)
        Utr, Ute = U[tr] / s, U[te] / s
        BSs = BS_frob * (s * s)
        Phi_tr = phi2_frob(Utr)
        Phi_te = phi2_frob(Ute)
        scores_tr = (Phi_tr @ BSs.T) @ basis
        scores_te = (Phi_te @ BSs.T) @ basis
        Xtr = np.concatenate([Utr, scores_tr], axis=1)
        Xte = np.concatenate([Ute, scores_te], axis=1)
        al, aq, info = _nested_select(Xtr, y[tr], fold[tr], n_lin=n_lin, lin_grid=LIN_GRID, quad_grid=QUAD_GRID)
        w, b, fit = _ridge_block(Xtr, y[tr], n_lin=n_lin, alpha_lin=al, alpha_quad=aq, compute_edf=True)
        if not fit.get("ok", False):
            continue
        yhat[te] = ridge_predict(Xte, w, b)
        diags.append({"fold": int(f), "alpha_lin": al, "alpha_quad": aq, "edf": fit.get("edf"), "n_comp": r, **info})
    edfs = [d.get("edf") for d in diags if d.get("edf") is not None]
    return yhat, {
        "n_comp": r,
        "median_edf": float(np.nanmedian(edfs)) if edfs else float("nan"),
        "median_aq": float(np.nanmedian([d["alpha_quad"] for d in diags])) if diags else float("nan"),
        "median_al": float(np.nanmedian([d["alpha_lin"] for d in diags])) if diags else float("nan"),
    }


def uq_contains_L(lin_grid=LIN_GRID, quad_grid=QUAD_GRID) -> bool:
    """UQ contains L iff every linear penalty has an α_Q=∞ counterpart."""
    has_inf = any(not np.isfinite(aq) or float(aq) == np.inf for aq in quad_grid)
    return bool(has_inf and set(lin_grid))
