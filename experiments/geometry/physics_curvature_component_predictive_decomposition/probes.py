"""IQ / TQ / UQ-v2 and BSH / BSTF nested probes. Does not overwrite frozen QLCA UQ."""

from __future__ import annotations

import numpy as np

from geometry.physics_local_probe_adaptation.ridge import ridge_predict
from geometry.physics_quadratic_label_chart_alignment.config import LIN_GRID, MIN_TEST, MIN_TRAIN, PRIMARY_D, QUAD_GRID
from geometry.physics_quadratic_label_chart_alignment.features import Gamma_from_gamma, n_quad, phi2_frob
from geometry.physics_quadratic_label_chart_alignment.models import (
    _bs_basis,
    _design_L,
    _design_constrained,
    _nested_select,
    _ridge_block,
    _scalar_rms,
    mse,
)

from .decompose import iso_unit_gamma

OMIT_QUAD = float("inf")
QUAD_GRID_V2 = tuple(QUAD_GRID) + (OMIT_QUAD,)


def iso_basis(d: int) -> np.ndarray:
    e = iso_unit_gamma(d)
    return e / max(float(np.linalg.norm(e)), 1e-18)


def tf_basis(d: int) -> np.ndarray:
    e = iso_basis(d)
    q = n_quad(d)
    P = np.eye(q) - np.outer(e, e)
    u, s, _ = np.linalg.svd(P, full_matrices=False)
    return u[:, s > 0.5]


def verify_quadratic_span(d: int = PRIMARY_D) -> dict[str, float]:
    e = iso_basis(d)
    tf = tf_basis(d)
    Q = np.column_stack([e, tf])
    return {
        "n_iso": 1,
        "n_tf": int(tf.shape[1]),
        "n_quad": n_quad(d),
        "orth_err": float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))),
        "span_err": float(np.linalg.norm(Q @ Q.T - np.eye(n_quad(d)))),
    }


def _design_IQ(U: np.ndarray) -> np.ndarray:
    Phi = phi2_frob(U)
    return np.concatenate([U, (Phi @ iso_basis(U.shape[1]))[:, None]], axis=1)


def _design_TQ(U: np.ndarray) -> np.ndarray:
    Phi = phi2_frob(U)
    return np.concatenate([U, Phi @ tf_basis(U.shape[1])], axis=1)


def _design_UQ(U: np.ndarray) -> np.ndarray:
    return np.concatenate([U, phi2_frob(U)], axis=1)


def _nested_select_v2(Xtr, ytr, fold_tr, *, n_lin: int, lin_grid=LIN_GRID, quad_grid=QUAD_GRID_V2):
    """Nested block ridge including an explicit omit-quadratic candidate."""
    finite = tuple(q for q in quad_grid if np.isfinite(q))
    al, aq, info = _nested_select(Xtr, ytr, fold_tr, n_lin=n_lin, lin_grid=lin_grid, quad_grid=finite)
    best = (float(info.get("cv_mse", np.inf)) if info.get("ok") else float("inf"), float(al), float(aq))
    Xlin = Xtr[:, :n_lin]
    al0, _, info0 = _nested_select(Xlin, ytr, fold_tr, n_lin=n_lin, lin_grid=lin_grid, quad_grid=(1.0,))
    score0 = float(info0.get("cv_mse", np.inf)) if info0.get("ok") else float("inf")
    if score0 < best[0] - 1e-15 or (abs(score0 - best[0]) <= 1e-15):
        best = (score0, float(al0), OMIT_QUAD)
        info = {**info0, "omit_quad": True}
    else:
        info = {**info, "omit_quad": False}
    return best[1], best[2], info


def _energy_rank(S: np.ndarray, frac: float, cap: int | None) -> int:
    if S.size == 0:
        return 1
    energy = np.cumsum(S * S) / max(float(np.sum(S * S)), 1e-18)
    r = int(np.searchsorted(energy, frac) + 1)
    r = max(1, min(r, S.size))
    if cap is not None:
        r = min(r, cap)
    return r


def mode_trace_content(basis: np.ndarray, d: int) -> np.ndarray:
    """C_trace of each retained singular vector in Frobenius Hessian space."""
    out = np.empty(basis.shape[1], dtype=np.float64)
    for i in range(basis.shape[1]):
        G = Gamma_from_gamma(basis[:, i], d)
        num = float(np.trace(G) ** 2)
        den = float(d * np.sum(G * G))
        out[i] = num / den if den > 1e-18 else float("nan")
    return out


def oof_component_model(
    U: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    *,
    kind: str,
    BS_frob: np.ndarray | None = None,
    n_comp: int | None = None,
    energy_frac: float | None = None,
    cap: int | None = 48,
    omit_quad: bool = False,
) -> tuple[np.ndarray, dict]:
    n = len(y)
    yhat = np.full(n, np.nan)
    diags = []
    n_lin = PRIMARY_D
    d = U.shape[1]
    basis = None
    r_comp = None
    S = None
    trace_c = None
    if kind in ("BSH", "BSTF", "BS"):
        assert BS_frob is not None
        U_svd, S, Vh = np.linalg.svd(BS_frob, full_matrices=False)
        if n_comp is not None:
            r_comp = max(1, min(int(n_comp), U_svd.shape[1]))
        elif energy_frac is not None:
            r_comp = _energy_rank(S, energy_frac, cap)
        else:
            r_comp = _energy_rank(S, 0.99, cap)
        basis = U_svd[:, :r_comp]
        trace_c = mode_trace_content(Vh[:r_comp].T, d)

    for f in sorted(set(fold.tolist())):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            continue
        s = max(_scalar_rms(U[tr]), 1e-8)
        Utr = U[tr] / s
        Ute = U[te] / s
        scale_t = s * s
        if kind == "L":
            Xtr, Xte = _design_L(Utr), _design_L(Ute)
            al, aq, info = _nested_select(Xtr, y[tr], fold[tr], n_lin=n_lin, quad_grid=(1.0,))
            aq = 1.0
        elif kind == "IQ":
            Xtr, Xte = _design_IQ(Utr), _design_IQ(Ute)
            sel = _nested_select_v2 if omit_quad else _nested_select
            al, aq, info = sel(Xtr, y[tr], fold[tr], n_lin=n_lin)
        elif kind == "TQ":
            Xtr, Xte = _design_TQ(Utr), _design_TQ(Ute)
            sel = _nested_select_v2 if omit_quad else _nested_select
            al, aq, info = sel(Xtr, y[tr], fold[tr], n_lin=n_lin)
        elif kind == "UQ2":
            Xtr, Xte = _design_UQ(Utr), _design_UQ(Ute)
            al, aq, info = _nested_select_v2(Xtr, y[tr], fold[tr], n_lin=n_lin)
        elif kind in ("BSH", "BSTF", "BS"):
            BSs = BS_frob * scale_t
            Xtr = _design_constrained(Utr, BSs, basis)
            Xte = _design_constrained(Ute, BSs, basis)
            al, aq, info = _nested_select(Xtr, y[tr], fold[tr], n_lin=n_lin)
            info = {**info, "n_comp": r_comp}
        else:
            raise ValueError(kind)
        if kind != "L" and np.isinf(aq):
            Xtr, Xte = Xtr[:, :n_lin], Xte[:, :n_lin]
            w, b, fit = _ridge_block(Xtr, y[tr], n_lin=n_lin, alpha_lin=al, alpha_quad=1.0, compute_edf=True)
        else:
            w, b, fit = _ridge_block(Xtr, y[tr], n_lin=n_lin, alpha_lin=al, alpha_quad=aq, compute_edf=True)
        if not fit.get("ok", False):
            continue
        yhat[te] = ridge_predict(Xte, w, b)
        diags.append({"fold": int(f), "alpha_lin": al, "alpha_quad": aq, **info, **fit, "scale_s": s})

    rec = {"folds": diags, "kind": kind, "n_comp": r_comp}
    if S is not None and r_comp:
        energy = np.cumsum(S * S) / max(float(np.sum(S * S)), 1e-18)
        rec["energy_captured"] = float(energy[r_comp - 1])
        rec["algebraic_rank"] = int(np.sum(S > 1e-8 * S[0])) if S.size else 0
        rec["median_mode_C_trace"] = float(np.nanmedian(trace_c)) if trace_c is not None else float("nan")
        rec["mean_mode_C_trace"] = float(np.nanmean(trace_c)) if trace_c is not None else float("nan")
        rec["n_modes"] = int(r_comp)
    rec["n_quad_block"] = {"IQ": 1, "TQ": n_quad(d) - 1, "UQ2": n_quad(d), "L": 0}.get(kind)
    rec["median_edf"] = float(np.nanmedian([f.get("edf", np.nan) for f in diags])) if diags else float("nan")
    rec["median_alpha_lin"] = float(np.nanmedian([f.get("alpha_lin", np.nan) for f in diags])) if diags else float("nan")
    rec["median_alpha_quad"] = float(
        np.nanmedian([f["alpha_quad"] for f in diags if np.isfinite(f.get("alpha_quad", np.nan))])
    ) if diags else float("nan")
    rec["frac_omit_quad"] = float(np.mean([np.isinf(f.get("alpha_quad", 0)) for f in diags])) if diags else float("nan")
    rec["mse"] = mse(y, yhat)
    return yhat, rec
