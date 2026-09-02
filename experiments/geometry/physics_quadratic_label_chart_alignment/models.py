"""Nested-tuned tangent linear / quadratic label models (L, UQ, BS, FQ) — fast path."""

from __future__ import annotations

import numpy as np

from geometry.physics_local_probe_adaptation.ridge import ridge_fit_intercept, ridge_predict

from .config import LIN_GRID, MIN_TEST, MIN_TRAIN, PRIMARY_D, PROBE_ALPHA, QUAD_GRID
from .features import phi2_frob


def _scalar_rms(U: np.ndarray) -> float:
    return float(np.sqrt(np.mean(U * U))) if U.size else 1.0


def _design_L(U: np.ndarray) -> np.ndarray:
    return np.asarray(U, dtype=np.float64)


def _design_UQ(U: np.ndarray) -> np.ndarray:
    return np.concatenate([U, phi2_frob(U)], axis=1)


def _bs_basis(BS_frob: np.ndarray, *, n_comp: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """Geometry-only left singular basis for BS/Q flat tensors."""
    U_svd, S, _ = np.linalg.svd(BS_frob, full_matrices=False)
    energy = np.cumsum(S * S) / max(float(np.sum(S * S)), 1e-18)
    r = int(np.searchsorted(energy, 0.99) + 1) if n_comp is None else int(n_comp)
    r = max(1, min(r, U_svd.shape[1], 48))
    return U_svd[:, :r], r, S[:r]


def _design_constrained(U: np.ndarray, BS_frob: np.ndarray, U_basis: np.ndarray) -> np.ndarray:
    Phi = phi2_frob(U)
    scores = (Phi @ BS_frob.T) @ U_basis
    return np.concatenate([U, scores], axis=1)


def _ridge_from_gram(
    XtX0: np.ndarray,
    Xty: np.ndarray,
    x_mean: np.ndarray,
    y_mean: float,
    *,
    n_lin: int,
    alpha_lin: float,
    alpha_quad: float,
) -> tuple[np.ndarray, float, bool]:
    p = XtX0.shape[0]
    XtX = XtX0.copy()
    pen = np.empty(p, dtype=np.float64)
    pen[:n_lin] = float(alpha_lin)
    pen[n_lin:] = float(alpha_quad)
    XtX.flat[:: p + 1] += pen
    try:
        w = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.zeros(p), y_mean, False
    b = y_mean - float(x_mean @ w)
    return w, float(b), True


def _nested_select(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    fold_tr: np.ndarray,
    *,
    n_lin: int,
    lin_grid=LIN_GRID,
    quad_grid=QUAD_GRID,
) -> tuple[float, float, dict]:
    """Gram-cached nested CV for block ridge (α_lin, α_quad)."""
    uniq = sorted(set(fold_tr.tolist()))
    if len(uniq) < 2 or len(ytr) < 16:
        return 100.0, 1000.0, {"ok": False}

    caches = []
    for f in uniq:
        te = fold_tr == f
        tr = ~te
        if int(tr.sum()) < 8 or int(te.sum()) < 4:
            continue
        Xa, ya = Xtr[tr], ytr[tr]
        m = np.isfinite(ya) & np.all(np.isfinite(Xa), axis=1)
        Xa, ya = Xa[m], ya[m]
        if len(ya) < 4:
            continue
        x_mean = Xa.mean(0)
        y_mean = float(ya.mean())
        Xc = Xa - x_mean
        yc = ya - y_mean
        XtX0 = Xc.T @ Xc
        Xty = Xc.T @ yc
        Xte = Xtr[te]
        yte = ytr[te]
        caches.append((XtX0, Xty, x_mean, y_mean, Xte, yte))
    if not caches:
        return 100.0, 1000.0, {"ok": False}

    best = (float("inf"), 100.0, 1000.0)
    for al in lin_grid:
        for aq in quad_grid:
            mses = []
            for XtX0, Xty, x_mean, y_mean, Xte, yte in caches:
                w, b, ok = _ridge_from_gram(
                    XtX0, Xty, x_mean, y_mean, n_lin=n_lin, alpha_lin=al, alpha_quad=aq
                )
                if not ok:
                    continue
                pred = Xte @ w + b
                m = np.isfinite(yte) & np.isfinite(pred)
                if int(m.sum()) < 2:
                    continue
                err = yte[m] - pred[m]
                mses.append(float(np.mean(err * err)))
            if not mses:
                continue
            score = float(np.mean(mses))
            if score < best[0] - 1e-15 or (
                abs(score - best[0]) <= 1e-15 and (aq > best[2] or (aq == best[2] and al > best[1]))
            ):
                best = (score, float(al), float(aq))
    return best[1], best[2], {"ok": True, "cv_mse": best[0]}


def _ridge_block(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_lin: int,
    alpha_lin: float,
    alpha_quad: float,
    compute_edf: bool = False,
) -> tuple[np.ndarray, float, dict]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    n, p = X.shape
    if n < 4:
        return np.zeros(p), float("nan"), {"ok": False}
    x_mean = X.mean(0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    XtX0 = Xc.T @ Xc
    w, b, ok = _ridge_from_gram(
        XtX0, Xc.T @ yc, x_mean, y_mean, n_lin=n_lin, alpha_lin=alpha_lin, alpha_quad=alpha_quad
    )
    if not ok:
        return np.zeros(p), y_mean, {"ok": False}
    edf = float("nan")
    if compute_edf:
        pen = np.empty(p, dtype=np.float64)
        pen[:n_lin] = float(alpha_lin)
        pen[n_lin:] = float(alpha_quad)
        XtX = XtX0.copy()
        XtX.flat[:: p + 1] += pen
        try:
            edf = float(np.trace(np.linalg.solve(XtX, XtX0)))
        except np.linalg.LinAlgError:
            edf = float("nan")
    return w, float(b), {"ok": True, "edf": edf}


def oof_predict_model(
    U: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    *,
    kind: str,
    BS_frob: np.ndarray | None = None,
    Q_frob: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Strict same-fold OOF predictions for L / UQ / BS / FQ with nested tuning."""
    n = len(y)
    yhat = np.full(n, np.nan)
    diags = []
    n_lin = PRIMARY_D

    basis = None
    r_comp = None
    if kind == "BS":
        assert BS_frob is not None
        basis, r_comp, _ = _bs_basis(BS_frob)
    elif kind == "FQ":
        assert Q_frob is not None
        basis, r_comp, _ = _bs_basis(Q_frob)

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
            Xtr = _design_L(Utr)
            Xte = _design_L(Ute)
            al, aq, info = _nested_select(Xtr, y[tr], fold[tr], n_lin=n_lin, quad_grid=(1.0,))
            aq = 1.0
        elif kind == "UQ":
            Xtr = _design_UQ(Utr)
            Xte = _design_UQ(Ute)
            al, aq, info = _nested_select(Xtr, y[tr], fold[tr], n_lin=n_lin)
        elif kind == "BS":
            BSs = BS_frob * scale_t
            Xtr = _design_constrained(Utr, BSs, basis)
            Xte = _design_constrained(Ute, BSs, basis)
            al, aq, info = _nested_select(Xtr, y[tr], fold[tr], n_lin=n_lin)
            info = {**info, "n_comp": r_comp}
        elif kind == "FQ":
            Qs = Q_frob * scale_t
            Xtr = _design_constrained(Utr, Qs, basis)
            Xte = _design_constrained(Ute, Qs, basis)
            al, aq, info = _nested_select(Xtr, y[tr], fold[tr], n_lin=n_lin)
            info = {**info, "n_comp": r_comp}
        else:
            raise ValueError(kind)

        w, b, fit = _ridge_block(Xtr, y[tr], n_lin=n_lin, alpha_lin=al, alpha_quad=aq, compute_edf=False)
        if not fit.get("ok", False):
            continue
        yhat[te] = ridge_predict(Xte, w, b)
        diags.append({"fold": int(f), "alpha_lin": al, "alpha_quad": aq, **info, **fit, "scale_s": s})

    return yhat, {"folds": diags, "kind": kind}


def oof_ambient_P(X: np.ndarray, y: np.ndarray, fold: np.ndarray, *, alpha: float = PROBE_ALPHA) -> np.ndarray:
    yhat = np.full(len(y), np.nan)
    for f in sorted(set(fold.tolist())):
        te = fold == f
        tr = ~te
        if tr.sum() < MIN_TRAIN or te.sum() < MIN_TEST:
            continue
        w, b, info = ridge_fit_intercept(X[tr], y[tr], alpha=alpha)
        if info["ok"]:
            yhat[te] = ridge_predict(X[te], w, b)
    return yhat


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < MIN_TEST:
        return float("nan")
    return float(np.mean((y[m] - yhat[m]) ** 2))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < MIN_TEST:
        return float("nan")
    return float(np.mean(np.abs(y[m] - yhat[m])))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < MIN_TEST:
        return float("nan")
    yt, yp = y[m], yhat[m]
    sst = float(np.sum((yt - yt.mean()) ** 2))
    if sst < 1e-18:
        return float("nan")
    return 1.0 - float(np.sum((yt - yp) ** 2)) / sst


from geometry.physics_activation_atlas.quadratic import quadratic_features

def _design_BS(U: np.ndarray, BS_frob: np.ndarray, *, n_comp: int | None = None) -> tuple:
    """Back-compat wrapper used by older tests."""
    basis, r, S = _bs_basis(BS_frob, n_comp=n_comp)
    return _design_constrained(U, BS_frob, basis), r, S


def fit_A_flat_fast(Xloc: np.ndarray, x0: np.ndarray, J: np.ndarray, *, alpha: float = 0.1) -> np.ndarray | None:
    """Cheap tangential warp A only (reuse frozen BS from NDC for FQ)."""
    from geometry.physics_activation_atlas.sphere_normal_quadratic import sphere_project_basis

    x0 = x0 / max(float(np.linalg.norm(x0)), 1e-12)
    J = sphere_project_basis(x0, J)
    U = (Xloc - x0) @ J
    Phi = quadratic_features(U)
    L = x0[None, :] + U @ J.T
    scale = np.linalg.norm(L, axis=1, keepdims=True)
    target_un = Xloc * np.maximum(scale, 1e-8)
    tang_res = (target_un - L) @ J
    q = Phi.shape[1]
    d = J.shape[1]
    G = Phi.T @ Phi
    G.flat[:: q + 1] += float(alpha)
    try:
        A = np.linalg.solve(G, Phi.T @ tang_res).T
    except np.linalg.LinAlgError:
        return None
    if A.shape != (d, q):
        return None
    return A
