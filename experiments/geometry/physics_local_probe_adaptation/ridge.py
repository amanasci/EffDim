"""Ridge with intercept matching sklearn sum-of-squares (α‖w‖² + ‖y−Xw−b‖²)."""

from __future__ import annotations

import numpy as np


def ridge_fit_intercept(X: np.ndarray, y: np.ndarray, *, alpha: float) -> tuple[np.ndarray, float, dict]:
    """Fit Ridge(fit_intercept=True) with sum-of-squares loss. Returns w, b, diag."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    n, f = X.shape
    if n < 4:
        return np.zeros(f), float("nan"), {"ok": False, "n": int(n), "cond": float("nan"), "edf": float("nan")}
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean
    XtX = Xc.T @ Xc
    np.fill_diagonal(XtX, np.diag(XtX) + float(alpha))
    try:
        L = np.linalg.cholesky(XtX)
    except np.linalg.LinAlgError:
        return np.zeros(f), y_mean, {"ok": False, "n": int(n), "cond": float("nan"), "edf": float("nan")}
    w = np.linalg.solve(L.T, np.linalg.solve(L, Xc.T @ yc))
    b = y_mean - float(x_mean @ w)
    # effective df ≈ tr(H); H = Xc (XtX)^{-1} Xc'
    # edf = tr(Xc inv XtX Xc') = tr(inv XtX Xc' Xc) = tr(inv XtX (XtX - αI))
    XtX0 = Xc.T @ Xc
    try:
        inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(f)))
        edf = float(np.trace(inv @ XtX0))
        cond = float(np.linalg.cond(XtX))
    except Exception:
        edf, cond = float("nan"), float("nan")
    return w.astype(np.float64), float(b), {"ok": True, "n": int(n), "cond": cond, "edf": edf, "L": L, "x_mean": x_mean, "Xc": Xc}


def ridge_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ w + float(b)


def ridge_solve_many(L: np.ndarray, Xc: np.ndarray, x_mean: np.ndarray, Yc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched RHS: Yc shape (n, B). Returns W (f,B), intercepts via means handled outside."""
    # W = solve(XtX, Xc' Yc)
    rhs = Xc.T @ Yc
    W = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
    return W


def nested_alpha_1se(
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    grid: tuple[float, ...],
    *,
    seed: int,
) -> tuple[float, dict]:
    """Inner CV on outer-train rows only; one-standard-error rule on MSE."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    fold = np.asarray(fold, dtype=int)
    m = np.isfinite(y)
    X, y, fold = X[m], y[m], fold[m]
    uniq = sorted(set(fold.tolist()))
    if len(uniq) < 2 or len(y) < 16:
        return float(grid[len(grid) // 2]), {"ok": False, "selected": float("nan")}
    rng = np.random.default_rng(seed)
    # use existing global folds within the train patch as inner folds
    scores = {a: [] for a in grid}
    for f in uniq:
        te = fold == f
        tr = ~te
        if tr.sum() < 8 or te.sum() < 4:
            continue
        for a in grid:
            w, b, info = ridge_fit_intercept(X[tr], y[tr], alpha=a)
            if not info["ok"]:
                continue
            pred = ridge_predict(X[te], w, b)
            mse = float(np.mean((y[te] - pred) ** 2))
            if np.isfinite(mse):
                scores[a].append(mse)
    rows = []
    for a in grid:
        arr = np.asarray(scores[a], float)
        if len(arr) == 0:
            rows.append({"alpha": a, "mean": np.nan, "se": np.nan, "n": 0})
            continue
        rows.append({"alpha": a, "mean": float(arr.mean()), "se": float(arr.std(ddof=1) / max(np.sqrt(len(arr)), 1.0)), "n": int(len(arr))})
    finite = [r for r in rows if np.isfinite(r["mean"])]
    if not finite:
        return float(grid[len(grid) // 2]), {"ok": False, "grid": rows}
    best = min(finite, key=lambda r: r["mean"])
    thr = best["mean"] + best["se"]
    # most regularized (largest α) with mean ≤ thr
    cand = [r for r in finite if r["mean"] <= thr]
    chosen = max(cand, key=lambda r: r["alpha"]) if cand else best
    return float(chosen["alpha"]), {"ok": True, "grid": rows, "selected": float(chosen["alpha"]), "best": best}
