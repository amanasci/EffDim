"""Synthetic curved manifold where ambient predictive direction rotates."""

from __future__ import annotations

import numpy as np

from .ridge import ridge_fit_intercept, ridge_predict


def run_synthetic(*, seed: int = 0, n_patch: int = 400, n_anchor: int = 24) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_anchor):
        # local angle θ rotates with "curvature proxy" κ
        kappa = float(rng.uniform(0.2, 2.0))
        theta = 0.5 * kappa
        # 2D tangent coords
        U = rng.normal(size=(n_patch, 2))
        # ambient embedding: rotate predictive axis into 3D
        c, s = np.cos(theta), np.sin(theta)
        # true signal along rotated direction in ambient
        w_true = np.array([c, s, 0.0])
        X = np.column_stack([U, 0.15 * kappa * (U[:, 0] ** 2 + U[:, 1] ** 2)])
        X = X + 0.05 * rng.normal(size=X.shape)
        y = X @ w_true + 0.1 * rng.normal(size=n_patch)
        # global probe: fit on all but with fixed wrong direction (ignore rotation)
        # simulate global as ridge on full data then evaluate — but for OOF use folds
        fold = np.tile(np.arange(5), n_patch // 5 + 1)[:n_patch]
        yhat_g = np.full(n_patch, np.nan)
        yhat_p = np.full(n_patch, np.nan)
        yhat_in = np.full(n_patch, np.nan)
        for f in range(5):
            te = fold == f
            tr = ~te
            # global-like: fit on train but features only first coord (wrong)
            wI, bI, _ = ridge_fit_intercept(X[tr, :1], y[tr], alpha=1.0)
            yhat_g[te] = ridge_predict(X[te, :1], wI, bI)
            wP, bP, _ = ridge_fit_intercept(X[tr], y[tr], alpha=1.0)
            yhat_p[te] = ridge_predict(X[te], wP, bP)
            wA, bA, _ = ridge_fit_intercept(X, y, alpha=1.0)
            yhat_in[te] = ridge_predict(X[te], wA, bA)
        mse_g = float(np.mean((y - yhat_g) ** 2))
        mse_p = float(np.mean((y - yhat_p) ** 2))
        mse_in = float(np.mean((y - yhat_in) ** 2))
        rows.append(
            {
                "kappa": kappa,
                "dMSE_GP": mse_g - mse_p,
                "mse_g": mse_g,
                "mse_p": mse_p,
                "mse_insample": mse_in,
                "insample_optimistic": mse_in < mse_p,
            }
        )
    tab = rows
    kappa = np.array([r["kappa"] for r in tab])
    dMSE = np.array([r["dMSE_GP"] for r in tab])
    mse_g = np.array([r["mse_g"] for r in tab])
    # shuffle destroys advantage association
    y_shuf = rng.permutation(dMSE)
    from scipy.stats import spearmanr

    rho = float(spearmanr(kappa, dMSE).statistic)
    rho_shuf = float(spearmanr(kappa, y_shuf).statistic)
    rho_g = float(spearmanr(kappa, mse_g).statistic)
    return {
        "ok": bool(rho > 0.2 and rho_g > 0.1 and abs(rho_shuf) < abs(rho)),
        "rho_kappa_dMSE": rho,
        "rho_kappa_mse_G": rho_g,
        "rho_kappa_dMSE_shuffled": rho_shuf,
        "mean_insample_optimistic": float(np.mean([r["insample_optimistic"] for r in tab])),
        "n": int(n_anchor),
    }
