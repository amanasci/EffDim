"""Qualitative synthetic checks that the (R, m) design can separate mechanisms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from .hashing import select_m, split_ab


def _cloud(n: int, rng: np.random.Generator, *, hetero: bool) -> tuple[np.ndarray, np.ndarray]:
    """2-D chart in R^3. Probe correlates with local |H| energy."""
    u = rng.normal(size=(n, 2))
    r = np.linalg.norm(u, axis=1, keepdims=True) + 1e-8
    if hetero:
        kappa = np.cos(3.0 * r[:, 0])  # sign/magnitude change with radius
    else:
        kappa = np.ones(n)
    z = 0.5 * kappa * (u[:, 0] ** 2 + u[:, 1] ** 2)
    X = np.column_stack([u, z]) + 0.02 * rng.normal(size=(n, 3))
    y = kappa + 0.15 * rng.normal(size=n)
    return X, y


def _kh_stat(X: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """Crude 2-D quadratic curvature energy from split halves (design check only)."""
    def hess(ii):
        U = X[ii, :2]
        z = X[ii, 2]
        # z ≈ 0.5 k (u^2+v^2) → fit k from ridge of [u^2, v^2, uv]
        Phi = np.column_stack([U[:, 0] ** 2, U[:, 1] ** 2, U[:, 0] * U[:, 1]])
        G = Phi.T @ Phi + 1e-3 * np.eye(3)
        b = np.linalg.solve(G, Phi.T @ z)
        H = np.array([b[0], b[1]])  # diagonal mean-curvature proxy
        return H

    return float(np.dot(hess(idx_a), hess(idx_b)))


def run_synthetic(*, seed: int = 0, n_rep: int = 8, n_anchor: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cells = [(2048, 512), (2048, 1024), (2048, 2048), (1024, 1024)]
    for hetero in (False, True):
        for R, m in cells:
            rhos = []
            for a in range(n_anchor):
                X, y = _cloud(int(R), rng, hetero=hetero)
                khs = []
                for r in range(n_rep):
                    pool = np.arange(R)
                    ch = select_m(pool, m, seed=seed + r, sample_id=a)
                    A, B = split_ab(np.arange(len(ch)), seed=seed + r, sample_id=a)
                    khs.append(_kh_stat(X[ch], A, B))
                rows.append(
                    {
                        "family": "heterogeneous" if hetero else "constant",
                        "R": R,
                        "m": m,
                        "sample_id": a,
                        "K_H_cross": float(np.median(khs)),
                        "y": float(np.mean(y)),
                    }
                )
    df = pd.DataFrame(rows)
    out = []
    for (fam, R, m), g in df.groupby(["family", "R", "m"]):
        x, yv = g.K_H_cross.to_numpy(float), g.y.to_numpy(float)
        msk = np.isfinite(x) & np.isfinite(yv)
        rho = float(spearmanr(x[msk], yv[msk]).statistic) if msk.sum() >= 5 else float("nan")
        rec = {"raw": rho, "controlled": rho, "n": int(msk.sum()), "family": fam, "R": int(R), "m": int(m)}
        out.append(rec)
    return pd.DataFrame(out)


run_synthetic = run_synthetic
