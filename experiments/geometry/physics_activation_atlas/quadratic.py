"""Regularized local quadratic chart models on fixed PCA coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quadratic_features(U: np.ndarray) -> np.ndarray:
    """Symmetric quadratic features: [u_a u_b for a<=b], shape (N, d(d+1)/2)."""
    n, d = U.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(U[:, a] * U[:, b])
    return np.stack(cols, axis=1).astype(np.float64)


def n_quad_features(d: int) -> int:
    return d * (d + 1) // 2


@dataclass
class QuadraticChart:
    mu: np.ndarray  # (D,)
    basis: np.ndarray  # (D, d) absorbs coord_std like decoder
    B_flat: np.ndarray  # (D, n_quad) coeffs for 0.5 * Phi(u) in ambient
    ridge: float
    output_normalize: bool = True

    def decode(self, U: np.ndarray) -> np.ndarray:
        Phi = quadratic_features(U)
        # f = Normalize(mu + W u + 0.5 * B_flat @ phi) with phi already u_a u_b
        # B_flat stores coeffs such that residual = B_flat @ phi (= 1/2 sum B_ab u_a u_b)
        linear = self.mu[None, :] + U @ self.basis.T
        resid = Phi @ self.B_flat.T
        Y = linear + resid
        if self.output_normalize:
            n = np.linalg.norm(Y, axis=1, keepdims=True)
            Y = Y / np.maximum(n, 1e-8)
        return Y.astype(np.float32)

    def jacobian_at(self, u: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of unnormalized map, then project for normalized output."""
        u = np.asarray(u, dtype=np.float64)
        d = u.shape[0]
        # unnormalized y = mu + W u + B_flat @ phi(u)
        Phi_u = quadratic_features(u[None, :])[0]
        y = self.mu + self.basis @ u + self.B_flat @ Phi_u
        # dy/du_k = W[:,k] + d/du_k (B_flat @ phi)
        # phi_{ab}=u_a u_b (a<=b); d phi_ab / du_k = delta_ak u_b + delta_bk u_a (with a==b => 2 u_a)
        J = self.basis.copy().astype(np.float64)  # (D, d)
        idx = 0
        for a in range(d):
            for b in range(a, d):
                coeff = self.B_flat[:, idx]  # (D,)
                if a == b:
                    J[:, a] += coeff * (2.0 * u[a])
                else:
                    J[:, a] += coeff * u[b]
                    J[:, b] += coeff * u[a]
                idx += 1
        if not self.output_normalize:
            return J
        # Normalize(y): DN = (I - x x^T)/|y| with x=y/|y|
        nrm = float(np.linalg.norm(y))
        if nrm < 1e-12:
            return J
        x = y / nrm
        # d(x)/du = (I - x x^T) J / nrm
        return ((np.eye(len(y)) - np.outer(x, x)) @ J) / nrm

    def hessian_at(self, u: np.ndarray) -> np.ndarray:
        """Second derivatives of normalized decoder, shape (D, d, d). Uses product rule."""
        u = np.asarray(u, dtype=np.float64)
        d = u.shape[0]
        D = self.mu.shape[0]
        # unnormalized Hessian is constant in u for quadratic: H_unnorm[:,a,b] = B_sym
        Hun = np.zeros((D, d, d), dtype=np.float64)
        idx = 0
        for a in range(d):
            for b in range(a, d):
                coeff = self.B_flat[:, idx]
                if a == b:
                    Hun[:, a, a] += 2.0 * coeff
                else:
                    Hun[:, a, b] += coeff
                    Hun[:, b, a] += coeff
                idx += 1
        if not self.output_normalize:
            return Hun
        # For normalized map, use FD on analytic Jacobian (stable, cheap for small d)
        eps = 1e-3
        J0 = self.jacobian_at(u)
        H = np.zeros((D, d, d), dtype=np.float64)
        for j in range(d):
            uj = u.copy()
            uj[j] += eps
            Jj = self.jacobian_at(uj)
            H[:, :, j] = (Jj - J0) / eps
        return 0.5 * (H + H.transpose(0, 2, 1))


def fit_quadratic_chart(
    pca: dict,
    U: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    U_va: np.ndarray,
    X_va: np.ndarray,
    w_va: np.ndarray,
    *,
    ridges: list[float] | None = None,
    output_normalize: bool = True,
) -> tuple[QuadraticChart, dict]:
    """Weighted ridge on quadratic residual after linear PCA reconstruction."""
    ridges = ridges or [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    mask = w > 1e-6
    if mask.sum() < 10:
        # degenerate: zero residual
        d = U.shape[1]
        chart = QuadraticChart(
            mu=pca["mu"].astype(np.float64),
            basis=(pca["basis"] * pca["coord_std"]).astype(np.float64),
            B_flat=np.zeros((X.shape[1], n_quad_features(d))),
            ridge=1.0,
            output_normalize=output_normalize,
        )
        return chart, {"ridge": 1.0, "val_mse": float("nan"), "degenerate": True}

    Ut, Xt, wt = U[mask], X[mask], w[mask]
    wt = wt / wt.sum()
    basis = (pca["basis"] * pca["coord_std"]).astype(np.float64)
    mu = pca["mu"].astype(np.float64)
    # residual target: X - Normalize(mu + W u) or unnormalized target before normalize
    linear = mu[None, :] + Ut @ basis.T
    if output_normalize:
        # fit in ambient before normalize toward X * |linear| ≈ direction; use X - linear_proj
        # Practical: target residual so that linear + resid ≈ X (then normalize at decode)
        target = Xt - linear
    else:
        target = Xt - linear
    Phi = quadratic_features(Ut)
    # weighted ridge: solve per ambient dim or matrix normal equations
    # (Phi^T W Phi + lam I) B^T = Phi^T W target
    sw = np.sqrt(wt)
    Pw = Phi * sw[:, None]
    Tw = target * sw[:, None]
    G0 = Pw.T @ Pw
    Rhs = Pw.T @ Tw  # (q, D)

    best = None
    best_mse = float("inf")
    for lam in ridges:
        G = G0 + lam * np.eye(G0.shape[0])
        try:
            B_flat = np.linalg.solve(G, Rhs).T  # (D, q)
        except np.linalg.LinAlgError:
            B_flat = np.linalg.lstsq(G, Rhs, rcond=None)[0].T
        chart = QuadraticChart(
            mu=mu, basis=basis, B_flat=B_flat.astype(np.float64), ridge=float(lam), output_normalize=output_normalize
        )
        if len(U_va) and w_va.sum() > 0:
            pred = chart.decode(U_va)
            mse = float(np.sum(w_va * ((pred - X_va) ** 2).sum(1)) / w_va.sum())
        else:
            pred = chart.decode(Ut)
            mse = float(np.sum(wt * ((pred - Xt) ** 2).sum(1)) / wt.sum())
        if mse < best_mse:
            best_mse = mse
            best = (chart, lam, mse)
    assert best is not None
    chart, lam, mse = best
    return chart, {"ridge": lam, "val_mse": mse, "degenerate": False}
