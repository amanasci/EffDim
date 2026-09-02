"""Numerical rank, energy rank, reachable Hessian subspace of B^S."""

from __future__ import annotations

import numpy as np

from .config import N_QUAD, ORIGINAL_ENERGY_FRAC, ORIGINAL_N_COMP_CAP, RANK_EPS_MULT


def singular_spectrum(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=np.float64)
    s = np.linalg.svd(B, compute_uv=False)
    return np.asarray(s, dtype=np.float64)


def rank_tolerance(shape: tuple[int, int], smax: float) -> float:
    """Documented numpy.linalg.matrix_rank default: max(m, n) * eps * smax."""
    m, n = int(shape[0]), int(shape[1])
    return float(RANK_EPS_MULT * max(m, n) * np.finfo(np.float64).eps * max(smax, 0.0))


def numerical_rank(S: np.ndarray, shape: tuple[int, int]) -> int:
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return 0
    tol = rank_tolerance(shape, float(S[0]))
    return int(np.sum(S > tol))


def energy_cdf(S: np.ndarray) -> np.ndarray:
    ss = np.asarray(S, dtype=np.float64) ** 2
    tot = float(np.sum(ss))
    if tot <= 0:
        return np.zeros_like(ss)
    return np.cumsum(ss) / tot


def energy_rank(S: np.ndarray, frac: float) -> int:
    cdf = energy_cdf(S)
    if cdf.size == 0:
        return 0
    return int(min(len(cdf), np.searchsorted(cdf, float(frac)) + 1))


def stable_rank(S: np.ndarray) -> float:
    ss = np.asarray(S, dtype=np.float64) ** 2
    s2 = float(np.sum(ss))
    s4 = float(np.sum(ss * ss))
    if s4 <= 0:
        return float("nan")
    return s2 * s2 / s4


def original_retained_rank(S: np.ndarray) -> int:
    """Exact rule in frozen models._bs_basis: 99% energy, then cap at 48."""
    r = energy_rank(S, ORIGINAL_ENERGY_FRAC)
    return int(max(1, min(r, len(S), ORIGINAL_N_COMP_CAP)))


def row_space_projector(B: np.ndarray, r: int | None = None, *, atol: float | None = None) -> np.ndarray:
    """P such that P γ is the projection of γ onto row(B) = range(V_r).

    B is (D, q); row space is in R^q. Thin SVD B = U Σ V^T, P = V_r V_r^T.
    """
    B = np.asarray(B, dtype=np.float64)
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    if r is None:
        r = numerical_rank(S, B.shape) if atol is None else int(np.sum(S > atol))
    r = int(max(0, min(r, Vt.shape[0])))
    if r == 0:
        return np.zeros((B.shape[1], B.shape[1]), dtype=np.float64)
    V = Vt[:r].T
    return V @ V.T


def reachable_fraction(gamma: np.ndarray, B: np.ndarray, r: int | None = None) -> float:
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    ng2 = float(g @ g)
    if ng2 < 1e-18:
        return float("nan")
    P = row_space_projector(B, r)
    pg = P @ g
    return float((pg @ pg) / ng2)


def spectrum_record(B: np.ndarray, *, tag: str, sample_id: int) -> dict:
    B = np.asarray(B, dtype=np.float64)
    S = singular_spectrum(B)
    shape = (int(B.shape[0]), int(B.shape[1]))
    nrank = numerical_rank(S, shape)
    r90 = energy_rank(S, 0.90)
    r95 = energy_rank(S, 0.95)
    r99 = energy_rank(S, 0.99)
    r_used = original_retained_rank(S)
    smax = float(S[0]) if S.size else float("nan")
    smin_ret = float(S[r_used - 1]) if r_used and r_used <= S.size else float("nan")
    cond = float(smax / max(smin_ret, 1e-18)) if np.isfinite(smax) and np.isfinite(smin_ret) else float("nan")
    cdf = energy_cdf(S)
    rec = {
        "sample_id": int(sample_id),
        "split": tag,
        "n_rows": shape[0],
        "n_cols": shape[1],
        "n_quad": int(N_QUAD),
        "numerical_rank": nrank,
        "rank_fraction_algebraic": float(nrank / max(N_QUAD, 1)),
        "r90": r90,
        "r95": r95,
        "r99": r99,
        "r_original": r_used,
        "rank_fraction_original": float(r_used / max(N_QUAD, 1)),
        "stable_rank": float(stable_rank(S)),
        "smax": smax,
        "cond_retained": cond,
        "tol": rank_tolerance(shape, smax if np.isfinite(smax) else 0.0),
        "energy_at_original": float(cdf[r_used - 1]) if r_used and r_used <= cdf.size else float("nan"),
        "svals": S.tolist(),
    }
    return rec


def sphere_normal_matrix(B: np.ndarray, x0: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, int]:
    """Project ambient B onto the complement of span(x0, J). Returns (N^T B, normal_dim)."""
    from geometry.physics_activation_atlas.sphere_normal_quadratic import sphere_project_basis

    x0n = np.asarray(x0, dtype=np.float64).reshape(-1)
    x0n = x0n / max(float(np.linalg.norm(x0n)), 1e-12)
    J = sphere_project_basis(x0n, np.asarray(J, dtype=np.float64))
    A = np.column_stack([x0n, J])
    Q, _ = np.linalg.qr(A, mode="reduced")
    from scipy.linalg import null_space

    N = null_space(Q.T, rcond=1e-8)
    Bn = N.T @ np.asarray(B, dtype=np.float64)
    return Bn, int(N.shape[1])
