"""Nested cross-fitted local PCA on sphere-tangent displacements.

One maximum-rank SVD per (anchor, scale, half); all candidate ranks are
prefixes. Agreement uses projector traces, so eigenvector signs and
rotations inside a prefix do not change A_i(d). Degenerate blocks are
accepted or rejected as a whole.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .sphere_coords import EPS

MIN_GAP_REL = 0.02


def nested_uncentred_svd(
    Z: np.ndarray,
    d_max: int,
    *,
    device: torch.device | None = None,
    centre: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return J (D, d_eff) nested prefixes and eigenvalues of C = Z^T Z / n.

    Primary analysis: do not subtract the local neighbour mean (`centre=False`).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if centre:
        Z = Z - Z.mean(axis=0, keepdims=True)
    n, D = Z.shape
    d_max = int(min(d_max, n - 1, D))
    if d_max < 1 or n < 3:
        return np.zeros((D, 0)), np.zeros(0)
    use_torch = (
        device is not None
        and getattr(device, "type", None) == "cuda"
        and torch.cuda.is_available()
        and min(n, D) > 64
    )
    if use_torch:
        Zt = torch.as_tensor(Z, device=device, dtype=torch.float32)
        try:
            q = min(max(d_max + 8, 2 * d_max), min(Zt.shape) - 1)
            _U, S, V = torch.svd_lowrank(Zt, q=q, niter=3)
            J = V[:, :d_max].detach().cpu().numpy().astype(np.float64)
            ev = (S[:d_max].detach().cpu().numpy() ** 2) / max(n, 1)
            # orthonormalize
            Q, _ = np.linalg.qr(J, mode="reduced")
            J = Q[:, : min(d_max, Q.shape[1])]
            return J, ev[: J.shape[1]]
        except Exception:  # noqa: BLE001
            pass
    _, S, Vt = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
    d_eff = min(d_max, Vt.shape[0])
    J = Vt[:d_eff].T
    ev = (S[:d_eff] ** 2) / max(n, 1)
    return J, ev


def eigengaps(ev: np.ndarray) -> np.ndarray:
    ev = np.asarray(ev, dtype=np.float64)
    g = np.full(len(ev), np.nan)
    for i in range(len(ev) - 1):
        g[i] = float(ev[i] - ev[i + 1])
    return g


def relative_eigengaps(ev: np.ndarray) -> np.ndarray:
    ev = np.asarray(ev, dtype=np.float64)
    g = np.full(len(ev), np.nan)
    for i in range(len(ev) - 1):
        g[i] = float((ev[i] - ev[i + 1]) / max(ev[i], EPS))
    return g


def degenerate_blocks(
    ev: np.ndarray,
    *,
    rel_gap_min: float = MIN_GAP_REL,
) -> list[tuple[int, int]]:
    """Inclusive 0-based index blocks. A gap below `rel_gap_min` merges ranks."""
    ev = np.asarray(ev, dtype=np.float64)
    if len(ev) == 0:
        return []
    rel = relative_eigengaps(ev)
    blocks: list[tuple[int, int]] = []
    start = 0
    for i in range(len(ev) - 1):
        if not np.isfinite(rel[i]) or rel[i] < rel_gap_min:
            continue
        blocks.append((start, i))
        start = i + 1
    blocks.append((start, len(ev) - 1))
    return blocks


def prefix_agreement(JA: np.ndarray, JB: np.ndarray, d: int) -> float:
    """A(d) = (1/d) tr(P^A_d P^B_d) = (1/d) ||JA[:d]^T JB[:d]||_F^2.

    Invariant to sign flips and to orthogonal rotations inside the prefix.
    """
    d = int(min(d, JA.shape[1], JB.shape[1]))
    if d <= 0:
        return float("nan")
    M = JA[:, :d].T @ JB[:, :d]
    return float(np.sum(M * M) / d)


def block_agreement(JA: np.ndarray, JB: np.ndarray, a: int, b: int) -> float:
    """A(a:b) = 1/(b-a+1) tr(P^A_{a:b} P^B_{a:b})."""
    if b < a:
        return float("nan")
    if JA.shape[1] <= b or JB.shape[1] <= b:
        return float("nan")
    w = b - a + 1
    M = JA[:, a : b + 1].T @ JB[:, a : b + 1]
    return float(np.sum(M * M) / w)


def incremental_agreement(JA: np.ndarray, JB: np.ndarray, d: int) -> float:
    """Added-direction (or last-block singleton) overlap: (JA_d · span JB_d)^2."""
    if d < 1 or JA.shape[1] < d or JB.shape[1] < d:
        return float("nan")
    u = JA[:, d - 1]
    # overlap with prefix-d of B (not only the d-th axis — rotation inside prefix)
    Pb_u = JB[:, :d] @ (JB[:, :d].T @ u)
    return float(np.dot(Pb_u, Pb_u) / max(np.dot(u, u), EPS))


def reconstruction_risk(Z: np.ndarray, J_train: np.ndarray, d: int) -> float:
    """E ||(I - P_d) z||^2 on rows of Z."""
    d = int(min(d, J_train.shape[1]))
    if d <= 0:
        return float(np.mean(np.sum(Z * Z, axis=1)))
    U = Z @ J_train[:, :d]
    recon = np.sum(U * U, axis=1)
    tot = np.sum(Z * Z, axis=1)
    return float(np.mean(np.maximum(tot - recon, 0.0)))


def crossfit_risk(ZA: np.ndarray, ZB: np.ndarray, JA: np.ndarray, JB: np.ndarray, d: int) -> float:
    """R(d) = 1/2 (E_B ||(I-P^A)z||^2 + E_A ||(I-P^B)z||^2)."""
    r_ba = reconstruction_risk(ZB, JA, d)
    r_ab = reconstruction_risk(ZA, JB, d)
    return 0.5 * (r_ba + r_ab)


def incremental_gain(R_prev: float, R_d: float, R0: float) -> float:
    den = max(R0, EPS)
    return float((R_prev - R_d) / den)


def radial_stratified_halves(radii: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split so both halves have comparable radius distributions."""
    radii = np.asarray(radii, dtype=np.float64)
    order = np.argsort(radii, kind="mergesort")
    rng = np.random.default_rng(seed)
    A: list[int] = []
    B: list[int] = []
    i = 0
    n = len(order)
    while i < n:
        if i + 1 >= n:
            (A if rng.random() < 0.5 else B).append(int(order[i]))
            break
        a, b = int(order[i]), int(order[i + 1])
        if rng.random() < 0.5:
            A.append(a)
            B.append(b)
        else:
            A.append(b)
            B.append(a)
        i += 2
    return np.asarray(A, dtype=np.int64), np.asarray(B, dtype=np.int64)


def rotate_block(J: np.ndarray, a: int, b: int, Q: np.ndarray) -> np.ndarray:
    """Apply orthogonal Q to columns a..b inclusive. Used in invariance tests."""
    out = J.copy()
    out[:, a : b + 1] = J[:, a : b + 1] @ Q
    return out


def flip_signs(J: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return J * signs[None, :]


def ambient_rotate(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return Z @ Q


def pca_diagnostics(ev: np.ndarray) -> dict[str, Any]:
    ev = np.asarray(ev, dtype=np.float64)
    tot = float(ev.sum()) if ev.size else 0.0
    gaps = eigengaps(ev)
    rel = relative_eigengaps(ev)
    p = ev / max(tot, EPS) if tot > 0 else ev
    return {
        "explained": p,
        "eigengaps": gaps,
        "rel_eigengaps": rel,
        "blocks": degenerate_blocks(ev),
        "total": tot,
    }
