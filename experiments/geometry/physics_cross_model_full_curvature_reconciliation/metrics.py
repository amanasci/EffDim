"""Historical full-curvature algebra. Verbatim port of cross_metric_pair.

Primary historical scalar (Phase 0):

    K_dir^cross = <H_A, H_B> + [2 / (d (d+2))] <B0_A, B0_B>_F

which is identical to the direction-averaged split-cross formula

    K_dir^cross = (2 <B_A, B_B>_F + <tr B_A, tr B_B>) / (d (d+2))

with H = (1/d) tr B in ambient space. Chart coordinates are the frozen
orthonormal PCA frame; Euclidean Frobenius on the unpacked (D, d, d)
tensor is the frozen convention. Negative cross-estimates are not clamped
before rank correlations. sqrt(max(cross, 0)) is visualization only.

Source of truth for the production fit path remains
`physics_activation_atlas.effdim_curvature_metrics.cross_metric_pair`.
This module duplicates the algebra so unit tests do not import torch.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def aniso_prefactor(d: int) -> float:
    return 2.0 / float(d * (d + 2))


def unpack_BS_symmetric(BS_flat: np.ndarray, d: int) -> np.ndarray:
    """Unpack (D,q) flat a<=b coeffs into symmetric B[D,d,d]. Off-diag: flat/2."""
    D = BS_flat.shape[0]
    B = np.zeros((D, d, d), dtype=np.float64)
    idx = 0
    for a in range(d):
        for b in range(a, d):
            if a == b:
                B[:, a, a] = BS_flat[:, idx]
            else:
                B[:, a, b] = 0.5 * BS_flat[:, idx]
                B[:, b, a] = 0.5 * BS_flat[:, idx]
            idx += 1
    return B


def pack_BS(B: np.ndarray) -> np.ndarray:
    """Pack symmetric B[D,d,d] into frozen flat (D,q) with off-diag = 2 B_ab."""
    _, d, _ = B.shape
    cols = []
    for a in range(d):
        for b in range(a, d):
            cols.append(B[:, a, a] if a == b else (2.0 * B[:, a, b]))
    return np.stack(cols, axis=1)


def decompose_tensors(BS_flat: np.ndarray, d: int) -> dict[str, np.ndarray | float]:
    B = unpack_BS_symmetric(BS_flat, d)
    H = B[:, np.arange(d), np.arange(d)].mean(axis=1)
    B0 = B.copy()
    for a in range(d):
        B0[:, a, a] = B[:, a, a] - H
    kh2 = float(np.dot(H, H))
    b0f = float(np.linalg.norm(B0) ** 2)
    bsf = float(np.linalg.norm(B) ** 2)
    ka2 = aniso_prefactor(d) * b0f
    kd2 = kh2 + ka2
    return {
        "H": H,
        "B0": B0,
        "B": B,
        "K_H": float(np.sqrt(max(kh2, 0.0))),
        "K_H2": kh2,
        "K_aniso2": ka2,
        "K_aniso": float(np.sqrt(max(ka2, 0.0))),
        "K_dir2": kd2,
        "K_dir": float(np.sqrt(max(kd2, 0.0))),
        "B_fro": float(np.sqrt(bsf)),
        "B0_fro": float(np.sqrt(b0f)),
    }


def metric_scalars(BS_flat: np.ndarray, d: int) -> dict[str, float]:
    t = decompose_tensors(BS_flat, d)
    return {k: float(t[k]) for k in ("K_H", "K_H2", "K_aniso", "K_aniso2", "K_dir", "K_dir2", "B_fro", "B0_fro")}


def tensor_agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    inner = float(np.dot(a, b))
    denom = na * nb
    return {
        "inner": inner,
        "R_signal": float(inner / denom) if denom > EPS else float("nan"),
        "norm_A": na,
        "norm_B": nb,
    }


def cross_metric_pair(BSA: np.ndarray, BSB: np.ndarray, d: int) -> dict[str, float]:
    a = decompose_tensors(BSA, d)
    b = decompose_tensors(BSB, d)
    khx = float(np.dot(a["H"], b["H"]))
    b0x = float(np.sum(a["B0"] * b["B0"]))
    kax = aniso_prefactor(d) * b0x
    kdx = khx + kax
    return {
        "K_H_cross": khx,
        "K_aniso_cross": kax,
        "K_dir_cross": kdx,
        "K_H_cross_plot": float(max(khx, 0.0)),
        "K_aniso_cross_plot": float(max(kax, 0.0)),
        "K_dir_cross_plot": float(max(kdx, 0.0)),
        "R_H": tensor_agreement(a["H"], b["H"])["R_signal"],
        "R_B0": tensor_agreement(a["B0"].ravel(), b["B0"].ravel())["R_signal"],
        "R_BS": tensor_agreement(a["B"].ravel(), b["B"].ravel())["R_signal"],
        "norm_H_mean": 0.5 * (a["K_H"] + b["K_H"]),
        "norm_dir_mean": 0.5 * (a["K_dir"] + b["K_dir"]),
        "norm_aniso_mean": 0.5 * (a["K_aniso"] + b["K_aniso"]),
    }


def monte_carlo_K_dir2(BS_flat: np.ndarray, d: int, n_dir: int = 4000, seed: int = 0) -> float:
    B = unpack_BS_symmetric(BS_flat, d)
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n_dir, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True) + EPS
    Bvv = np.einsum("ia,jab,ib->ij", V, B, V)
    return float(np.mean(np.sum(Bvv**2, axis=1)))


def tensor_frobenius_inner(BA: np.ndarray, BB: np.ndarray) -> float:
    return float(np.sum(BA * BB))


def packed_sqrt2_inner_from_flat(flatA: np.ndarray, flatB: np.ndarray, d: int) -> float:
    """√2 weights on unique packed *B_ab* (not on the frozen 2 B_ab flats)."""
    BA = unpack_BS_symmetric(flatA, d)
    BB = unpack_BS_symmetric(flatB, d)
    acc = 0.0
    for a in range(d):
        acc += float(np.dot(BA[:, a, a], BB[:, a, a]))
        for b in range(a + 1, d):
            acc += 2.0 * float(np.dot(BA[:, a, b], BB[:, a, b]))
    return acc


def kb_from_kh_kaniso(kh: float, kaniso: float, d: int) -> float:
    pref = aniso_prefactor(d)
    return float(kaniso / pref + d * kh)


def kdir_from_kb_kh(kb: float, kh: float, d: int) -> float:
    tr_inner = float(d * d * kh)
    return float((2.0 * kb + tr_inner) / float(d * (d + 2)))


def normal_projector_apply(V: np.ndarray, x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    x0 = (x0 / max(np.linalg.norm(x0), EPS)).astype(np.float64)
    Q, _ = np.linalg.qr(np.column_stack([x0, J]), mode="reduced")
    if V.ndim == 1:
        return V - Q @ (Q.T @ V)
    return V - Q @ (Q.T @ V)


def sphere_normal_residual(Q: np.ndarray, x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    """B^S_ab = P_{N,S} Q_ab. Do not subtract Q^R separately after this projection."""
    d = Q.shape[1]
    B = np.zeros_like(Q)
    for a in range(d):
        for b in range(d):
            B[:, a, b] = normal_projector_apply(Q[:, a, b], x0, J)
    return B


def forced_radial_Q(x0: np.ndarray, J: np.ndarray) -> np.ndarray:
    d = J.shape[1]
    g = J.T @ J
    x0u = x0 / max(np.linalg.norm(x0), EPS)
    return -g[None, :, :] * x0u[:, None, None]


def orthogonality_residuals(B: np.ndarray, x0: np.ndarray, J: np.ndarray) -> dict[str, float]:
    x0u = x0 / max(np.linalg.norm(x0), EPS)
    rad = float(np.max(np.abs(np.einsum("i,iab->ab", x0u, B))))
    tan = float(np.max(np.abs(np.einsum("ia,iab->ab", J, B))))
    return {"max_abs_x0_dot": rad, "max_abs_J_dot": tan}


def directional_mc_cross(BSA: np.ndarray, BSB: np.ndarray, n_dir: int = 4000, seed: int = 0) -> float:
    d = BSA.shape[1]
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n_dir, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True) + EPS
    Avv = np.einsum("ia,jab,ib->ij", V, BSA, V)
    Bvv = np.einsum("ia,jab,ib->ij", V, BSB, V)
    return float(np.mean(np.sum(Avv * Bvv, axis=1)))


def split_scalars(flatA: np.ndarray, flatB: np.ndarray, d: int) -> dict[str, float]:
    cross = cross_metric_pair(flatA, flatB, d)
    sa, sb = metric_scalars(flatA, d), metric_scalars(flatB, d)
    a = decompose_tensors(flatA, d)
    b = decompose_tensors(flatB, d)
    kb = tensor_frobenius_inner(a["B"], b["B"])
    return {
        **cross,
        "K_B_cross": kb,
        "K_dir_identity": kdir_from_kb_kh(kb, cross["K_H_cross"], d),
        "K_H_A": sa["K_H"],
        "K_H_B": sb["K_H"],
        "K_dir_A": sa["K_dir"],
        "K_dir_B": sb["K_dir"],
        "K_aniso_A": sa["K_aniso"],
        "K_aniso_B": sb["K_aniso"],
        "B_fro_A": sa["B_fro"],
        "B_fro_B": sb["B_fro"],
        "B0_fro_A": sa["B0_fro"],
        "B0_fro_B": sb["B0_fro"],
        "trace_energy_mean": 0.5 * (float(d * sa["K_H"] ** 2) + float(d * sb["K_H"] ** 2)),
        "traceless_energy_mean": 0.5 * (sa["B0_fro"] ** 2 + sb["B0_fro"] ** 2),
        "full_energy_mean": 0.5 * (sa["B_fro"] ** 2 + sb["B_fro"] ** 2),
    }
