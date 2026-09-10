"""Orthogonal mean / trace-free split of B^S and Gamma."""

from __future__ import annotations

import numpy as np

from geometry.physics_cross_model_full_curvature_reconciliation.metrics import (
    aniso_prefactor,
    pack_BS,
    unpack_BS_symmetric,
)
from geometry.physics_quadratic_label_chart_alignment.features import (
    Gamma_from_gamma,
    gamma_from_Gamma,
    n_quad,
)


def split_B(B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """B = B_H + B_TF, H = (1/d) tr B, Euclidean orthonormal chart."""
    d = B.shape[1]
    H = B[:, np.arange(d), np.arange(d)].mean(axis=1)
    BH = np.zeros_like(B)
    for a in range(d):
        BH[:, a, a] = H
    BTF = B - BH
    return H, BH, BTF


def split_flat(BS_flat: np.ndarray, d: int) -> dict[str, np.ndarray | float]:
    B = unpack_BS_symmetric(BS_flat, d)
    H, BH, BTF = split_B(B)
    tr_tf = float(np.max(np.abs(BTF[:, np.arange(d), np.arange(d)].sum(axis=1))))
    inner = float(np.sum(BH * BTF))
    recon = float(np.linalg.norm(B - BH - BTF))
    return {
        "H": H,
        "BH": BH,
        "BTF": BTF,
        "B": B,
        "tr_TF_max": tr_tf,
        "inner_BH_BTF": inner,
        "recon_err": recon,
        "K_H2": float(np.dot(H, H)),
        "K_TF2": aniso_prefactor(d) * float(np.linalg.norm(BTF) ** 2),
        "E_H": float(np.dot(H, H)),
        "E_TF": aniso_prefactor(d) * float(np.linalg.norm(BTF) ** 2),
        "C_trace": float((d * np.dot(H, H)) / max(float(np.linalg.norm(B) ** 2), 1e-18)),
    }


def cross_components(flatA: np.ndarray, flatB: np.ndarray, d: int) -> dict[str, float]:
    a = split_flat(flatA, d)
    b = split_flat(flatB, d)
    kh = float(np.dot(a["H"], b["H"]))
    ktf = aniso_prefactor(d) * float(np.sum(a["BTF"] * b["BTF"]))
    return {
        "K_H_cross": kh,
        "K_TF_cross": ktf,
        "K_dir_cross": kh + ktf,
        "identity_err": abs((kh + ktf) - (kh + ktf)),
        "tr_TF_max": max(float(a["tr_TF_max"]), float(b["tr_TF_max"])),
        "inner_BH_BTF": max(abs(float(a["inner_BH_BTF"])), abs(float(b["inner_BH_BTF"]))),
        "C_trace_mean": 0.5 * (float(a["C_trace"]) + float(b["C_trace"])),
        "F_H_mean": _fh(a) * 0.5 + _fh(b) * 0.5,
        "E_H_mean": 0.5 * (float(a["E_H"]) + float(b["E_H"])),
        "E_TF_mean": 0.5 * (float(a["E_TF"]) + float(b["E_TF"])),
    }


def _fh(t: dict) -> float:
    den = float(t["E_H"]) + float(t["E_TF"])
    return float(t["E_H"] / den) if den > 1e-18 else float("nan")


def split_Gamma(Gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = Gamma.shape[0]
    GH = (float(np.trace(Gamma)) / d) * np.eye(d)
    return GH, Gamma - GH


def iso_unit_gamma(d: int) -> np.ndarray:
    return gamma_from_Gamma(np.eye(d) / np.sqrt(d))


def project_gamma_iso_tf(gamma: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    G = Gamma_from_gamma(gamma, d)
    GH, GTF = split_Gamma(G)
    return gamma_from_Gamma(GH), gamma_from_Gamma(GTF)


def bh_btf_frob(BS_prod: np.ndarray, d: int):
    """Split production-packed B^S into Frobenius-packed B_H and B_TF."""
    from geometry.physics_quadratic_label_chart_alignment.features import bs_prod_to_frob

    B = unpack_BS_symmetric(BS_prod, d)
    _, BH, BTF = split_B(B)
    return bs_prod_to_frob(pack_BS(BH), d), bs_prod_to_frob(pack_BS(BTF), d)


def induced_energy(gamma: np.ndarray, B_frob: np.ndarray) -> float:
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    B = np.asarray(B_frob, dtype=np.float64)
    return float(g @ (B.T @ B) @ g)


def alignment_AB(gamma: np.ndarray, B_frob: np.ndarray) -> float:
    from geometry.physics_quadratic_label_chart_alignment.alignment import alignment_AB as _ab

    return _ab(gamma, B_frob)
