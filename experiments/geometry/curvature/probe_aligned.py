"""Probe-aligned second fundamental form B_w = <w_N, II>."""

from __future__ import annotations

import numpy as np

from .metric import energy_g


def complete_normal_w(w: np.ndarray, proj: dict[str, np.ndarray]) -> np.ndarray:
    """w_N = (I - P_T) w."""
    return proj["P_N"] @ np.asarray(w, dtype=np.float64)


def sphere_normal_w(w: np.ndarray, proj: dict[str, np.ndarray]) -> np.ndarray:
    return proj["P_NS"] @ np.asarray(w, dtype=np.float64)


def probe_aligned_second_fundamental_form(w_N: np.ndarray, II: np.ndarray) -> np.ndarray:
    """B_w_ab = <w_N, II_ab>."""
    return np.einsum("i,iab->ab", w_N, II)


def sphere_component(w: np.ndarray, xhat: np.ndarray, g: np.ndarray) -> np.ndarray:
    """B_w^R = -(w^T x) g. Algebraic diagnostic, not representation bending."""
    return -float(np.dot(w, xhat)) * g


def shape_and_sphere(Bw: np.ndarray, w: np.ndarray, xhat: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bR = sphere_component(w, xhat, g)
    return Bw - bR, bR


def sphere_term_norm_g(bR: np.ndarray, ginv: np.ndarray, w: np.ndarray, xhat: np.ndarray, d: int) -> dict[str, float]:
    R = float(np.sqrt(max(energy_g(bR, ginv), 0.0)))
    ident = float(np.sqrt(d) * abs(float(np.dot(w, xhat))))
    return {"R": R, "identity": ident, "ok": abs(R - ident) < 1e-6}


def q_task_aligned_cross_energy(w_N: np.ndarray, BA: np.ndarray, BB: np.ndarray, ginv: np.ndarray) -> float:
    """E = <<w_N, B_A>, <w_N, B_B>>_g. Signed. Not averaged-then-squared."""
    from .quadratic import split_half_cross_energy

    bA = probe_aligned_second_fundamental_form(w_N, BA)
    bB = probe_aligned_second_fundamental_form(w_N, BB)
    return split_half_cross_energy(bA, bB, ginv)
