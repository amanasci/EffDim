"""Unit and synthetic tests for the recovered full-curvature algebra."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .metrics import (
    cross_metric_pair,
    directional_mc_cross,
    forced_radial_Q,
    metric_scalars,
    monte_carlo_K_dir2,
    orthogonality_residuals,
    pack_BS,
    packed_sqrt2_inner_from_flat,
    sphere_normal_residual,
    split_scalars,
    tensor_frobenius_inner,
    normal_projector_apply,
)


def _frame(D: int = 20, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=D)
    x0 /= np.linalg.norm(x0)
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    J = J - np.outer(x0, x0 @ J)
    J, _ = np.linalg.qr(J)
    return x0, J[:, :d], rng


def _rand_sym(D: int, d: int, rng: np.random.Generator) -> np.ndarray:
    B = rng.normal(size=(D, d, d))
    return 0.5 * (B + np.transpose(B, (0, 2, 1)))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float(np.dot(rx, ry) / (np.linalg.norm(rx) * np.linalg.norm(ry)))


def test_pure_sphere() -> None:
    x0, J, _ = _frame()
    Q = forced_radial_Q(x0, J)
    B = sphere_normal_residual(Q, x0, J)
    assert float(np.linalg.norm(B)) < 1e-10


def test_pure_tangential() -> None:
    x0, J, rng = _frame()
    d = J.shape[1]
    coeff = rng.normal(size=(d, d, d))
    coeff = 0.5 * (coeff + np.transpose(coeff, (0, 2, 1)))
    Q = np.einsum("ia,abc->ibc", J, coeff)
    B = sphere_normal_residual(Q, x0, J)
    assert float(np.linalg.norm(B)) < 1e-10


def test_pure_mean_bending() -> None:
    x0, J, _ = _frame(d=6)
    e = np.zeros_like(x0)
    e[0] = 1.0
    nvec = normal_projector_apply(e, x0, J)
    nvec /= np.linalg.norm(nvec)
    d = J.shape[1]
    B = np.zeros((x0.size, d, d))
    for a in range(d):
        B[:, a, a] = nvec
    s = metric_scalars(pack_BS(B), d)
    assert s["K_H"] > 0.5
    assert s["K_dir"] > 0.5
    assert s["K_aniso"] < 1e-8


def test_traceless_saddle() -> None:
    x0, J, _ = _frame(d=4)
    e = np.zeros_like(x0)
    e[1] = 1.0
    nvec = normal_projector_apply(e, x0, J)
    nvec /= np.linalg.norm(nvec)
    d = J.shape[1]
    B = np.zeros((x0.size, d, d))
    B[:, 0, 0] = nvec
    B[:, 1, 1] = -nvec
    s = metric_scalars(pack_BS(B), d)
    assert abs(s["K_H"]) < 1e-8
    assert s["K_dir"] > 0.05
    assert s["K_aniso"] > 0.05


def test_coordinate_rotation() -> None:
    x0, J, rng = _frame(d=5)
    B = sphere_normal_residual(_rand_sym(x0.size, 5, rng), x0, J)
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    Br = np.einsum("ia,dab,jb->dij", Q, B, Q)
    s0 = metric_scalars(pack_BS(B), 5)
    s1 = metric_scalars(pack_BS(Br), 5)
    for k in ("K_H", "K_aniso", "K_dir", "B_fro"):
        assert abs(s0[k] - s1[k]) < 1e-8


def test_ambient_rotation() -> None:
    x0, J, rng = _frame(d=5)
    B = sphere_normal_residual(_rand_sym(x0.size, 5, rng), x0, J)
    R, _ = np.linalg.qr(rng.normal(size=(x0.size, x0.size)))
    Br = np.einsum("ij,jab->iab", R, B)
    s0 = metric_scalars(pack_BS(B), 5)
    s1 = metric_scalars(pack_BS(Br), 5)
    for k in ("K_H", "K_aniso", "K_dir", "B_fro"):
        assert abs(s0[k] - s1[k]) < 1e-8


def test_packed_sqrt2() -> None:
    _, _, rng = _frame(d=6, D=18)
    B = _rand_sym(18, 6, rng)
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    flat = pack_BS(B)
    t = tensor_frobenius_inner(B, B)
    p = packed_sqrt2_inner_from_flat(flat, flat, 6)
    assert abs(t - p) / max(abs(t), 1e-12) < 1e-10


def test_directional_identity() -> None:
    _, _, rng = _frame(d=5, D=16)
    B = _rand_sym(16, 5, rng)
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    flat = pack_BS(B)
    closed = metric_scalars(flat, 5)["K_dir2"]
    mc = monte_carlo_K_dir2(flat, 5, n_dir=12000, seed=1)
    assert abs(mc - closed) / max(closed, 1e-8) < 0.15


def test_split_cross_identity() -> None:
    _, _, rng = _frame(d=5, D=16)
    A = 0.5 * (_rand_sym(16, 5, rng) + np.transpose(_rand_sym(16, 5, rng), (0, 2, 1)))
    C = 0.5 * (_rand_sym(16, 5, rng) + np.transpose(_rand_sym(16, 5, rng), (0, 2, 1)))
    sc = split_scalars(pack_BS(A), pack_BS(C), 5)
    mc = directional_mc_cross(A, C, n_dir=12000, seed=2)
    assert abs(sc["K_dir_cross"] - sc["K_dir_identity"]) < 1e-10
    assert abs(mc - sc["K_dir_cross"]) / max(abs(sc["K_dir_cross"]), 1e-8) < 0.15
    # unused import guard: historical name still exported
    assert "K_H_cross" in cross_metric_pair(pack_BS(A), pack_BS(C), 5)


def test_radial_tangential_orthogonality() -> None:
    x0, J, rng = _frame(d=5)
    Q = _rand_sym(x0.size, 5, rng)
    B = sphere_normal_residual(Q, x0, J)
    o = orthogonality_residuals(B, x0, J)
    assert o["max_abs_x0_dot"] < 1e-10
    assert o["max_abs_J_dot"] < 1e-10


def test_sample_alignment() -> None:
    rng = np.random.default_rng(4)
    n = 64
    df = pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "K_dir_cross": rng.normal(size=n),
            "r2_G": rng.normal(size=n),
        }
    )
    rho0 = _spearman(df.K_dir_cross.to_numpy(float), df.r2_G.to_numpy(float))
    shuf = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    aligned = shuf.merge(df[["sample_id", "r2_G"]], on="sample_id", suffixes=("", "_orig"))
    rho1 = _spearman(aligned.K_dir_cross.to_numpy(float), aligned.r2_G.to_numpy(float))
    assert abs(rho0 - rho1) < 1e-12


TESTS: list[tuple[str, Callable[[], None]]] = [
    ("pure_sphere", test_pure_sphere),
    ("pure_tangential", test_pure_tangential),
    ("pure_mean_bending", test_pure_mean_bending),
    ("traceless_saddle", test_traceless_saddle),
    ("coordinate_rotation", test_coordinate_rotation),
    ("ambient_rotation", test_ambient_rotation),
    ("packed_sqrt2", test_packed_sqrt2),
    ("directional_identity", test_directional_identity),
    ("split_cross_identity", test_split_cross_identity),
    ("radial_tangential_orthogonality", test_radial_tangential_orthogonality),
    ("sample_alignment", test_sample_alignment),
]


def run_unit_tests() -> dict[str, Any]:
    rows = []
    for name, fn in TESTS:
        try:
            fn()
            rows.append({"name": name, "ok": True, "error": ""})
        except Exception as exc:  # noqa: BLE001
            rows.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "n": len(rows),
        "n_pass": int(sum(r["ok"] for r in rows)),
        "ok": all(r["ok"] for r in rows),
        "rows": rows,
    }
