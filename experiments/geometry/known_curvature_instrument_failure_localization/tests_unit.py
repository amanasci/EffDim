"""Focused unit tests for the failure-localization algebra."""

from __future__ import annotations

import numpy as np
import torch

from geometry.known_curvature_point_patch_fixture_audit.estimator_d import split_indices
from geometry.known_curvature_point_patch_fixture_audit.fixtures import (
    autodiff_geometry,
    f0_np,
    f2_np,
)
from geometry.known_curvature_point_patch_fixture_audit.geometry import (
    hessian_fd,
    hess_from_bs_flat,
    pack_BS,
    rel_err,
    unpack_BS_symmetric,
)
from geometry.known_curvature_point_patch_fixture_audit.io_util import hash_stable_order

from .config import D_LAT, SPLIT_SEED
from .q_ladder import align_tensor, fit_residual_quadratic, procrustes_R, quad_phi


def run_unit_tests(Q: np.ndarray) -> dict:
    rows = []

    def rec(name: str, ok: bool, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    rng = np.random.default_rng(0)
    d = 6
    n = 400
    U = rng.standard_normal((n, d))
    # Known quadratic: resid = Phi(u) S^T with S ambient
    S_true = rng.standard_normal((12, d * (d + 1) // 2)) * 0.05
    Phi = quad_phi(U)
    resid = Phi @ S_true.T
    J = np.eye(12, d)
    x0 = np.zeros(12)
    x0[0] = 1.0
    Y = x0[None, :] + U @ J.T + resid
    # Bypass sphere projectors by using a dummy that keeps residual: fit in R^12
    # Use pinv path with U exact.
    from geometry.known_curvature_point_patch_fixture_audit.geometry import hess_from_bs_flat as h2

    G = Phi.T @ Phi
    Shat = np.linalg.lstsq(Phi, resid, rcond=None)[0].T
    rec("exact_frame_known_quadratic", rel_err(Shat, S_true) < 1e-8, rel=rel_err(Shat, S_true))

    B = rng.standard_normal((8, 4, 4))
    B = 0.5 * (B + np.transpose(B, (0, 2, 1)))
    packed = pack_BS(B)
    unp = unpack_BS_symmetric(packed, 4)
    rec("pack_unpack_contraction", rel_err(unp, B) < 1e-12)

    J1, _ = np.linalg.qr(rng.standard_normal((20, 5)))
    Rtrue, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    J2 = J1 @ Rtrue
    Rhat = procrustes_R(J1, J2)
    rec("procrustes_rotation", rel_err(Rhat, Rtrue) < 1e-8)

    # Radial subtraction: residual after PNS should be ⟂ x0 and ⟂ T
    z0 = 0.2 * np.ones(D_LAT) / np.sqrt(D_LAT)
    geo = autodiff_geometry("F0", z0, Q)
    PNS = geo["P_NS"]
    v = rng.standard_normal(geo["G"].shape[0])
    vn = PNS @ v
    rec("exact_radial_subtraction", abs(np.dot(vn, geo["G_hat"])) < 1e-8 and np.max(np.abs(geo["J"].T @ vn)) < 1e-6)

    geo2 = autodiff_geometry("F2", z0, Q)
    rec("F2_truth_zero_trace_mean", geo2["H_norm"] < 1e-5, H=geo2["H_norm"])

    geo0 = autodiff_geometry("F0", z0, Q)
    rec("F0_truth_zero_residual_B", geo0["B_fro"] < 1e-8, B=geo0["B_fro"])

    # Q1 on an exactly quadratic chart: F0 has B^S=0, so exact-frame LS must recover ~0.
    zc = 0.15 * np.ones(D_LAT) / np.sqrt(D_LAT)
    g0q = autodiff_geometry("F0", zc, Q)
    Uq = 0.02 * rng.standard_normal((300, D_LAT))
    from geometry.known_curvature_point_patch_fixture_audit.fixtures import GENERATORS_NP, f0_np

    Yq = f0_np(zc[None, :] + Uq, Q)
    fit = fit_residual_quadratic(Yq, g0q["G"], g0q["J"], U=Uq, ridge=0.0)
    rec(
        "Q1_recovers_exact_quadratic_chart",
        fit["K_dir"] < 1e-6 and fit["B_fro"] < 1e-4,
        K_dir=fit["K_dir"],
        B_fro=fit["B_fro"],
    )

    # Autodiff vs FD on the generator (same path D uses for Hessians)
    g1 = autodiff_geometry("F1", zc, Q)
    fn = lambda u: GENERATORS_NP["F1"](u, Q)
    Hfd = hessian_fd(fn, zc, 1e-4)
    rec("D_path_autodiff_vs_fd", rel_err(Hfd, g1["Hess"]) < 5e-2, rel=rel_err(Hfd, g1["Hess"]))

    ids = np.arange(30)
    a = hash_stable_order(ids, 1)
    b = hash_stable_order(ids[::-1], 1)
    rec("sample_id_invariance", np.array_equal(a, b))

    tr, ho = split_indices(100, SPLIT_SEED)
    rec("split_no_overlap", len(set(tr).intersection(set(ho))) == 0)

    n_pass = sum(r["ok"] for r in rows)
    return {"n_tests": len(rows), "n_passed": int(n_pass), "all_passed": n_pass == len(rows), "rows": rows}
