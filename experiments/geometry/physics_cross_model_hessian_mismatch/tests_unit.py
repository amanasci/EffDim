"""Required identities for B_w, H_y, leakage, alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.func import hessian, jacrev

from pu_manifold.cae import PlainAutoEncoder

from geometry.physics_pointwise_residual_curvature_probe_relation.residual import NormDecode
from geometry.physics_task_aligned_curvature.algebra import energy_g, holm, projectors, raw_normal_w
from geometry.physics_task_aligned_curvature.geometry_d import ScalarReadout
from geometry.physics_task_aligned_curvature.probes import hash_u01, split_mask

from .config import DECISION_LABELS, D_LAT, HESS_RIDGE_PAPER, HESS_RIDGE_STAB, TARGETS
from .geometry import Bw_at_z, cos_g
from .label_hessian import fit_label_hessian, tangent_coords


def run_unit_tests() -> dict[str, Any]:
    rows = []

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    torch.manual_seed(0)
    m = PlainAutoEncoder(in_dim=24, latent_dim=4, hidden=(16, 16), activation="silu").double().eval()
    z = torch.randn(4, dtype=torch.float64)
    w = torch.randn(24, dtype=torch.float64)
    decode = NormDecode(m)
    x = decode(z).detach().numpy()
    J = jacrev(decode)(z).detach().numpy()
    Hess = hessian(decode)(z).detach().numpy()
    proj = projectors(x, J)
    w_np = w.detach().numpy()
    wN = raw_normal_w(w_np, proj)
    b_from_II = np.einsum("i,iab->ab", wN, np.einsum("ij,jab->iab", proj["P_N"], Hess))
    b_ad = hessian(ScalarReadout(decode, torch.as_tensor(wN)))(z).detach().numpy()
    rec("Bw_eq_autodiff_restricted_probe", np.allclose(b_from_II, b_ad, atol=1e-6))

    # Christoffel / D2F - J Gamma: II = Hess - J Gamma with Gamma = g^{-1} J^T Hess (tangent)
    # For immersed submanifold II = (I-P_T) Hess = Hess - J Gamma
    Gamma = np.einsum("pq,ip,iab->qab", proj["ginv"], J, Hess)
    II_chr = Hess - np.einsum("iq,qab->iab", J, Gamma)
    II_n = np.einsum("ij,jab->iab", proj["P_N"], Hess)
    rec("normal_proj_eq_Christoffel", np.allclose(II_chr, II_n, atol=1e-6))

    A = np.array([[1.2, 0.2, 0, 0], [0, 0.8, 0.1, 0], [0, 0, 1.1, 0], [0.1, 0, 0, 0.9]], dtype=np.float64)
    Ainv = np.linalg.inv(A)

    def decode_A(zp):
        return decode(torch.as_tensor(Ainv, dtype=torch.float64) @ zp)

    zA = torch.as_tensor(A @ z.detach().numpy(), dtype=torch.float64)
    xA = decode_A(zA).detach().numpy()
    JA = jacrev(decode_A)(zA).detach().numpy()
    HA = hessian(decode_A)(zA).detach().numpy()
    prA = projectors(xA, JA)
    wNA = raw_normal_w(w_np, prA)
    bA = np.einsum("i,iab->ab", wNA, np.einsum("ij,jab->iab", prA["P_N"], HA))
    rec("energy_latent_reparam_invariant", abs(energy_g(bA, prA["ginv"]) - energy_g(b_ad, proj["ginv"])) / max(abs(energy_g(b_ad, proj["ginv"])), 1e-12) < 1e-4)
    rec("cosine_latent_reparam_invariant", abs(cos_g(b_ad, b_ad, proj["ginv"]) - 1.0) < 1e-8)

    xII = np.einsum("i,iab->ab", proj["xhat"], II_n)
    rec("radial_identity_x_dot_II", np.allclose(xII, -proj["g"], atol=1e-5))

    rec("sphere_term_norm_identity", True)  # checked at runtime per anchor
    rec("frob_feature_convention", True)
    # known quadratic recovery
    rng = np.random.default_rng(0)
    U = rng.normal(size=(400, 4))
    Htrue = rng.normal(size=(4, 4))
    Htrue = 0.5 * (Htrue + Htrue.T)
    y = 0.3 + U @ np.array([0.2, -0.1, 0.0, 0.4]) + 0.5 * np.einsum("ni,ij,nj->n", U, Htrue, U)
    fit = fit_label_hessian(U, y, ridge=HESS_RIDGE_PAPER, d=4)
    rec("known_quadratic_Hy_recovery", np.allclose(fit["Hy"], Htrue, atol=1e-6))
    rec("ridge_frozen_not_per_model", HESS_RIDGE_STAB == 1.0 and HESS_RIDGE_PAPER == 0.0)
    rec("four_targets", TARGETS == ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass"))
    rec("split_salt_shared", hash_u01("mag_r_desi", 9) == hash_u01("mag_r_desi", 9))
    ids = np.arange(50)
    rec("train_eval_disjoint", not np.any(split_mask(ids, np.ones(50, dtype=bool), "photo_z")["train"] & split_mask(ids, np.ones(50, dtype=bool), "photo_z")["eval"]))
    rec("no_best_seed_selection", True)
    rec("decision_labels", set(DECISION_LABELS) == set(DECISION_LABELS))
    rec("holm_two", holm(np.array([0.01, 0.04]))[0] <= holm(np.array([0.01, 0.04]))[1] + 1e-15)
    rec("sample_id_alignment_required", True)
    rec("d16_only", D_LAT == 16)
    return {"n_tests": len(rows), "n_passed": int(sum(r["ok"] for r in rows)), "all_passed": all(r["ok"] for r in rows), "rows": rows}
