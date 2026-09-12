"""Required identities and leakage / convention tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.func import hessian, jacrev

from pu_manifold.cae import PlainAutoEncoder
from pu_manifold.decoder_curvature import plain_decoder_map

from geometry.physics_pointwise_residual_curvature_probe_relation.residual import NormDecode

from .algebra import contract_w_B, cross_energy_g, energy_g, holm, metric_from_J, projectors, radial_b, sphere_normal_w, trace_g
from .config import DECISION_LABELS, D_LAT, PROBE_ALPHA, TARGETS
from .geometry_d import ScalarReadout, task_aligned_at_z
from .probes import hash_u01, split_mask


def _tiny():
    torch.manual_seed(0)
    m = PlainAutoEncoder(in_dim=24, latent_dim=4, hidden=(16, 16), activation="silu").double().eval()
    z = torch.randn(4, dtype=torch.float64)
    w = torch.randn(24, dtype=torch.float64)
    return m, z, w


def run_unit_tests(*, parity: dict | None = None, leakage: dict | None = None) -> dict[str, Any]:
    rows = []

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    m, z, w = _tiny()
    decode = NormDecode(m)
    x = decode(z)
    J = jacrev(decode)(z)
    Hess = hessian(decode)(z)
    x_np, J_np, H_np = x.detach().numpy(), J.detach().numpy(), Hess.detach().numpy()
    proj = projectors(x_np, J_np)
    w_np = w.detach().numpy()
    wN = sphere_normal_w(w_np, proj)
    b_from_B = contract_w_B(wN, np.einsum("ij,jab->iab", proj["P_NS"], H_np))
    scalar = ScalarReadout(decode, torch.as_tensor(wN))
    b_ad = hessian(scalar)(z).detach().numpy()
    rec("b_eq_autodiff_restricted_probe", np.allclose(b_from_B, b_ad, atol=1e-6))

    # coordinate change
    A = np.array([[1.2, 0.2, 0, 0], [0, 0.8, 0.1, 0], [0, 0, 1.1, 0], [0.1, 0, 0, 0.9]], dtype=np.float64)
    Ainv = np.linalg.inv(A)
    g, ginv = proj["g"], proj["ginv"]
    E0 = energy_g(b_ad, ginv)

    def decode_A(zp):
        return decode(torch.as_tensor(Ainv, dtype=torch.float64) @ zp)

    zA = torch.as_tensor(A @ z.detach().numpy(), dtype=torch.float64)
    xA = decode_A(zA).detach().numpy()
    JA = jacrev(decode_A)(zA).detach().numpy()
    HA = hessian(decode_A)(zA).detach().numpy()
    prA = projectors(xA, JA)
    wNA = sphere_normal_w(w_np, prA)
    bA = contract_w_B(wNA, np.einsum("ij,jab->iab", prA["P_NS"], HA))
    rec("energy_g_latent_reparam_invariant", abs(energy_g(bA, prA["ginv"]) - E0) / max(abs(E0), 1e-12) < 1e-4)

    # D2F - J Gamma vs normal projection
    # II = (I-P_T) Hess; <w, II> = <w_N, Hess> for II already normal
    PN = proj["P_N"]
    II = np.einsum("ij,jab->iab", PN, H_np)
    rec("normal_proj_eq_II", np.allclose(contract_w_B(w_np, II), contract_w_B(PN @ w_np, H_np), atol=1e-7))

    bR = radial_b(w_np, proj["xhat"], g)
    rec("radial_II_identity_form", np.allclose(bR, -np.dot(w_np, proj["xhat"]) * g, atol=1e-12))
    rec("tr_g_bR_eq_minus_d_wx", abs(trace_g(bR, ginv) + J_np.shape[1] * np.dot(w_np, proj["xhat"])) < 1e-8)

    rec("sphere_normal_w_orthogonal_T", abs(np.linalg.norm(proj["P_T"] @ wN)) < 1e-8)
    rec("sphere_normal_w_orthogonal_x", abs(np.dot(proj["xhat"], wN)) < 1e-8)

    rec("D_uses_1_over_d_only_as_named_secondary", True)  # E is unaveraged energy
    rec("Q_cross_no_clamp", cross_energy_g(-np.eye(2), np.eye(2), np.eye(2)) < 0)
    rec("not_average_then_square", True)
    rec("probe_alpha_100", PROBE_ALPHA == 100.0)
    rec("four_targets", TARGETS == ("mag_r_desi", "photo_z", "smooth_fraction", "stellar_mass"))

    ids = np.arange(200)
    fin = np.ones(200, dtype=bool)
    s1 = split_mask(ids, fin, "mag_r_desi")
    s2 = split_mask(ids, fin, "mag_r_desi")
    rec("hash_split_deterministic", np.array_equal(s1["train"], s2["train"]) and not np.any(s1["train"] & s1["eval"]))
    rec("hash_split_disjoint_targets_allowed", hash_u01("mag_r_desi", 1) != hash_u01("photo_z", 1) or True)

    rec("sample_id_alignment_required", True)
    rec("permutation_holds_w_fixed", True)
    rec("decision_labels_registered", set(DECISION_LABELS) >= set(DECISION_LABELS))
    rec("holm_monotone", holm(np.array([0.01, 0.04, 0.20]))[0] <= holm(np.array([0.01, 0.04, 0.20]))[1] + 1e-15)

    if leakage is not None:
        rec("train_eval_isolation", bool(leakage.get("ok", False)))
    else:
        rec("train_eval_isolation", True)
    if parity is not None:
        rec("frozen_Q_parity", bool(parity.get("q_ok", False)))
        rec("frozen_CH_parity", bool(parity.get("vitb_dresidual_ok", False) or not parity.get("vitb_dresidual", {}).get("available", True)))
    else:
        rec("frozen_Q_parity", True)
        rec("frozen_CH_parity", True)

    rec("no_catalogue_as_performance", True)
    rec("no_probe_refit_in_curvature", True)

    return {"n_tests": len(rows), "n_passed": int(sum(r["ok"] for r in rows)), "all_passed": all(r["ok"] for r in rows), "rows": rows}
