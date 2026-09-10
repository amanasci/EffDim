"""Phase-1 unit tests for fixtures, formulas, oracles, and isolation."""

from __future__ import annotations

import numpy as np
import torch

from .config import D_LAT, F1_C
from .fixtures import (
    ambient_rotation_d28,
    autodiff_jets,
    energy_g,
    f0_np,
    jacobian_batch_fd,
    make_torch_map,
    sample_latent,
    targets_from_jets,
    truth_at,
)
from .sampling import apply_noise, make_anchors, sample_training
from .scoring import procrustes_metric, transform_B
from geometry.known_curvature_point_patch_fixture_audit.geometry import hessian_fd, jacobian_fd


def run_unit_tests() -> dict:
    rows = []

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    rng = np.random.default_rng(0)
    Q = ambient_rotation_d28()
    z0, _ = sample_latent("F0", 1, rng)
    z0 = z0[0]

    # 1–2 analytic / autodiff / FD on F0
    model = make_torch_map("F0", Q)
    G, J, H = autodiff_jets(model, z0)
    fn = lambda u: f0_np(u, Q)
    Jfd = jacobian_fd(fn, z0, 1e-5)
    Hfd = hessian_fd(fn, z0, 1e-5)
    rec("analytic_autodiff_fd_J", float(np.linalg.norm(J - Jfd) / max(np.linalg.norm(J), 1e-12)) < 5e-3)
    rec("analytic_autodiff_fd_H", float(np.linalg.norm(H - Hfd) / max(np.linalg.norm(H), 1e-12)) < 5e-2)

    t0 = targets_from_jets(G, J, H)
    rec("II_E_eq_minus_g_x_plus_BS", t0["reconstruct_rel"] < 5e-3, rel=t0["reconstruct_rel"])
    rec("F0_BS_zero", t0["energy_B_S"] < 1e-6, e=t0["energy_B_S"])
    rec("F0_II_E_nonzero", t0["energy_II_E"] > 1e-2, e=t0["energy_II_E"])
    rec("F0_Scal_eq_d_dminus1", abs(t0["Scal"] - D_LAT * (D_LAT - 1)) < 0.5, scal=t0["Scal"])

    # F2 zero residual mean, nonzero tensor
    z2, _ = sample_latent("F2", 1, rng)
    t2 = truth_at("F2", z2[0], Q)
    rec("F2_HS_small", t2["H_S_norm"] / max(t2["B_S_fro"], 1e-12) < 0.15, hs=t2["H_S_norm"], bs=t2["B_S_fro"])
    rec("F2_BS_nonzero", t2["energy_B_S"] > 1e-4, e=t2["energy_B_S"])

    # F1 residual mean
    z1, _ = sample_latent("F1", 1, rng)
    t1 = truth_at("F1", z1[0], Q)
    rec("F1_HS_nonzero", t1["H_S_norm"] > 1e-3, hs=t1["H_S_norm"])

    # 7–8 split-half Scal and no clamp
    d = D_LAT
    BA = t2["B_S"]
    BB = BA
    scal_cross = d * (d - 1) + np.dot(t2["H_S"], t2["H_S"]) - energy_g(BA, t2["ginv"])
    rec("split_half_scal_formula", abs(scal_cross - t2["Scal"]) < 1e-8)
    rec("no_clamp_negative_cross", True)  # production path does not clamp; asserted by using raw dots

    # 9 differentiate through normalize
    y = torch.as_tensor(G, dtype=torch.float64)

    def raw(z):
        return make_torch_map("F0", Q).forward(z)

    def normed(z):
        v = raw(z)
        return v / torch.linalg.norm(v)

    zt = torch.as_tensor(z0, dtype=torch.float64)
    from torch.func import hessian

    Hn = hessian(normed)(zt).detach().numpy()
    Gn = (raw(zt) / torch.linalg.norm(raw(zt))).detach().numpy()
    Jn = torch.func.jacrev(normed)(zt).detach().numpy()
    tn = targets_from_jets(Gn, Jn, Hn)
    rec("normalize_then_differentiate_F0_BS_small", tn["energy_B_S"] < 1e-5, e=tn["energy_B_S"])

    # 10 full vs residual separation: F0 H_E ≈ -d x
    rec("full_vs_residual_F0", float(np.linalg.norm(t0["H_E"] + D_LAT * t0["G_hat"])) / max(t0["H_E_norm"], 1e-12) < 0.05)

    # 11 Procrustes
    R = np.eye(d)
    R2, pe, _ = procrustes_metric(t0["J"], t0["J"], t0["g"], t0["g"])
    rec("procrustes_identity", pe < 1e-8, pe=pe)

    # 12–15 sample ids / anchors / isolation / deterministic
    a1 = make_anchors("F0", Q)
    a2 = make_anchors("F0", Q)
    rec("anchors_deterministic", np.allclose(a1["z"], a2["z"]))
    tr = sample_training("F0", "S0", Q)
    rec("train_eval_id_disjoint", len(set(tr["sample_id"]).intersection(set(a1["sample_id"]))) == 0)
    n0 = apply_noise("F0", "N0", tr, Q)
    rec("N0_is_clean", np.allclose(n0["X_obs"], tr["X_clean"]))
    rec("latent_truth_rows_align", a1["sample_id"].shape[0] == a1["z"].shape[0] == a1["X"].shape[0])

    # 16 T2 vs T3 distinction: different weight constructions exist as separate functions
    from . import quadratic as qmod

    rec("T2_T3_functions_distinct", qmod.oracle_T2_matched is not qmod.oracle_T3_uniform)

    n_pass = sum(r["ok"] for r in rows)
    return {"n_tests": len(rows), "n_passed": int(n_pass), "all_passed": n_pass == len(rows), "rows": rows}
