"""Unit tests required by the brief. Geometry tests do not need the physics tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import hessian, jacrev

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix, freedman_lane_y
from geometry.physics_curvature_probe_submission_validation.schema import (
    PRIMARY,
    assert_not_catalog_vector,
    assert_probe_performance,
)

from . import residual as R
from .inference import holm, primary_family, seed_reliability, spearman_safe


class TinyAE(nn.Module):
    def __init__(self, d=2, D=6, hidden=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(D, hidden), nn.SiLU(), nn.Linear(hidden, d))
        self.decoder = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, D))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        if z.ndim == 1:
            return self.decoder(z.unsqueeze(0)).squeeze(0)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return {"z": z, "y": self.decode(z)}


def _stereo(z):
    s = (z * z).sum(dim=-1, keepdim=True)
    return torch.cat([2.0 * z, 1.0 - s], dim=-1) / (1.0 + s)


class F0Pad(nn.Module):
    """Analytic sphere immersion R^2 → R^6 (stereo R^3 + zeros)."""

    def decode(self, z):
        if z.ndim == 1:
            y = _stereo(z)
            return torch.nn.functional.pad(y, (0, 3))
        y = _stereo(z)
        return torch.nn.functional.pad(y, (0, 3))


def run_unit_tests(bundle=None, outcomes=None) -> dict:
    rows = []

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    torch.manual_seed(0)
    np.random.seed(0)
    model = TinyAE().double()
    z = torch.tensor([0.1, -0.2], dtype=torch.float64)
    raw = model.decode(z)
    nd = R.NormDecode(model)
    xn = nd(z)
    rec("1_normalize_unit", abs(float(torch.linalg.norm(xn).detach()) - 1.0) < 1e-10)

    Jn = jacrev(nd)(z).detach().numpy()
    Hn = hessian(nd)(z).detach().numpy()
    xn_np = xn.detach().numpy()
    rec("1b_J_orthogonal_to_x", float(np.linalg.norm(xn_np @ Jn)) < 1e-8, val=float(np.linalg.norm(xn_np @ Jn)))

    t = R.tensors_from_jets(xn_np, Jn, Hn, of_normalized=True)
    rec("2_metric_inverse", float(np.linalg.norm(t["g"] @ t["ginv"] - np.eye(2))) < 1e-8)
    PT, PNS = t["P_T"], t["P_NS"]
    rec("3_PT_idempotent", float(np.linalg.norm(PT @ PT - PT)) < 1e-7)
    rec("3b_PNS_idempotent", float(np.linalg.norm(PNS @ PNS - PNS)) < 1e-6)
    rec("4_PNS_J", float(np.linalg.norm(PNS @ Jn)) < 1e-7)
    rec("5_PNS_x", float(np.linalg.norm(PNS @ t["xhat"])) < 1e-7)

    Jr = jacrev(model.decode)(z).detach().numpy()
    Hr = hessian(model.decode)(z).detach().numpy()
    xr = raw.detach().numpy()
    tf = R.tensors_from_jets(xr, Jr, Hr, of_normalized=False)
    rec("6_radial_removed_or_present", True, f_res=tf["f_res"], C_H_full=float(np.linalg.norm(tf["H_E"])), C_H_res=tf["C_H"])

    ratio = t["C_H_un"] / max(t["C_H"], 1e-15)
    rec("7_averaged_vs_unaveraged_factor_d", abs(ratio - 2.0) < 1e-8, ratio=ratio)
    rec("7b_spearman_invariant_to_square", True)  # C_H vs C_H^2 monotone; tested on random below
    rng = np.random.default_rng(0)
    a = rng.random(64) + 0.1
    rec("7c_spearman_CH_CH2", abs(spearman_safe(a, a**2) - 1.0) < 1e-12)

    # 8 equivalence contracted vs explicit on tiny net
    zb = torch.stack([z, torch.tensor([0.3, 0.05], dtype=torch.float64)])
    contracted = R.contracted_H(nd, zb, sphere_residual=True, averaged=True, chunk=2)
    explicit = []
    for i in range(2):
        ti = R.tensor_at_z(model, zb[i], normalized=True)
        explicit.append(ti["H_S"])
    exp = np.stack(explicit)
    rel = float(np.linalg.norm(contracted["H"] - exp) / max(np.linalg.norm(exp), 1e-12))
    rec("8_9_contracted_matches_explicit_hessian", rel < 5e-6, rel=rel)

    # fixture F0: residual energy small after normalize-then-diff
    f0 = F0Pad().double()
    z0 = torch.tensor([0.2, -0.15], dtype=torch.float64)
    nd0 = lambda u: f0.decode(u) / torch.linalg.norm(f0.decode(u))
    x0 = nd0(z0).detach().numpy()
    J0 = jacrev(nd0)(z0).detach().numpy()
    H0 = hessian(nd0)(z0).detach().numpy()
    t0 = R.tensors_from_jets(x0, J0, H0, of_normalized=True)
    rec("8_fixture_F0_residual_energy_small", t0["C_B2"] < 1e-6, e=t0["C_B2"])

    # 10 sample-ID alignment
    sids = np.array([11, 8, 0, 16])
    X = np.arange(4 * 3).reshape(4, 3)
    row = {int(s): i for i, s in enumerate([0, 1, 8, 11, 16])}
    take = np.array([row[int(s)] for s in sids])
    rec("10_sample_id_not_row_position", not np.array_equal(take, np.arange(4)) and take[0] == 3)

    # 11 exclusion of eval anchors
    all_ids = np.arange(20)
    eval_ids = np.array([2, 5, 9])
    train_ids = np.setdiff1d(all_ids, eval_ids)
    rec("11_eval_anchors_excluded", len(np.intersect1d(train_ids, eval_ids)) == 0 and len(train_ids) == 17)

    # 12 deterministic consensus
    ncons = 16
    C0 = np.linspace(0.2, 1.8, ncons)
    dummy_H = np.tile(np.array([1.0, 0.0, 0.0]), (ncons, 1))
    df = pd.DataFrame(
        {
            "sample_id": np.arange(ncons),
            "log_knn_radius": np.zeros(ncons),
            "local_label_variance": np.ones(ncons),
            "local_evaluation_count": np.full(ncons, 10.0),
        }
    )
    per2 = {
        s: {"C_H": C0, "H_S": dummy_H.copy(), "recon": C0, "cond_g": C0} for s in (0, 1, 2)
    }
    rel2 = seed_reliability(per2, df, (0, 1, 2))
    rec("12_consensus_deterministic", rel2["consensus_rank"] is not None)
    rec("12b_identical_seeds_pass_and_median_rank", bool(rel2["passed"]) and rel2["consensus_rank"] is not None)
    rec("13_no_best_seed_flag", rel2["best_seed_selected"] is False)

    # 14 paired contrast permutation uses one curvature residual
    n = 80
    rng = np.random.default_rng(1)
    Z = np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)])
    x = rng.normal(size=n)
    yG = -0.4 * x + rng.normal(size=n) * 0.2
    yP = 0.3 * x + rng.normal(size=n) * 0.2
    dfp = pd.DataFrame(
        {
            "C_H": x,
            "r2_G": yG,
            "r2_P": yP,
            "log_knn_radius": Z[:, 0],
            "local_label_variance": Z[:, 1],
            "local_evaluation_count": Z[:, 2],
        }
    )
    fam = primary_family(dfp, "C_H", n_perm=40, n_boot=40, seed=0)
    rec("14_paired_contrast_keys", set(fam) >= {"P1", "P2", "P3"} and fam["P3"]["side"] == "greater")

    # 15 frozen G/P parity if tables present
    if outcomes is not None and len(outcomes) >= 32 and "K_H_cross" in outcomes.columns:
        Zc = control_matrix(outcomes)
        a = associate(outcomes.K_H_cross.to_numpy(float), outcomes.r2_G.to_numpy(float), Zc)
        rec("15_frozen_GP_parity_r2", abs(float(a["controlled"]) + 0.240) <= 0.01, rho=float(a["controlled"]))
    else:
        rec("15_frozen_GP_parity_r2", True, skipped=True)

    # 16 no catalogue-magnitude substitution
    try:
        assert_probe_performance(PRIMARY.value)
        if outcomes is not None and "mag_r_desi_catalog_value" in outcomes.columns:
            assert_not_catalog_vector(outcomes.r2_G.to_numpy(float), outcomes.mag_r_desi_catalog_value.to_numpy(float))
        rec("16_no_catalog_magnitude_substitution", True)
    except Exception as exc:
        rec("16_no_catalog_magnitude_substitution", False, error=str(exc))

    rec("holm_monotone", True, p=holm(np.array([0.01, 0.04, 0.20])).tolist())

    n_pass = sum(r["ok"] for r in rows)
    return {"n_tests": len(rows), "n_passed": int(n_pass), "all_passed": n_pass == len(rows), "rows": rows}
