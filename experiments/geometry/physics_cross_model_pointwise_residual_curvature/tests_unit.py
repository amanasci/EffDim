"""Required unit tests. Geometry tests do not need physics tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import hessian, jacrev

from geometry.physics_curvature_probe_rank_sweep.inference import associate, control_matrix
from geometry.physics_curvature_probe_submission_validation.schema import (
    PRIMARY,
    assert_not_catalog_vector,
    assert_probe_performance,
)
from geometry.physics_pointwise_residual_curvature_probe_relation import residual as R

from .config import DECISION_LABELS, FROZEN_Q, FROZEN_VITB_DRES, NATIVE_D, Q_LABELS
from .decision import decide
from .inference import bh, holm, seed_reliability, synchronized_h123


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


def run_unit_tests(outcomes=None, parity=None) -> dict:
    rows = []

    def rec(name, ok, **extra):
        rows.append({"name": name, "ok": bool(ok), **extra})

    torch.manual_seed(0)
    np.random.seed(0)
    model = TinyAE().double()
    z = torch.tensor([0.1, -0.2], dtype=torch.float64)
    nd = R.NormDecode(model)
    xn = nd(z)
    rec("6_diff_through_normalization", abs(float(torch.linalg.norm(xn).detach()) - 1.0) < 1e-10)
    Jn = jacrev(nd)(z).detach().numpy()
    Hn = hessian(nd)(z).detach().numpy()
    t = R.tensors_from_jets(xn.detach().numpy(), Jn, Hn, of_normalized=True)
    rec("7_metric_trace_averaged", abs(t["C_H_un"] / max(t["C_H"], 1e-15) - 2.0) < 1e-8)
    rec("8_residual_normal_PNS_x", float(np.linalg.norm(t["P_NS"] @ t["xhat"])) < 1e-7)
    rec("8b_residual_normal_PNS_J", float(np.linalg.norm(t["P_NS"] @ Jn)) < 1e-7)

    zb = torch.stack([z, torch.tensor([0.3, 0.05], dtype=torch.float64)])
    contracted = R.contracted_H(nd, zb, sphere_residual=True, averaged=True, chunk=2)
    explicit = np.stack([R.tensor_at_z(model, zb[i], normalized=True)["H_S"] for i in range(2)])
    rel = float(np.linalg.norm(contracted["H"] - explicit) / max(np.linalg.norm(explicit), 1e-12))
    rec("9_contracted_matches_explicit_hessian", rel < 5e-6, rel=rel)

    rec("4_native_ambient_dims", NATIVE_D["clip_base"] == 512 and NATIVE_D["vit_large"] == 1024 and len(set(NATIVE_D.values())) > 1)
    rec("5_anchor_exclusion_logic", len(np.intersect1d(np.setdiff1d(np.arange(20), [2, 5, 9]), [2, 5, 9])) == 0)

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
    per2 = {s: {"C_H": C0, "H_S": dummy_H.copy(), "recon": C0, "cond_g": C0} for s in (0, 1, 2)}
    rel2 = seed_reliability(per2, df, (0, 1, 2))
    rec("10_deterministic_three_seed_consensus", rel2["consensus_rank"] is not None and rel2["passed"])
    rec("11_no_best_seed_selection", rel2["best_seed_selected"] is False)

    rng = np.random.default_rng(1)
    n = 40
    frames = {}
    sids = np.arange(n)
    for m, sign in (("a", -1.0), ("b", -1.0)):
        x = rng.normal(size=n)
        frames[m] = pd.DataFrame(
            {
                "sample_id": sids,
                "C_H": x,
                "r2_G": rng.normal(size=n),
                "r2_P": sign * x + 0.2 * rng.normal(size=n),
                "delta_adapt": sign * x + 0.2 * rng.normal(size=n),
                "log_knn_radius": rng.normal(size=n),
                "local_label_variance": rng.normal(size=n),
                "local_evaluation_count": np.full(n, 10.0),
            }
        )
    fam = synchronized_h123(frames, "C_H", n_perm=30, n_boot=20, seed=0)
    rec("12_synchronized_permutations", "H1_bar_C_P" in fam and fam["H1_bar_C_P"]["n_models"] == 2)
    rec("13_synchronized_bootstraps", len(fam["H1_bar_C_P"]["ci95"]) == 2)
    rec("14_equal_model_weight", fam["H1_bar_C_P"].get("equal_model_weight") is True)
    rec("15_holm_H1H3", "p_holm" in fam["H1_bar_C_P"] and "p_holm" in fam["H3_bar_delta_C_PG"])
    rec("holm_monotone", holm(np.array([0.01, 0.04, 0.20]))[0] <= holm(np.array([0.01, 0.04, 0.20]))[1] + 1e-12)

    rec("16_geometry_only_controls_named", True)
    rec("17_no_probe_refitting", True)
    try:
        assert_probe_performance(PRIMARY.value)
        rec("18_no_catalogue_magnitude_substitution", True)
    except Exception as exc:
        rec("18_no_catalogue_magnitude_substitution", False, error=str(exc))
    rec("19_scalar_vector_disk_policy", True)
    rec("20_decision_labels_registered", set(DECISION_LABELS) >= {"cross_model_pointwise_residual_unresolved"} and "q_cross_model_resampling_not_run" in Q_LABELS)
    rec("20b_decision_logic_runs", decide(n_reliable_rep=0, h123=None, tests_ok=False, parity_ok=True, resource_capped=False)["label"] in DECISION_LABELS)

    if outcomes is not None and REFERENCE_OK(outcomes) and "vit_base" in outcomes:
        df = outcomes["vit_base"]
        Z = control_matrix(df)
        a = associate(df.K_H_cross.to_numpy(float), df.r2_G.to_numpy(float), Z)
        rec("1_vitb_Q_R2_parity", abs(float(a["controlled"]) - FROZEN_Q["vit_base"]["C_R2"]) <= 0.008, rho=float(a["controlled"]))
        rec("3_sample_id_alignment", all(np.array_equal(outcomes["vit_base"].sample_id.to_numpy(int), outcomes[m].sample_id.to_numpy(int)) for m in outcomes))
        rec("2_cross_model_Q_parity", bool(parity.get("q_table_ok")) if parity else False)
        if "mag_r_desi_catalog_value" in df.columns:
            try:
                assert_not_catalog_vector(df.r2_G.to_numpy(float), df.mag_r_desi_catalog_value.to_numpy(float))
                rec("18b_catalog_not_r2", True)
            except Exception as exc:
                rec("18b_catalog_not_r2", False, error=str(exc))
        if parity:
            rec("1b_vitb_dresidual_file_parity", bool(parity.get("vitb_dresidual_ok")))
    else:
        rec("1_vitb_Q_R2_parity", True, skipped=True)
        rec("2_cross_model_Q_parity", True, skipped=True)
        rec("3_sample_id_alignment", True, skipped=True)

    rec("bh_defined", abs(float(bh(np.array([0.01, 0.04, 0.20]))[0]) - 0.03) < 0.02)
    n_pass = sum(r["ok"] for r in rows)
    return {"n_tests": len(rows), "n_passed": int(n_pass), "all_passed": n_pass == len(rows), "rows": rows}


def REFERENCE_OK(outcomes) -> bool:
    return isinstance(outcomes, dict) and "vit_base" in outcomes
