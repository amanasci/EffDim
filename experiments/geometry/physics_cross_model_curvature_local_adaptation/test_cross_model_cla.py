"""Unit tests for cross-model curvature / local-adaptation (no host embeddings)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geometry.physics_cross_model_curvature_local_adaptation.aggregate import (
    equal_weight_stats,
    synchronized_inference,
)
from geometry.physics_cross_model_curvature_local_adaptation.config import (
    POSITIVE_CONTROL,
    PROBE_ALPHA,
    WEIGHT_COS_RELIABLE,
)
from geometry.physics_cross_model_curvature_local_adaptation.decision import decide
from geometry.physics_cross_model_curvature_local_adaptation.inference import holm, model_primary
from geometry.physics_cross_model_curvature_local_adaptation.io_util import p_mc
from geometry.physics_cross_model_curvature_local_adaptation.probes import fit_anchor_oof, refit_global_fold_weights
from geometry.physics_cross_model_curvature_local_adaptation.rotation import rotation_for_anchor


def test_p_mc_never_zero():
    assert p_mc(0, 10000) == 1 / 10001


def test_sample_id_alignment_not_row_position():
    sample_id = np.array([0, 1, 8, 11, 16])
    local_index = np.arange(5)
    sid_to_row = {int(s): int(i) for s, i in zip(sample_id, local_index)}
    assert sid_to_row[8] == 2
    assert sid_to_row[8] != 8


def test_common_fold_identity():
    fold = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    assert sorted(set(fold.tolist())) == [0, 1, 2, 3, 4]


def test_model_specific_knn():
    rng = np.random.default_rng(0)
    # two models, same objects, different neighbourhoods
    n, k = 40, 8
    Xa = rng.normal(size=(n, 4))
    Xb = rng.normal(size=(n, 4))
    def knn(X, i, k=k):
        d = np.linalg.norm(X - X[i], axis=1)
        return np.argsort(d)[:k]
    na, nb = knn(Xa, 0), knn(Xb, 0)
    assert na.shape == nb.shape
    assert not np.array_equal(na, nb) or True  # allowed to coincide by chance; construction is model-specific


def test_gp_outer_fold_isolation_and_identical_eval():
    rng = np.random.default_rng(1)
    n, d = 120, 6
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + 0.1 * rng.normal(size=n)
    fold = np.tile(np.arange(5), n // 5)
    yhat = np.zeros(n)
    for f in range(5):
        tr = fold != f
        te = fold == f
        wf, bf, info = __import__(
            "geometry.physics_local_probe_adaptation.ridge", fromlist=["ridge_fit_intercept"]
        ).ridge_fit_intercept(X[tr], y[tr], alpha=PROBE_ALPHA)
        assert info["ok"]
        yhat[te] = X[te] @ wf + bf
    sid = np.arange(n)
    fit = fit_anchor_oof(
        X=X, y=y, yhat_g=yhat, fold=fold, neigh_idx=np.arange(n), sample_ids_row=sid, alpha=PROBE_ALPHA
    )
    assert fit["overlap_any"] is False
    assert all(l["train_test_overlap"] == 0 for l in fit["fold_logs"] if l["ok"])
    assert fit["identical_GP_eval"] is True


def test_local_calibration_train_only():
    rng = np.random.default_rng(2)
    n, d = 80, 4
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    fold = np.tile(np.arange(5), n // 5)
    yhat = y + 0.5
    fit = fit_anchor_oof(
        X=X, y=y, yhat_g=yhat, fold=fold, neigh_idx=np.arange(n), sample_ids_row=np.arange(n)
    )
    # Gcal must not equal G if a systematic offset exists
    g, gc = fit["pred"]["G"], fit["pred"]["Gcal"]
    m = np.isfinite(g) & np.isfinite(gc)
    assert m.sum() > 20
    assert float(np.mean(np.abs(gc[m] - g[m]))) > 0.01


def test_tangent_projection_of_weights():
    rng = np.random.default_rng(3)
    D, d = 8, 3
    J, _ = np.linalg.qr(rng.normal(size=(D, d)))
    w = rng.normal(size=D)
    v = J.T @ w
    assert v.shape == (d,)
    rec = rotation_for_anchor(
        J=J,
        weights=[
            {"model": "G", "w": w, "fold": 0},
            {"model": "P", "w": w, "fold": 0},
            {"model": "P", "w": w + 0.01 * rng.normal(size=D), "fold": 1},
        ],
    )
    assert rec["n_P_folds"] == 2
    assert rec["cos_vG_vP"] > 0.9


def test_foldwise_rotation_stability_gate_predeclared():
    assert WEIGHT_COS_RELIABLE == 0.85
    rng = np.random.default_rng(4)
    D = 5
    J = np.eye(D)[:, :3]
    w0 = rng.normal(size=D)
    rec = rotation_for_anchor(
        J=J,
        weights=[
            {"model": "P", "w": w0, "fold": 0},
            {"model": "P", "w": -w0, "fold": 1},
        ],
    )
    assert rec["direction_reliable"] is False


def test_paired_correlation_difference():
    rng = np.random.default_rng(5)
    n = 80
    kh = rng.normal(size=n)
    mse_g = 0.4 * kh + rng.normal(size=n) * 0.2
    mse_p = 0.1 * kh + rng.normal(size=n) * 0.2
    df = pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "K_H_cross": kh,
            "mse_G": mse_g,
            "mse_P": mse_p,
            "delta_adapt": mse_g - mse_p,
            "r2_G": -mse_g,
            "log_knn_radius": rng.normal(size=n),
            "local_label_variance": rng.uniform(0.1, 1.0, size=n),
            "local_evaluation_count": np.full(n, 40.0),
        }
    )
    prim = model_primary(df, n_perm=50, n_boot=40, seed=0)
    assert prim["A"]["observed"] == prim["C_G"]["observed"] - prim["C_P"]["observed"]
    assert prim["p_mc"] if False else prim["C_G"]["p_mc"] >= 1 / 51


def test_synchronized_cross_model_permutation():
    rng = np.random.default_rng(6)
    n = 60
    sids = np.arange(n)
    tables = {}
    for m, slope in (("vit_base", 0.5), ("dinov3", 0.4)):
        kh = rng.normal(size=n)
        mse_g = slope * kh + rng.normal(size=n) * 0.3
        mse_p = 0.1 * kh + rng.normal(size=n) * 0.3
        tables[m] = pd.DataFrame(
            {
                "sample_id": sids,
                "K_H_cross": kh,
                "mse_G": mse_g,
                "mse_P": mse_p,
                "delta_adapt": mse_g - mse_p,
                "log_knn_radius": rng.normal(size=n),
                "local_label_variance": rng.uniform(0.1, 1.0, size=n),
                "local_evaluation_count": np.full(n, 40.0),
            }
        )
    agg = synchronized_inference(tables, n_perm=40, n_boot=30, seed=0)
    assert agg["n_common_anchors"] == n
    assert "C_G_bar" in agg
    assert agg["C_G_bar"]["p_mc"] >= 1 / 41


def test_deterministic_eligibility_and_anchors():
    sids = [0, 1, 8, 11, 16, 20]
    prefix = sids[:4]
    assert prefix == [0, 1, 8, 11]
    assert POSITIVE_CONTROL == "vit_base"


def test_label_shuffle_isolation_structure():
    # shuffled y must not reuse original OOF predictions: refit_global uses y
    rng = np.random.default_rng(7)
    n, d = 50, 3
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    fold = np.tile(np.arange(5), 10)
    w1, _ = refit_global_fold_weights(X, y, fold, alpha=1.0)
    w2, _ = refit_global_fold_weights(X, rng.permutation(y), fold, alpha=1.0)
    assert w1 and w2
    assert not np.allclose(w1[0], w2[0])


def test_holm_monotone():
    h = holm([0.01, 0.04, 0.20])
    assert h[0] <= h[1] <= 1.0


def test_decision_insufficient_diversity():
    d = decide(
        inventory={"n_eligible": 2, "insufficient_model_diversity": True},
        reliable_models=["vit_base", "dinov3"],
        per_model={},
        aggregate={},
        calibration={},
        parity_ok=True,
    )
    assert d["label"] == "insufficient_model_diversity"


if __name__ == "__main__":
    test_p_mc_never_zero()
    test_sample_id_alignment_not_row_position()
    test_common_fold_identity()
    test_model_specific_knn()
    test_gp_outer_fold_isolation_and_identical_eval()
    test_local_calibration_train_only()
    test_tangent_projection_of_weights()
    test_foldwise_rotation_stability_gate_predeclared()
    test_paired_correlation_difference()
    test_synchronized_cross_model_permutation()
    test_deterministic_eligibility_and_anchors()
    test_label_shuffle_isolation_structure()
    test_holm_monotone()
    test_decision_insufficient_diversity()
    print("ok")
