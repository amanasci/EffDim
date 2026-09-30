"""Tensor fidelity: full probe-facing tensors vs exact truth on the in-sphere fixture."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), RUNNERS / name)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


tf = _load("10_tensor_fidelity_run.py")


def _small_generator(seed=20260905):
    c = tf.adj.SMOKE
    return tf.adj.InSphereGenerator(c["d"], c["D"], c["a"], c["bump_widths"], c["bump_amps"], seed=seed)


class _Reparam(torch.nn.Module):
    """G'(z') = G(A z'): the same manifold in another chart."""
    def __init__(self, G, A):
        super().__init__(); self.G = G; self.A = torch.as_tensor(A, dtype=torch.float64)
        self.d, self.D = G.d, G.D
    def decode(self, z):
        return self.G.decode(z @ self.A.T)


def test_truth_exactness():
    G = _small_generator()
    z = tf.adj.draw_latents(8, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=1)
    geo = tf.truth_geometry(G, z)
    assert geo["max_abs_H_rad_plus_d"] < 1e-8


def test_ambient_lift_is_chart_invariant():
    G = _small_generator()
    rng = np.random.default_rng(3)
    A = rng.standard_normal((G.d, G.d)) + 2 * np.eye(G.d)
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=2)
    zp = np.linalg.solve(A, z.T).T                       # A z' = z
    geo = tf.truth_geometry(G, z)
    geo_p = tf.truth_geometry(_Reparam(G, A), zp)
    w = rng.standard_normal(G.D)
    T = tf.probe_facing_tensors(geo, w)["pf_full"]
    Tp = tf.probe_facing_tensors(geo_p, w)["pf_full"]
    np.testing.assert_allclose(tf.tensor_cosine(T, geo, Tp, geo_p), 1.0, atol=1e-10)
    # the difference norm comes from |a|^2 + |b|^2 - 2<a,b>; its floor is ~sqrt(machine eps)
    np.testing.assert_allclose(tf.relative_error(Tp, geo_p, T, geo), 0.0, atol=1e-6)


def test_probe_facing_tensors_match_split_columns_norms():
    G = _small_generator()
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=4)
    geo = tf.truth_geometry(G, z)
    w = np.random.default_rng(5).standard_normal(G.D)
    pt = tf.probe_facing_tensors(geo, w)
    hess = np.zeros_like(pt["pf_full"])
    cols = tf.split_columns(geo, w, 0.0, hess, pt["pf_full"], G.d)["cols"]
    np.testing.assert_allclose(tf.ppf.metric_norms(pt["pf_full"], geo["ginv"])["fro"], cols["pf_full"], rtol=1e-12)
    np.testing.assert_allclose(tf.ppf.metric_norms(pt["pf_tan"], geo["ginv"])["fro"], cols["pf_tan"], rtol=1e-12)


def test_scoring_self_and_negative():
    G = _small_generator()
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=6)
    geo = tf.truth_geometry(G, z)
    T = tf.probe_facing_tensors(geo, np.random.default_rng(7).standard_normal(G.D))["pf_full"]
    np.testing.assert_allclose(tf.tensor_cosine(T, geo, T, geo), 1.0, atol=1e-12)
    np.testing.assert_allclose(tf.tensor_cosine(-T, geo, T, geo), -1.0, atol=1e-12)
    np.testing.assert_allclose(tf.relative_error(T, geo, T, geo), 0.0, atol=1e-12)


def _four(T):
    return {"pf_full": T, "pf_tan": T, "hess_y": T, "mismatch": T}


def test_score_skips_nan_anchors():
    G = _small_generator()
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=8)
    geo = tf.truth_geometry(G, z)
    T = tf.probe_facing_tensors(geo, np.random.default_rng(9).standard_normal(G.D))["pf_full"]
    T_est = T.copy(); T_est[2] = np.nan
    s = tf.score_tensors(_four(T_est), _four(T), geo, geo)
    assert s["n_valid_pf_full"] == 5
    assert s["cos_pf_full_p50"] == pytest.approx(1.0, abs=1e-12)


def test_mismatch_scoring_near_zero_truth():
    G = _small_generator()
    z = tf.adj.draw_latents(6, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=10)
    geo = tf.truth_geometry(G, z)
    H = tf.probe_facing_tensors(geo, np.random.default_rng(11).standard_normal(G.D))["pf_full"]
    est = _four(H); truth = _four(H)
    truth["mismatch"] = np.zeros_like(H)                     # true mismatch exactly zero
    est["mismatch"] = 1e-3 * H                               # small estimated mismatch
    s = tf.score_tensors(est, truth, geo, geo)
    assert s["n_valid_mismatch"] == 0 and np.isnan(s["cos_mismatch_p50"])
    assert s["relerr_mismatch_p50"] == pytest.approx(1e-3, rel=1e-9)   # normalised by |hess_y_true|
