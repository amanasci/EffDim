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


def test_labels_names_and_scaling():
    G = _small_generator()
    z = tf.adj.draw_latents(2000, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=12)
    lab = tf.make_labels(G, z)
    assert list(lab["y"]) == ["lin", "nonlin", "lam0", "lam0.5", "lam1", "lam2"]
    assert np.std(lab["y"]["lam0"]) == pytest.approx(1.0, rel=1e-9)     # ambient term at unit std
    for name, y in lab["y"].items():
        assert y.shape == (2000,) and np.all(np.isfinite(y)), name


def test_covariant_hessian_identity_at_lambda_zero():
    """At lambda = 0 the label is <w0, G(z)>/s: its covariant Hessian is <w0, II>/s = <w0_N, II>/s."""
    G = _small_generator()
    z = tf.adj.draw_latents(500, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=13)
    lab = tf.make_labels(G, z)
    za = z[:8]
    geo = tf.truth_geometry(G, za)
    got = tf.covariant_hessian(lab["f"]["lam0"], geo, za)
    want = tf.probe_facing_tensors(geo, lab["w0"])["pf_full"] / lab["scales"]["ambient"]
    np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-8)


def test_label_functions_match_values():
    G = _small_generator()
    z = tf.adj.draw_latents(300, G.d, (0.4, 0.6, 0.9), (0.2, 0.5, 0.3), seed=14)
    lab = tf.make_labels(G, z)
    zt = torch.as_tensor(z[:5], dtype=torch.float64)
    for name, f in lab["f"].items():
        with torch.no_grad():
            np.testing.assert_allclose(f(zt).numpy(), lab["y"][name][:5], rtol=1e-12, err_msg=name)


import json


def test_config_for():
    c = tf.config_for("small", 16000)
    assert (c["d"], c["D"], c["n"], c["k"], c["n_anchors"]) == (4, 64, 16000, 500, 64)
    assert tf.config_for("small", None)["k"] == 128
    p = tf.config_for("full", None)
    assert (p["d"], p["D"], p["n"], p["k"], p["n_anchors"]) == (16, 768, 86471, 2048, 512)


def test_end_to_end_smoke(tmp_path):
    rec = tmp_path / "10_tensor_fidelity.jsonl"
    rows = tf.run_config(tf.config_for("small", 4000), 0.0, 0, 3, "cpu", "small", rec)
    assert [r["label"] for r in rows] == ["lin", "nonlin", "lam0", "lam0.5", "lam1", "lam2"]
    on_disk = [json.loads(l) for l in rec.read_text().splitlines()]
    assert [r["label"] for r in on_disk] == [r["label"] for r in rows]
    for r in rows:
        for t in ("pf_full", "pf_tan", "hess_y"):
            assert np.isfinite(r[f"cos_{t}_p50"]) and -1.0 <= r[f"cos_{t}_p50"] <= 1.0, (r["label"], t)
        assert np.isfinite(r["rho_mismatch"]) and np.isfinite(r["rho_align"])
        assert r["max_abs_H_rad_plus_d"] < 1e-8


def test_refuses_production_record(tmp_path, monkeypatch):
    bad = tmp_path / f"{tf.ppf.PRODUCTION_STEMS[0]}_x.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "small", "--threads", "8", "--record-path", str(bad)])
    with pytest.raises(SystemExit, match="refusing"):
        tf.main()
    assert not bad.exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU refusal")
def test_cuda_without_gpu_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "small", "--threads", "8", "--device", "cuda",
                                      "--record-path", str(tmp_path / "r.jsonl")])
    with pytest.raises(SystemExit, match="CUDA is not available"):
        tf.main()


rep = _load("10_tensor_fidelity_report.py")


def _row(n, noise, seed, label, cos, rho, mode="small"):
    r = {"experiment": "tensor-fidelity", "row": "result", "mode": mode, "n": n, "noise_frac": noise, "seed": seed,
         "label": label, "rho_mismatch": rho, "rho_align": rho, "var_explained": 0.99}
    for t in tf.TENSORS:
        r.update({f"cos_{t}_p50": cos, f"cos_{t}_p25": cos, f"relerr_{t}_p50": 0.1, f"n_valid_{t}": 64})
    return r


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


LABELS = ["lin", "nonlin", "lam0", "lam0.5", "lam1", "lam2"]


def test_report_dedupes_reruns(tmp_path):
    p = tmp_path / "r.jsonl"
    _write(p, [{"row": "environment"}, _row(4000, 0.0, 0, "lin", 0.1, 0.1), _row(4000, 0.0, 0, "lin", 0.9, 0.9)])
    rows = rep.load_rows([p])
    assert len(rows) == 1 and rows[0]["cos_pf_full_p50"] == 0.9


def test_pass_lines_use_largest_n_and_skip_lam0_rho(tmp_path):
    rows = [_row(4000, 0.0, s, lab, 0.1, 0.1) for s in range(3) for lab in LABELS]            # small n fails
    rows += [_row(64000, 0.0, s, lab, 0.9, (0.0 if lab == "lam0" else 0.8)) for s in range(3) for lab in LABELS]
    rows += [_row(64000, 0.5, s, lab, 0.1, 0.1) for s in range(3) for lab in LABELS]           # noisy ignored
    pl = rep.pass_lines(rows)
    assert pl["n"] == 64000 and pl["pass"] is True
    assert pl["per_label"]["lam0"]["rho"] is None


def test_pass_lines_fail(tmp_path):
    rows = [_row(64000, 0.0, s, lab, 0.9, 0.5) for s in range(3) for lab in LABELS]
    assert rep.pass_lines(rows)["pass"] is False


def test_write_report(tmp_path):
    rows = [_row(n, nz, s, lab, 0.9, 0.8) for n in (4000, 16000) for nz in (0.0, 0.25) for s in range(2) for lab in LABELS]
    rows += [_row(86471, nz, 0, lab, 0.85, 0.75, mode="full") for nz in (0.0, 0.25) for lab in LABELS]
    rep.write_report(rows, tmp_path)
    text = (tmp_path / "REPORT.md").read_text()
    assert text.startswith("# Tensor fidelity") and "PASS" in text and "## Paper scale" in text
    assert (tmp_path / "fig_tensor_fidelity.png").stat().st_size > 0
