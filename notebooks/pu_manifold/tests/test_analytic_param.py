"""
Fast synthetic-fixture tests for the ``pu_manifold.analytic_param`` module -- the separating
experiment (SC-2; D-12, D-13, D-14, D-15).

No HuggingFace access, no gitignored cache. Not collected by the core `effdim` test suite
(``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_analytic_param.py -q

Every test here pins a function against an input whose answer is known independently
(``scipy.integrate.quad``, ``sklearn.datasets.make_swiss_roll``'s own output,
``decoder_curvature.swiss_roll_analytic_H_vector``), never against "plausible" -- same
discipline as ``test_decoder_curvature.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.integrate import quad
from sklearn.datasets import make_swiss_roll

from pu_manifold import cae, chart_curvature, curvature_probe
from pu_manifold import decoder_curvature as dc
from pu_manifold import analytic_param as ap

FIXTURE_SEED = 20260807
"""Matches ``notebooks/diagnostics/decoder_substrate_screen_run.py``'s ``FIXTURE_SEED`` and
this phase's other fixed-seed artifacts, so this module's own numbers are directly comparable
to theirs."""


def _standard_fixture(n: int = 3000):
    """The standard Swiss roll fixture PLUS the raw ruling coordinate and centring vector
    ``curvature_probe.make_swiss_roll_fixture`` does not return -- recovered from a direct
    ``make_swiss_roll`` call at the identical ``n``/``seed``, per Task 1's ``<read_first>``
    note."""
    fixture = curvature_probe.make_swiss_roll_fixture(n=n, seed=FIXTURE_SEED)
    X_raw, t_raw = make_swiss_roll(n_samples=n, noise=0.0, random_state=FIXTURE_SEED)
    center = X_raw.mean(axis=0)
    assert np.max(np.abs(t_raw - fixture["t"])) == 0.0, (
        "direct make_swiss_roll call must reproduce the fixture's own t exactly at the same "
        "n/seed -- otherwise center/y_raw recovered here do not correspond to fixture['X']"
    )
    return fixture, X_raw, center


# --- Task 1: exact intrinsic-coordinate helpers -------------------------------------------


def test_swiss_roll_arc_length_matches_quad():
    points = [0.1, 1.0, 3.0, 7.0, 10.5]
    deviations = []
    for t in points:
        analytic = ap.swiss_roll_arc_length(np.array([t]))[0]
        numeric, _ = quad(lambda u: np.sqrt(1.0 + u * u), 0.0, t)
        deviations.append(abs(analytic - numeric))
    max_dev = max(deviations)
    print(f"swiss_roll_arc_length vs scipy.integrate.quad: max abs deviation = {max_dev:.6e}")
    assert max_dev < 1e-12


def test_swiss_roll_arc_length_strictly_increasing_over_fixture_range():
    fixture, _, _ = _standard_fixture(n=3000)
    t = fixture["t"]
    assert t.min() > 4.7  # sanity: matches the recorded fixture range (4.7147..14.1305)
    assert t.max() < 14.2
    t_sorted = np.sort(t)
    s = ap.swiss_roll_arc_length(t_sorted)
    assert np.all(np.diff(s) > 0.0)


def test_swiss_roll_intrinsic_plane_columns():
    fixture, _, _ = _standard_fixture(n=500)
    t = fixture["t"]
    y_scaled = fixture["X"][:, 1]
    global_std = fixture["global_std"]

    plane = ap.swiss_roll_intrinsic_plane(t, y_scaled, global_std)
    assert plane.shape == (500, 2)
    assert np.array_equal(plane[:, 1], y_scaled)  # bit-identical, never re-scaled
    expected_col0 = ap.swiss_roll_arc_length(t) / global_std
    assert np.max(np.abs(plane[:, 0] - expected_col0)) == 0.0


def test_swiss_roll_exact_ambient_reproduces_fixture():
    fixture, X_raw, center = _standard_fixture(n=1000)
    t = fixture["t"]
    y_raw = X_raw[:, 1]
    global_std = fixture["global_std"]

    ambient = ap.swiss_roll_exact_ambient(t, y_raw, center, global_std)
    max_dev = np.max(np.abs(ambient - fixture["X"]))
    print(f"swiss_roll_exact_ambient vs fixture['X']: max abs deviation = {max_dev:.6e}")
    assert max_dev < 1e-12


# --- Task 1: guard behaviour on the two new decoder classes -------------------------------


def test_assert_c2_decoder_on_analytic_swiss_roll_decoder():
    model = ap.AnalyticSwissRollDecoder(center=np.zeros(3), global_std=1.0)
    assert dc.assert_c2_decoder(model) == "no-activation-modules"


def test_assert_c2_decoder_on_analytic_param_decoder():
    model = ap.AnalyticParamDecoder(hidden=(8, 8), activation="silu")
    assert dc.assert_c2_decoder(model) == "silu"


# --- Task 1: the exactness floor -----------------------------------------------------------


def test_analytic_swiss_roll_decoder_reproduces_analytic_H_vector():
    """The exactness floor D-13 defines PASS as a measured degradation from: the tracer
    applied to a closed-form decoder, at the closed form's own spiral-parameter coordinates,
    reproduces the independent analytic formula row for row."""
    fixture, X_raw, center = _standard_fixture(n=800)
    t = fixture["t"]
    y_scaled = fixture["X"][:, 1]
    global_std = fixture["global_std"]

    model = ap.AnalyticSwissRollDecoder(center=center, global_std=global_std)
    z = torch.tensor(np.stack([t, y_scaled], axis=1), dtype=torch.float64)

    out = dc.plain_decoder_curvature(model, z)
    H_true = dc.swiss_roll_analytic_H_vector(t, global_std)

    max_dev = float((out["H_vec"].detach().cpu().numpy() - H_true).__abs__().max())
    print(f"AnalyticSwissRollDecoder floor vs swiss_roll_analytic_H_vector: "
          f"max abs deviation = {max_dev:.6e}")
    assert max_dev < 1e-8


# --- Task 1: the training protocol is reused unchanged -------------------------------------


def test_train_mlp_decoder_runs_and_reduces_loss_on_analytic_param_decoder():
    fixture, _, _ = _standard_fixture(n=300)
    t = fixture["t"]
    y_scaled = fixture["X"][:, 1]
    global_std = fixture["global_std"]

    z_train = torch.tensor(
        ap.swiss_roll_intrinsic_plane(t, y_scaled, global_std), dtype=torch.float32
    )
    x_train = torch.tensor(fixture["X"], dtype=torch.float32)

    torch.manual_seed(0)
    model = ap.AnalyticParamDecoder(hidden=(16, 16), activation="silu")
    cfg = dict(seed=0, lr=1e-3, weight_decay=1e-4, batch=64, max_epochs=10)
    fit = cae.train_mlp_decoder(model, z_train, x_train, cfg)

    losses = [h["total"] for h in fit["history"]]
    assert len(losses) == fit["epochs_run"] > 0
    assert losses[-1] < losses[0]


# --- Task 1: the end-to-end path -------------------------------------------------------------


def test_separator_end_to_end_floor_then_trained_net():
    """Standard fixture -> spiral-parameter and arc-length coordinate pairs ->
    AnalyticSwissRollDecoder through plain_decoder_curvature -> curvature_fidelity_report
    against swiss_roll_analytic_H_vector (the floor row) -> a small AnalyticParamDecoder
    trained a few epochs by cae.train_mlp_decoder -> through the SAME unchanged tracer -> the
    same four read-outs (the degraded row). No quality bar is asserted on the trained row --
    measuring what it costs is the experiment; pinning it would invent the threshold D-13
    forbids.
    """
    fixture, X_raw, center = _standard_fixture(n=500)
    t = fixture["t"]
    y_scaled = fixture["X"][:, 1]
    global_std = fixture["global_std"]

    # --- floor row: closed-form decoder at spiral-parameter coordinates ---
    floor_model = ap.AnalyticSwissRollDecoder(center=center, global_std=global_std)
    z_floor = torch.tensor(np.stack([t, y_scaled], axis=1), dtype=torch.float64)
    floor_out = dc.plain_decoder_curvature(floor_model, z_floor)
    H_true = dc.swiss_roll_analytic_H_vector(t, global_std)
    floor_report = chart_curvature.curvature_fidelity_report(floor_out["H_vec"], H_true)

    assert abs(floor_report["median_cosine_similarity"] - 1.0) < 1e-6

    # --- degraded row: a small net fit on the analytic parameterization ---
    z_intrinsic = ap.swiss_roll_intrinsic_plane(t, y_scaled, global_std)
    z_train = torch.tensor(z_intrinsic, dtype=torch.float32)
    x_train = torch.tensor(fixture["X"], dtype=torch.float32)

    torch.manual_seed(0)
    net = ap.AnalyticParamDecoder(hidden=(32, 32), activation="silu")
    cfg = dict(seed=0, lr=1e-3, weight_decay=1e-4, batch=64, max_epochs=30)
    cae.train_mlp_decoder(net, z_train, x_train, cfg)
    net = net.double()

    z_net = torch.tensor(z_intrinsic, dtype=torch.float64)
    net_out = dc.plain_decoder_curvature(net, z_net)
    net_report = chart_curvature.curvature_fidelity_report(net_out["H_vec"], H_true)

    for key in (
        "median_cosine_similarity",
        "median_magnitude_ratio",
        "magnitude_ratio_cv",
        "calibration_slope",
    ):
        assert np.isfinite(floor_report[key]), f"floor {key} is not finite: {floor_report[key]!r}"
        assert np.isfinite(net_report[key]), f"net {key} is not finite: {net_report[key]!r}"

    print(
        "end-to-end floor row: "
        f"median_cosine={floor_report['median_cosine_similarity']:.6f} "
        f"median_ratio={floor_report['median_magnitude_ratio']:.6f}"
    )
    print(
        "end-to-end net row:   "
        f"median_cosine={net_report['median_cosine_similarity']:.6f} "
        f"median_ratio={net_report['median_magnitude_ratio']:.6f}"
    )
