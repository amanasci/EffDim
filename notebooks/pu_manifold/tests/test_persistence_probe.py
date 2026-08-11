"""
Fast synthetic-fixture tests for the ``pu_manifold.persistence_probe`` module -- the
persistent-homology agreement instrument (SC-3; D-01, D-03, D-04, D-05, D-06).

No HuggingFace access, no gitignored cache. Not collected by the core `effdim` test suite
(``pyproject.toml``'s ``testpaths = ["tests"]`` excludes this directory) -- run explicitly:

    python -m pytest notebooks/pu_manifold/tests/test_persistence_probe.py -q

Every test here pins a function against an input whose answer is known independently (a
circle, a disc, an identity comparison, a zero-max-persistence reference, the standard
Swiss roll fixture's own measured structural inequality) or against an equivalent
reimplementation, never merely "plausible" -- same discipline as
``test_decoder_curvature.py`` and ``test_analytic_param.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
from ripser import ripser

from pu_manifold import analytic_param, cae, curvature_probe
from pu_manifold import persistence_probe as pp

FIXTURE_SEED = 20260807
"""Matches ``notebooks/diagnostics/decoder_substrate_screen_run.py``'s ``FIXTURE_SEED`` and
this phase's other fixed-seed artifacts, so this module's own numbers are directly comparable
to theirs."""


def _circle_fixture(n: int = 400, seed: int = 0) -> np.ndarray:
    """A seeded uniform sample on the unit circle, ``(n, 2)``. Known answer: ``beta_1 = 1``,
    one loop."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _disc_fixture(n: int = 400, seed: int = 1) -> np.ndarray:
    """A seeded uniform sample on the unit disc, ``(n, 2)``. Known answer: contractible,
    ``beta_1 = 0`` -- every H1 bar it carries is sampling noise."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


# --- Task 1: end-to-end tracer + known-answer discipline -----------------------------------


def test_persistence_diagram_returns_two_degrees_never_computes_h2():
    circle = _circle_fixture(n=200, seed=20260810)
    D, _ = pp.cloud_distance_matrix(circle, prescale=False)
    dgms = pp.persistence_diagram(D)
    assert len(dgms) == 2


def test_finite_pairs_removes_h0_single_infinite_death():
    circle = _circle_fixture(n=200, seed=20260810)
    D, _ = pp.cloud_distance_matrix(circle, prescale=False)

    raw_dgms = ripser(D, maxdim=pp.PH_MAXDIM, distance_matrix=True)["dgms"]
    raw_h0 = raw_dgms[0]
    n_infinite = int(np.sum(~np.isfinite(raw_h0[:, 1])))
    assert n_infinite == 1

    filtered_h0 = pp.finite_pairs(raw_h0)
    assert np.all(np.isfinite(filtered_h0[:, 1]))
    assert filtered_h0.shape[0] == raw_h0.shape[0] - 1


def test_bottleneck_self_is_exactly_zero():
    circle = _circle_fixture(n=200, seed=20260810)
    D, _ = pp.cloud_distance_matrix(circle, prescale=False)
    h1 = pp.persistence_diagram(D)[1]
    assert pp.ph_agreement(h1, h1)["bottleneck"] == 0.0


def test_wasserstein_self_is_near_zero():
    """Measured exactly ``0.0`` on a real 139-feature diagram during planning; it is an LP
    objective, so a tiny nonzero is acceptable and a large one is not."""
    fixture = curvature_probe.make_swiss_roll_fixture(n=300, seed=FIXTURE_SEED)
    D, _ = pp.cloud_distance_matrix(fixture["X"], prescale=False)
    h0 = pp.persistence_diagram(D)[0]
    agreement = pp.ph_agreement(h0, h0)
    assert agreement["wasserstein"] < 1e-9


def test_circle_vs_disc_known_answer_and_saturation():
    circle = _circle_fixture(n=400, seed=20260810)
    disc = _disc_fixture(n=400, seed=20260811)
    Dc, _ = pp.cloud_distance_matrix(circle, prescale=False)
    Dd, _ = pp.cloud_distance_matrix(disc, prescale=False)
    h1c = pp.persistence_diagram(Dc)[1]
    h1d = pp.persistence_diagram(Dd)[1]

    life_c = h1c[:, 1] - h1c[:, 0]
    life_d = h1d[:, 1] - h1d[:, 0] if h1d.shape[0] else np.zeros(0)
    own_largest_c = float(Dc.max())
    own_largest_d = float(Dd.max())

    assert int(np.sum(life_c > 0.20 * own_largest_c)) == 1
    assert int(np.sum(life_d > 0.20 * own_largest_d)) == 0

    # model=disc, ref=circle: bottleneck(circle, disc) must equal saturation_value(circle).
    agreement = pp.ph_agreement(h1d, h1c)
    assert abs(agreement["bottleneck"] - pp.saturation_value(h1c)) < 1e-6
    assert pp.is_saturated(agreement["bottleneck"], h1c) is True
    assert agreement["saturated"] is True


def test_max_persistence_empty_diagram_is_zero_not_nan_not_exception():
    assert pp.max_persistence(np.zeros((0, 2))) == 0.0


def test_ph_agreement_zero_max_persistence_returns_nan_never_inf():
    circle = _circle_fixture(n=200, seed=20260810)
    D, _ = pp.cloud_distance_matrix(circle, prescale=False)
    h1 = pp.persistence_diagram(D)[1]
    empty = np.zeros((0, 2))

    agreement = pp.ph_agreement(h1, empty)
    assert agreement["ref_max_persistence"] == 0.0
    assert np.isnan(agreement["bottleneck_norm"])
    assert np.isnan(agreement["wasserstein_norm"])
    assert not np.isinf(agreement["bottleneck_norm"])
    assert not np.isinf(agreement["wasserstein_norm"])
    assert np.isfinite(agreement["bottleneck"])
    assert np.isfinite(agreement["wasserstein"])


def test_standard_fixture_intrinsic_reference_thin_denominator_structural():
    """On the standard fixture's 600-point held-out subsample: the intrinsic-plane
    reference's own top H1 life is strictly greater than 0 and strictly less than 0.10 of
    its own cloud's largest pairwise distance, and its life-fraction is strictly less than
    the ambient reference's -- the structural inequality behind
    ``02.6-SCREENING-RULE-02.md``'s thin ``(H1, intrinsic)`` denominator travelling caveat.
    Asserts the STRUCTURAL relation, not a literal -- the exact figures are
    subsample-procedure-dependent (``02.6-RESEARCH.md`` Pitfall 2's own ``0.420`` does not
    reproduce under a different subsample of the same fixture)."""
    fixture = curvature_probe.make_swiss_roll_fixture(n=3000, seed=FIXTURE_SEED)
    idx = np.random.default_rng(FIXTURE_SEED).permutation(3000)[:600]

    intrinsic = analytic_param.swiss_roll_intrinsic_plane(
        fixture["t"][idx], fixture["X"][idx, 1], fixture["global_std"]
    )
    ambient = fixture["X"][idx]

    Di, _ = pp.cloud_distance_matrix(intrinsic, prescale=False)
    Da, _ = pp.cloud_distance_matrix(ambient, prescale=False)

    h1i = pp.persistence_diagram(Di)[1]
    h1a = pp.persistence_diagram(Da)[1]

    life_i = pp.max_persistence(h1i)
    life_a = pp.max_persistence(h1a)
    own_largest_i = float(Di.max())
    own_largest_a = float(Da.max())
    frac_i = life_i / own_largest_i
    frac_a = life_a / own_largest_a

    print(
        f"intrinsic top H1 life={life_i!r} own_largest_distance={own_largest_i!r} "
        f"fraction={frac_i!r}"
    )
    print(
        f"ambient top H1 life={life_a!r} own_largest_distance={own_largest_a!r} "
        f"fraction={frac_a!r}"
    )

    assert 0.0 < life_i < 0.10 * own_largest_i
    assert frac_i < frac_a


def test_readout_matrix_end_to_end_one_candidate_one_seed():
    """Task 1's tracer: the whole 16-cell slice at deliberately small scale. Standard-shaped
    fixture -> intrinsic and ambient references -> one ``PlainAutoEncoder`` trained a
    handful of epochs -> latent and decoder-image clouds, both references built on the SAME
    held-out row indices -> all 16 cells present and each either finite or ``nan`` with its
    zero denominator recorded beside it. Kept under ~10 seconds."""
    fixture = curvature_probe.make_swiss_roll_fixture(n=400, seed=FIXTURE_SEED)
    n = fixture["X"].shape[0]

    perm = np.random.default_rng(0).permutation(n)
    n_holdout = int(round(0.20 * n))
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    torch.manual_seed(0)
    model = cae.PlainAutoEncoder(3, 2, hidden=(32, 32), activation="silu")
    cfg = dict(seed=0, lr=3e-4, weight_decay=1e-4, batch=64, max_epochs=15)
    X_train = torch.tensor(fixture["X"][train_idx], dtype=torch.float32)
    cae.train_plain_ae(model, X_train, cfg)

    model.eval()
    X_holdout = torch.tensor(fixture["X"][holdout_idx], dtype=torch.float32)
    with torch.no_grad():
        z = model.encode(X_holdout).numpy()
        y = model.forward(X_holdout)["y"].numpy()

    intrinsic_ref = analytic_param.swiss_roll_intrinsic_plane(
        fixture["t"][holdout_idx], fixture["X"][holdout_idx, 1], fixture["global_std"]
    )
    ambient_ref = fixture["X"][holdout_idx]

    spaces = {"latent": z, "decoder_image": y}
    references = {"intrinsic": intrinsic_ref, "ambient": ambient_ref}

    result = pp.readout_matrix(spaces, references, prescale=True)

    assert set(pp.READOUT_CELLS) <= set(result.keys())

    for cell in pp.READOUT_CELLS:
        value = result[cell]
        assert not np.isinf(value), f"{cell} is inf"
        assert np.isfinite(value) or np.isnan(value)

        space, reference, hk, _ = cell.split("|")
        ref_key = f"{reference}|{hk}|ref_max_persistence"
        assert ref_key in result
        if np.isnan(value):
            assert result[ref_key] == 0.0

    for space in ("latent", "decoder_image"):
        for reference in ("intrinsic", "ambient"):
            for hk in ("H0", "H1"):
                sat_key = f"{space}|{reference}|{hk}|saturated"
                assert sat_key in result
                assert isinstance(result[sat_key], bool)

    for cloud_name in ("latent", "decoder_image", "intrinsic", "ambient"):
        assert f"{cloud_name}|applied_scale" in result

    forbidden_terms = ("score", "weighted", "combined", "average", "_sum", "ranked")
    for key in result:
        assert not any(term in key.lower() for term in forbidden_terms), key
