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
import persim
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
    subsample-procedure-dependent (``02.6-RESEARCH.md`` Pitfall 2's own measured figure for
    this cell does not reproduce under a different subsample of the same fixture)."""
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


@pytest.fixture(scope="module")
def standard_references():
    """The standard fixture's 600-point held-out subsample, shared by every Task 2 guard
    test below: ``curvature_probe.make_swiss_roll_fixture(n=3000, seed=20260807)``, then
    ``np.random.default_rng(20260807).permutation(3000)[:600]``. Computed once per
    test-file run so the four guards stay well under 90 seconds together. Returns the
    intrinsic-plane and ambient H1 diagrams plus each cloud's own largest pairwise
    distance (for the printed record), and both references' full distance matrices for
    the pre-scaling guard."""
    fixture = curvature_probe.make_swiss_roll_fixture(n=3000, seed=FIXTURE_SEED)
    idx = np.random.default_rng(FIXTURE_SEED).permutation(3000)[:600]

    intrinsic = analytic_param.swiss_roll_intrinsic_plane(
        fixture["t"][idx], fixture["X"][idx, 1], fixture["global_std"]
    )
    ambient = fixture["X"][idx]

    Di, _ = pp.cloud_distance_matrix(intrinsic, prescale=False)
    Da, _ = pp.cloud_distance_matrix(ambient, prescale=False)

    h1_intrinsic = pp.persistence_diagram(Di)[1]
    h1_ambient = pp.persistence_diagram(Da)[1]

    return {
        "ambient_cloud": ambient,
        "h1_intrinsic": h1_intrinsic,
        "h1_ambient": h1_ambient,
        "own_largest_intrinsic": float(Di.max()),
        "own_largest_ambient": float(Da.max()),
    }


# --- Task 2: the two travelling caveats, pinned at the exact place they fire ---------------


def test_saturation_between_the_two_references_themselves(standard_references):
    """The important new evidence: the phase's most-watched cell,
    ``latent|ambient|H1|bottleneck_norm`` (and its ``decoder_image`` twin), is expected to
    be SATURATED for a candidate whose latent has correctly unrolled the roll, because the
    two references are already saturated against each other before any candidate diagram
    enters the comparison. Standard fixture, 600-point held-out subsample (recipe above).
    Structural relations pinned, not literals; measured figures printed and transcribed
    into the plan SUMMARY."""
    h1_intrinsic = standard_references["h1_intrinsic"]
    h1_ambient = standard_references["h1_ambient"]

    # model=intrinsic (stand-in for a correctly-unrolled candidate diagram), ref=ambient --
    # exactly the (space, reference=ambient, H1, bottleneck) cell's own roles.
    agreement = pp.ph_agreement(h1_intrinsic, h1_ambient)
    measured_bottleneck = agreement["bottleneck"]
    measured_saturation = pp.saturation_value(h1_ambient)

    empty = np.zeros((0, 2))
    measured_empty_vs_ambient = pp.ph_agreement(empty, h1_ambient)["bottleneck"]

    print(
        f"bottleneck(ambient_H1, intrinsic_H1)={measured_bottleneck!r} "
        f"saturation_value(ambient_H1)={measured_saturation!r} "
        f"bottleneck(empty, ambient_H1)={measured_empty_vs_ambient!r}"
    )
    print(
        f"ambient top H1 life={pp.max_persistence(h1_ambient)!r} "
        f"own_largest_distance={standard_references['own_largest_ambient']!r}"
    )
    print(
        f"intrinsic top H1 life={pp.max_persistence(h1_intrinsic)!r} "
        f"own_largest_distance={standard_references['own_largest_intrinsic']!r}"
    )

    assert abs(measured_bottleneck - measured_saturation) < 1e-6
    assert abs(measured_bottleneck - measured_empty_vs_ambient) < 1e-6
    assert pp.is_saturated(measured_bottleneck, h1_ambient) is True
    assert agreement["saturated"] is True


def test_empty_versus_nonempty_bottleneck_equals_saturation_and_stays_finite(
    standard_references,
):
    h1_ambient = standard_references["h1_ambient"]
    empty = np.zeros((0, 2))

    raw_bottleneck = float(persim.bottleneck(empty, h1_ambient))
    assert abs(raw_bottleneck - pp.saturation_value(h1_ambient)) < 1e-9

    agreement = pp.ph_agreement(empty, h1_ambient)
    assert np.isfinite(agreement["bottleneck"])
    assert np.isfinite(agreement["wasserstein"])
    assert agreement["ref_max_persistence"] > 0.0
    assert np.isfinite(agreement["bottleneck_norm"])
    assert np.isfinite(agreement["wasserstein_norm"])
    assert agreement["saturated"] is True


def test_ph_agreement_zero_denominator_returns_nan_never_inf_and_never_raises(
    standard_references,
):
    h1_ambient = standard_references["h1_ambient"]
    empty = np.zeros((0, 2))

    agreement = pp.ph_agreement(h1_ambient, empty)
    assert agreement["ref_max_persistence"] == 0.0
    assert np.isnan(agreement["bottleneck_norm"])
    assert np.isnan(agreement["wasserstein_norm"])
    assert not np.isinf(agreement["bottleneck_norm"])
    assert not np.isinf(agreement["wasserstein_norm"])
    assert np.isfinite(agreement["bottleneck"])
    assert np.isfinite(agreement["wasserstein"])


def test_prescale_is_an_isometry_up_to_uniform_scale(standard_references):
    """Building a diagram from a cloud with ``prescale`` on and with it off produces
    diagrams (and the underlying distance matrices) that differ by exactly the applied
    global scalar -- so the choice changes units and never the ordering of features.

    Tolerance measured, not assumed: ``topoae.pairwise_distances_f64`` goes through
    ``torch.cdist``'s formula-based computation (``||a||^2 + ||b||^2 - 2*a.b``, not a
    direct subtraction), which loses float64 precision proportional to the operand
    magnitude. Measured on this exact fixture: max abs difference between
    ``D_scaled`` and ``D_native * scale`` is ``5.114e-08``, comfortably inside the
    ``1e-6`` bound used below -- a real, tight regression bound, not a loosened
    "just pass" tolerance."""
    ambient = standard_references["ambient_cloud"]

    D_scaled, scale = pp.cloud_distance_matrix(ambient, prescale=True)
    D_native, native_scale = pp.cloud_distance_matrix(ambient, prescale=False)

    assert native_scale == 1.0
    assert scale > 0.0

    max_abs_diff_matrix = float(np.max(np.abs(D_scaled - D_native * scale)))
    print(f"prescale isometry: max abs diff (distance matrix) = {max_abs_diff_matrix!r}")
    assert max_abs_diff_matrix < 1e-6

    dgms_scaled = pp.persistence_diagram(D_scaled)
    dgms_native = pp.persistence_diagram(D_native)
    for k in range(2):
        assert dgms_scaled[k].shape == dgms_native[k].shape
        if dgms_scaled[k].shape[0]:
            diff = np.abs(dgms_scaled[k] - dgms_native[k] * scale)
            assert float(diff.max()) < 1e-6


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
