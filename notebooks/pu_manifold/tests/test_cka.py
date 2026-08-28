"""Behavioral tests for `cka.py`: the estimator's closed-form behavior, the double-centering
trap, the Gram-matrix-once identity, dtype agreement, and every freeze-guard branch. This suite
loads no PU data, trains nothing, and reads nothing from `notebooks/.cache/`.

Load-bearing tests: `test_unbiased_hsic_matches_reference` (would catch a transposed Gram or a
mis-signed correction term), `test_double_centering_changes_the_answer` (the D8-02 centering
trap, pinned behaviorally rather than by source grep), `test_cka_on_subset_matches_direct` (the
Gram-matrix-once identity this phase's entire runtime budget depends on), and
`test_assert_preregistered_rejects_unset_constant` (parametrized over every
`cka._REQUIRED_CONSTANTS` entry -- a constant added later without a guard entry fails this
suite).
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import cka  # noqa: E402


ATOL_CLOSED_FORM = 1e-6
ATOL_INDEPENDENCE = 0.05
RTOL_REFERENCE = 1e-12
ATOL_DTYPE = 1e-5


def _random_orthogonal(p, rng):
    a = rng.standard_normal((p, p))
    q, _ = np.linalg.qr(a)
    return q


# --- estimator correctness ------------------------------------------------------------------


def test_unbiased_hsic_matches_reference():
    """A reference value computed in this test from the same Song et al. (2012) formula,
    written independently as three explicit terms, rather than by calling the module -- this is
    the test that would catch a transposed Gram or a mis-signed correction term."""
    rng = np.random.default_rng(20260827)
    n = 20
    A = rng.standard_normal((n, n))
    K = A @ A.T
    B = rng.standard_normal((n, n))
    L = B @ B.T

    Kt = K.copy()
    np.fill_diagonal(Kt, 0.0)
    Lt = L.copy()
    np.fill_diagonal(Lt, 0.0)
    ones = np.ones(n)

    term1 = np.trace(Kt @ Lt)
    term2 = (ones @ Kt @ ones) * (ones @ Lt @ ones) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (ones @ Kt @ Lt @ ones)
    reference = (term1 + term2 - term3) / (n * (n - 3))

    measured = cka.unbiased_hsic(K, L)
    np.testing.assert_allclose(measured, reference, rtol=RTOL_REFERENCE)


def test_unbiased_hsic_raises_below_n4():
    rng = np.random.default_rng(20260827)
    for n in (2, 3):
        K = rng.standard_normal((n, n))
        L = rng.standard_normal((n, n))
        with pytest.raises(ValueError):
            cka.unbiased_hsic(K, L)
    n = 4
    K = rng.standard_normal((n, n))
    L = rng.standard_normal((n, n))
    cka.unbiased_hsic(K, L)  # succeeds -- does not raise


def test_double_centering_changes_the_answer():
    """The pin against Pitfall 1 (D8-02).

    **Deviation from the plan's literal test description, recorded here and in
    08-01-SUMMARY.md:** the plan describes this test as feeding a double-centered Gram to
    `unbiased_hsic` and asserting the result differs from the raw zero-diagonal result. Measured
    directly (both by hand-derivation and numerically to ~1e-16 relative on multiple random
    seeds/sizes): the Song et al. (2012) U-statistic correction terms (`term2`/`term3`) make
    `unbiased_hsic` provably INVARIANT to double-centering its input -- `unbiased_hsic(H K H,
    H L H) == unbiased_hsic(K, L)` to machine precision. This is the correct, intended behavior
    of the correction terms (they are what makes explicit centering unnecessary in the first
    place), not a bug -- so a mathematically correct implementation can never satisfy "differs
    by more than 1e-9 relative" under that literal construction, and the plan's assertion
    direction was inverted from what a correct implementation actually does.

    The REAL trap Pitfall 1 warns against is a different, genuinely detectable failure mode:
    silently substituting the CLASSICAL BIASED HSIC formula (`tr(K_c L_c) / (n-1)^2` on
    double-centered `K_c`/`L_c`, Kornblith et al. 2019's `HSIC_0`) in place of the unbiased
    correction-term formula, while calling the result "unbiased". This test pins THAT trap: it
    builds the classical biased quantity directly (never via `cka.unbiased_hsic`) and asserts it
    differs materially from `cka.unbiased_hsic`'s own output on the SAME raw `K`/`L` -- so a
    future edit that quietly replaces the correction-term computation with the classical
    centered-and-divide-by-(n-1)^2 formula fails this test.
    """
    rng = np.random.default_rng(20260827)
    n = 50
    A = rng.standard_normal((n, n))
    K = A @ A.T
    B = rng.standard_normal((n, n))
    L = B @ B.T

    unbiased_value = cka.unbiased_hsic(K, L)

    H = np.eye(n) - np.ones((n, n)) / n
    biased_value = np.trace(H @ K @ H @ L) / (n - 1) ** 2

    relative_diff = abs(biased_value - unbiased_value) / abs(unbiased_value)
    assert relative_diff > 1e-9

    # Positive confirmation of the invariance property itself, so a future reader does not
    # mistake this test's construction for an oversight: unbiased_hsic on an EXPLICITLY
    # pre-centered pair matches its own raw-input value to near machine precision.
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    centered_value = cka.unbiased_hsic(K_centered, L_centered)
    np.testing.assert_allclose(centered_value, unbiased_value, rtol=1e-9)


# --- invariance ladder (D8-16) ---------------------------------------------------------------


def test_linear_cka_invariances():
    rng = np.random.default_rng(20260827)
    n, p = 2000, 64
    Z = rng.standard_normal((n, p))
    Q = _random_orthogonal(p, rng)

    K = cka.linear_gram(Z, np.float64)

    K_rot = cka.linear_gram(Z @ Q, np.float64)
    assert abs(cka.cka(K, K_rot) - 1.0) < ATOL_CLOSED_FORM

    K_scaled = cka.linear_gram(3.0 * Z, np.float64)
    assert abs(cka.cka(K, K_scaled) - 1.0) < ATOL_CLOSED_FORM

    Z_indep = rng.standard_normal((n, p))
    K_indep = cka.linear_gram(Z_indep, np.float64)
    assert abs(cka.cka(K, K_indep)) < ATOL_INDEPENDENCE


def test_rbf_cka_invariances():
    rng = np.random.default_rng(20260827)
    n, p = 2000, 64
    Z = rng.standard_normal((n, p))
    Q = _random_orthogonal(p, rng)
    sigma = cka.median_pairwise_distance(Z)  # frozen for this entire test -- never recomputed

    K = cka.rbf_gram(Z, sigma, np.float64)

    K_rot = cka.rbf_gram(Z @ Q, sigma, np.float64)
    assert abs(cka.cka(K, K_rot) - 1.0) < ATOL_CLOSED_FORM

    K_scaled = cka.rbf_gram(3.0 * Z, sigma, np.float64)
    assert abs(cka.cka(K, K_scaled) - 1.0) > ATOL_CLOSED_FORM  # RBF is NOT scale-invariant

    Z_indep = rng.standard_normal((n, p))
    K_indep = cka.rbf_gram(Z_indep, sigma, np.float64)
    assert abs(cka.cka(K, K_indep)) < ATOL_INDEPENDENCE


def test_noise_ladder_monotone():
    rng = np.random.default_rng(20260827)
    n, p = 2000, 64
    Z = rng.standard_normal((n, p))
    sigma = cka.median_pairwise_distance(Z)
    K_lin = cka.linear_gram(Z, np.float64)
    K_rbf = cka.rbf_gram(Z, sigma, np.float64)

    noise_scales = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    lin_values, rbf_values = [], []
    for scale in noise_scales:
        Z_noisy = Z + scale * rng.standard_normal((n, p))
        lin_values.append(cka.cka(K_lin, cka.linear_gram(Z_noisy, np.float64)))
        rbf_values.append(cka.cka(K_rbf, cka.rbf_gram(Z_noisy, sigma, np.float64)))

    assert all(lin_values[i] > lin_values[i + 1] for i in range(len(lin_values) - 1))
    assert all(rbf_values[i] > rbf_values[i + 1] for i in range(len(rbf_values) - 1))


# --- Gram-matrix-once / index-many identity ---------------------------------------------------


def test_cka_on_subset_matches_direct():
    rng = np.random.default_rng(20260827)
    n, p = 500, 32
    Z1 = rng.standard_normal((n, p))
    Z2 = rng.standard_normal((n, p))
    K_full = cka.linear_gram(Z1, np.float64)
    L_full = cka.linear_gram(Z2, np.float64)

    idx = rng.choice(n, size=200, replace=False)
    idx.sort()

    via_subset = cka.cka_on_subset(K_full, L_full, idx)
    direct = cka.cka(
        cka.linear_gram(Z1[idx], np.float64), cka.linear_gram(Z2[idx], np.float64)
    )
    np.testing.assert_allclose(via_subset, direct, atol=1e-10)


def test_gram_dtype_agreement():
    rng = np.random.default_rng(20260827)
    n, p = 3000, 64
    Z1 = rng.standard_normal((n, p))
    Z2 = rng.standard_normal((n, p))

    value64 = cka.cka(cka.linear_gram(Z1, np.float64), cka.linear_gram(Z2, np.float64))
    value32 = cka.cka(cka.linear_gram(Z1, np.float32), cka.linear_gram(Z2, np.float32))
    np.testing.assert_allclose(value32, value64, atol=ATOL_DTYPE)


def test_sigma_is_required_argument():
    """Pin against the per-subset-bandwidth confound (D8-03): `rbf_gram`'s `sigma` parameter
    has no default, so a call site that omits it cannot silently reintroduce the confound."""
    sig = inspect.signature(cka.rbf_gram)
    assert sig.parameters["sigma"].default is inspect.Parameter.empty


# --- freeze guard -------------------------------------------------------------------------


_PLAUSIBLE_FILLED_VALUES = {
    "KERNELS": ("linear", "rbf"),
    "SIGMA_MULTIPLIERS": (0.5, 1.0, 2.0),
    "SIGMA_HSC": 1.2345,
    "SIGMA_LEGACYSURVEY": 2.3456,
    "GRAM_DTYPE": "float32",
    "HSIC_ESTIMATOR_RULE": "unbiased HSIC, Song et al. 2012, never double-centered.",
    "SIGMA_FREEZE_RULE": "sigma is frozen once, globally, per modality, before any subset exists.",
    "ALIGNMENT_METRIC": "cka",
    "SUPERSEDES": ("crossmodal_curvature.ALIGNMENT_METRIC",),
    "SUPERSESSION_RULE": (
        "Phase 8 supersedes D7-07 by phase decision, never by editing the sealed module."
    ),
    "SWISS_ROLL_APPLICABILITY_RULE": (
        "The Swiss roll standing rule is declared not applicable to Phase 8."
    ),
    "RBF_IS_NON_GATING": True,
    "SIGMA_LADDER_IS_NON_GATING": True,
    "DIAGNOSTICS_ARE_NON_GATING": True,
}


def test_assert_preregistered_passes_when_all_constants_set(monkeypatch):
    for name, value in _PLAUSIBLE_FILLED_VALUES.items():
        monkeypatch.setattr(cka, name, value)
    cka.assert_preregistered()


@pytest.mark.parametrize("name", cka._REQUIRED_CONSTANTS)
def test_assert_preregistered_rejects_unset_constant(name, monkeypatch):
    """For each name in `_REQUIRED_CONSTANTS`, monkeypatch every OTHER required constant to a
    plausible filled value and leave the one under test at its real (UNSET) module value, then
    assert `assert_preregistered()` raises `RuntimeError` naming it. This is the mechanism that
    makes a constant added later without a guard entry fail this suite."""
    for other_name, filled_value in _PLAUSIBLE_FILLED_VALUES.items():
        if other_name != name:
            monkeypatch.setattr(cka, other_name, filled_value)
    with pytest.raises(RuntimeError, match=name):
        cka.assert_preregistered()
