"""Behavioral tests for `cka.py`: the estimator's closed-form behavior, the double-centering
trap, the Gram-matrix-once identity, dtype agreement, and every freeze-guard branch. This suite
loads no PU data, trains nothing, and reads nothing from `notebooks/.cache/`.

Load-bearing tests: `test_unbiased_hsic_matches_reference` (would catch a transposed Gram or a
mis-signed correction term), `test_double_centering_changes_the_answer` (the D8-02 centering
trap, pinned behaviorally rather than by source grep), `test_cka_on_subset_matches_direct` (the
Gram-matrix-once identity this phase's entire runtime budget depends on),
`test_assert_preregistered_rejects_unset_constant` (parametrized over every
`cka._REQUIRED_CONSTANTS` entry -- a constant added later without a guard entry fails this
suite), and the freeze-ancestry proof (`test_freeze_commit_is_a_strict_ancestor_of_head`),
added by 08-04 once the freeze commit (D8-22) existed.
"""
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import cka  # noqa: E402
from pu_manifold import density_stratified_null as dsn  # noqa: E402


ATOL_CLOSED_FORM = 1e-6
ATOL_INDEPENDENCE = 0.05
RTOL_REFERENCE = 1e-12
ATOL_DTYPE = 1e-5

# D8-22's freeze commit -- the commit that filled all 45 of cka.py's pre-registered constants
# (08-04-SUMMARY.md). Every later Phase 8 number must be a strict git descendant of this commit.
FREEZE_COMMIT_SHA = "816863cae2209261470d1d041dcc4484a3056947"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _freeze_commit_exists() -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{FREEZE_COMMIT_SHA}^{{commit}}"],
        cwd=_repo_root(),
        capture_output=True,
    )
    return result.returncode == 0


def _freeze_commit_is_strict_ancestor_of_head() -> bool:
    """True only once at least one commit exists after the freeze commit. Immediately after the
    freeze commit itself (HEAD == freeze commit), this is False and the strict-ancestry test
    below is skipped rather than failed -- the freeze commit being HEAD is the expected state at
    that moment, not a defect (mirrors `test_density_stratified_null.py`'s own precedent)."""
    if not _freeze_commit_exists():
        return False
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT_SHA, "HEAD"],
        cwd=_repo_root(),
    )
    if is_ancestor.returncode != 0:
        return False
    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{FREEZE_COMMIT_SHA}..HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return int(count_result.stdout.strip()) >= 1


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


# --- within-density-stratum tertile split (D8-05/06/07/08) ---------------------------------


def test_tertile_within_stratum_split():
    """Tertile 3 holds the highest-`h` third WITHIN every stratum, not globally. Strata are
    constructed so a purely GLOBAL top-third split would put only stratum-2 points into tertile
    3 (h spans 0-299 / 1000-1299 / 10000-10299 for strata 0/1/2); the within-stratum split must
    instead draw exactly one third of EACH stratum into tertile 3."""
    n_per_stratum = 300
    strata = np.repeat([0, 1, 2], n_per_stratum)
    h = np.concatenate(
        [
            np.arange(n_per_stratum, dtype=float),
            np.arange(n_per_stratum, dtype=float) + 1000.0,
            np.arange(n_per_stratum, dtype=float) + 10000.0,
        ]
    )
    n = strata.shape[0]

    t1, t2, t3 = cka.tertile_split_within_strata(h, strata)

    all_idx = np.concatenate([t1, t2, t3])
    assert np.array_equal(np.sort(all_idx), np.arange(n))
    assert len(set(t1.tolist()) & set(t2.tolist())) == 0
    assert len(set(t1.tolist()) & set(t3.tolist())) == 0
    assert len(set(t2.tolist()) & set(t3.tolist())) == 0

    t3_stratum_counts = np.bincount(strata[t3], minlength=3)
    assert np.all(t3_stratum_counts == n_per_stratum // 3)


def test_tertile_split_density_marginals_match():
    """The three subsets' per-stratum count vectors are equal up to each stratum's own
    ``n_s % 3`` remainder (the remainder-to-last-block convention only ever inflates tertile 3
    relative to tertile 1, never tertile 1 relative to tertile 2)."""
    rng = np.random.default_rng(20260827)
    n = 900
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 3)
    h = rng.standard_normal(n)

    t1, t2, t3 = cka.tertile_split_within_strata(h, strata)

    counts1 = np.bincount(strata[t1], minlength=3)
    counts2 = np.bincount(strata[t2], minlength=3)
    counts3 = np.bincount(strata[t3], minlength=3)
    stratum_sizes = np.bincount(strata, minlength=3)

    for s in range(3):
        remainder = int(stratum_sizes[s] % 3)
        assert counts1[s] == counts2[s]
        assert counts3[s] - counts1[s] == remainder


def test_tertile_split_equal_n_at_every_s():
    """**Deviation from the plan's literal acceptance bound, recorded here and in
    08-02-SUMMARY.md:** the plan states the max-minus-min pooled subset size is "at most the
    number of strata." Dividing a stratum of size `n_s` into 3 contiguous blocks (`n_s // 3`
    each, remainder to the last) can leave a remainder of 0, 1 OR 2 points (`n_s % 3`), so the
    pooled-across-strata difference is `S * (n_s % 3)`, which reaches `2 * S`, not `1 * S`, when
    every stratum's remainder is 2 -- exactly what happens at `S=20` and `S=50` on n=10,000
    (`n // S` = 500 and 200, both `% 3 == 2`). Only `S=10` (`n // S = 1000`, `% 3 == 1`) matches
    the plan's literal "at most S" bound. This is the same class of plan-test-specification bug
    as 08-01-SUMMARY.md's `test_double_centering_changes_the_answer` correction: the underlying
    protection (equal-n subsets up to a small, bounded remainder) is preserved and pinned exactly
    -- this test asserts the TRUE, measured `S * (n_s % 3)` relationship rather than the plan's
    unsatisfiable-for-a-correct-implementation literal bound.
    """
    rng = np.random.default_rng(20260827)
    n = 10_000
    density = rng.standard_normal(n)
    h = rng.standard_normal(n)

    for s_count in (10, 20, 50):
        strata = dsn.density_strata(density, s_count)
        t1, t2, t3 = cka.tertile_split_within_strata(h, strata)
        sizes = [t1.shape[0], t2.shape[0], t3.shape[0]]
        assert max(sizes) - min(sizes) <= 2 * s_count

        stratum_size = n // s_count  # all strata are equal-sized: n % s_count == 0 at this grid
        remainder = stratum_size % 3
        assert max(sizes) - min(sizes) == s_count * remainder


def test_realized_h_contrast_exceeds_one_for_nonconstant_h():
    rng = np.random.default_rng(20260827)
    n = 900
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 3)
    h = rng.uniform(1.0, 10.0, size=n)

    tertiles = cka.tertile_split_within_strata(h, strata)
    contrast = cka.realized_h_contrast(h, tertiles)
    assert contrast > 1.0


def test_tertile_split_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        cka.tertile_split_within_strata(np.zeros(10), np.zeros(9))


def test_tertile_split_raises_on_small_stratum():
    strata = np.array([0, 0, 1, 1, 1, 1])
    h = np.arange(6, dtype=float)
    with pytest.raises(ValueError):
        cka.tertile_split_within_strata(h, strata)


# --- tertile-difference panel and the within-stratum label-permutation null (D8-10/D8-11) --


def _synthetic_grams(rng, n, p=16):
    """A cheap pair of linear Gram matrices for the null tests below -- built BEFORE any test
    disables the Gram builders, so the resulting K_full/L_full dicts are the caller's only
    reference to the underlying feature matrices."""
    Z1 = rng.standard_normal((n, p))
    Z2 = rng.standard_normal((n, p))
    K_full = {"linear": cka.linear_gram(Z1, np.float64)}
    L_full = {"linear": cka.linear_gram(Z2, np.float64)}
    return K_full, L_full


def test_tertile_gap_panel_returns_gap_per_kernel():
    rng = np.random.default_rng(20260827)
    n = 300
    K_full, L_full = _synthetic_grams(rng, n)
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 3)
    h = rng.standard_normal(n)
    tertiles = cka.tertile_split_within_strata(h, strata)

    panel = cka.tertile_gap_panel(K_full, L_full, tertiles)
    assert set(panel.keys()) == {"linear"}
    row = panel["linear"]
    assert set(row.keys()) == {"cka_t1", "cka_t2", "cka_t3", "gap"}
    np.testing.assert_allclose(row["gap"], row["cka_t3"] - row["cka_t1"])


def test_tertile_gap_panel_raises_on_key_mismatch():
    rng = np.random.default_rng(20260827)
    n = 300
    K_full, _ = _synthetic_grams(rng, n)
    _, L_full = _synthetic_grams(rng, n)
    L_full["extra"] = L_full.pop("linear")
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 3)
    h = rng.standard_normal(n)
    tertiles = cka.tertile_split_within_strata(h, strata)

    with pytest.raises(ValueError):
        cka.tertile_gap_panel(K_full, L_full, tertiles)


def test_stratified_tertile_null_preserves_sizes():
    rng = np.random.default_rng(20260827)
    n = 600
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 6)
    h = rng.standard_normal(n)
    K_full, L_full = _synthetic_grams(rng, n)

    observed_tertiles = cka.tertile_split_within_strata(h, strata)
    observed_sizes = tuple(t.shape[0] for t in observed_tertiles)

    n_resamples = 25
    null = cka.stratified_tertile_label_null(h, strata, K_full, L_full, n_resamples, seed=1)
    assert set(null.keys()) == {"linear"}
    assert null["linear"].shape == (n_resamples,)

    # Re-derive every resample's permuted tertile sizes from the SAME seed and draw order the
    # function itself uses, and confirm all three sizes equal the observed split's sizes
    # exactly, elementwise, for all 25 resamples.
    rng2 = np.random.default_rng(1)
    strat_indices = [np.where(strata == s)[0] for s in np.unique(strata)]
    for _ in range(n_resamples):
        h_perm = h.copy()
        for idx in strat_indices:
            h_perm[idx] = h[rng2.permutation(idx)]
        permuted_tertiles = cka.tertile_split_within_strata(h_perm, strata)
        permuted_sizes = tuple(t.shape[0] for t in permuted_tertiles)
        assert permuted_sizes == observed_sizes


def test_null_panel_has_one_array_per_kernel():
    rng = np.random.default_rng(20260827)
    n = 600
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 6)
    h = rng.standard_normal(n)

    Z1 = rng.standard_normal((n, 16))
    Z2 = rng.standard_normal((n, 16))
    sigma = cka.median_pairwise_distance(Z1)
    K_full = {
        "linear": cka.linear_gram(Z1, np.float64),
        "rbf": cka.rbf_gram(Z1, sigma, np.float64),
    }
    L_full = {
        "linear": cka.linear_gram(Z2, np.float64),
        "rbf": cka.rbf_gram(Z2, sigma, np.float64),
    }

    null = cka.stratified_tertile_label_null(h, strata, K_full, L_full, n_resamples=25, seed=2)
    assert set(null.keys()) == {"linear", "rbf"}
    for arr in null.values():
        assert arr.shape == (25,)


def test_null_is_seed_reproducible():
    rng = np.random.default_rng(20260827)
    n = 600
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 6)
    h = rng.standard_normal(n)
    K_full, L_full = _synthetic_grams(rng, n)

    null_a = cka.stratified_tertile_label_null(h, strata, K_full, L_full, n_resamples=25, seed=42)
    null_b = cka.stratified_tertile_label_null(h, strata, K_full, L_full, n_resamples=25, seed=42)
    null_c = cka.stratified_tertile_label_null(h, strata, K_full, L_full, n_resamples=25, seed=43)

    assert np.array_equal(null_a["linear"], null_b["linear"])
    assert not np.array_equal(null_a["linear"], null_c["linear"])


def test_null_does_not_rebuild_grams(monkeypatch):
    """Passes Gram matrices built BEFORE the Gram builders are disabled below, proving
    `stratified_tertile_label_null`'s resample loop cannot be re-deriving them: if the loop
    called `linear_gram`/`rbf_gram` at all, this test would raise."""
    rng = np.random.default_rng(20260827)
    n = 600
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 6)
    h = rng.standard_normal(n)
    K_full, L_full = _synthetic_grams(rng, n)

    def _raise(*args, **kwargs):
        raise AssertionError("a Gram builder was called inside the null's resample loop")

    monkeypatch.setattr(cka, "linear_gram", _raise)
    monkeypatch.setattr(cka, "rbf_gram", _raise)

    null = cka.stratified_tertile_label_null(h, strata, K_full, L_full, n_resamples=25, seed=7)
    assert null["linear"].shape == (25,)


def test_null_threshold_is_two_tailed():
    null_array = np.arange(100, dtype=float)  # 0..99
    low, high = cka.null_threshold(null_array, quantile_per_tail=0.975)
    assert low < high
    np.testing.assert_allclose(low, np.quantile(null_array, 0.025))
    np.testing.assert_allclose(high, np.quantile(null_array, 0.975))


# --- verdict rules: clearance at every S, per-d independence, seed unanimity (D8-09/12/13/15) -


def test_per_d_verdict_requires_rule():
    with pytest.raises(RuntimeError):
        cka.per_d_verdict({10: 0.1}, {10: (-0.05, 0.05)}, rule="")


def test_per_d_verdict_raises_on_key_mismatch():
    with pytest.raises(ValueError):
        cka.per_d_verdict({10: 0.1, 20: 0.1}, {10: (-0.05, 0.05)}, rule="rule naming S_GRID")


def test_verdict_requires_clearance_at_every_s():
    thresholds_by_s = {10: (-0.05, 0.05), 20: (-0.05, 0.05), 50: (-0.05, 0.05)}

    # Two of three S values clear; the third does not -> DOES NOT CLEAR, n_s_cleared == 2.
    gaps_two_of_three = {10: 0.2, 20: 0.2, 50: 0.01}
    result = cka.per_d_verdict(gaps_two_of_three, thresholds_by_s, rule="rule naming S_GRID")
    assert result["verdict"] == "DOES NOT CLEAR"
    assert result["n_s_cleared"] == 2

    # All three S clear -> CLEARS AT EVERY S.
    gaps_all_three = {10: 0.2, 20: 0.2, 50: 0.2}
    result_all = cka.per_d_verdict(gaps_all_three, thresholds_by_s, rule="rule naming S_GRID")
    assert result_all["verdict"] == "CLEARS AT EVERY S"
    assert result_all["n_s_cleared"] == 3


def test_middle_tertile_does_not_gate():
    rng = np.random.default_rng(20260827)
    n = 300
    K_full, L_full = _synthetic_grams(rng, n)
    density = rng.standard_normal(n)
    strata = dsn.density_strata(density, 3)
    h = rng.standard_normal(n)
    tertiles = cka.tertile_split_within_strata(h, strata)
    panel = cka.tertile_gap_panel(K_full, L_full, tertiles)

    gap = panel["linear"]["gap"]
    # S_GRID is frozen to (10, 20, 50) as of the 08-04 freeze commit; per_d_verdict's coverage
    # guard requires gaps_by_s/thresholds_by_s to cover every S in it, so this structural test
    # (proving the middle tertile is never read) supplies the same synthetic gap at all three S
    # values rather than the single arbitrary key it used pre-freeze.
    gaps_by_s = {s: gap for s in cka.S_GRID}
    thresholds_by_s = {s: (gap - 1.0, gap + 1.0) for s in cka.S_GRID}  # deliberately does not clear
    result_before = cka.per_d_verdict(gaps_by_s, thresholds_by_s, rule="rule naming S_GRID")

    # Sabotage the middle tertile's reported CKA. gap = cka_t3 - cka_t1 never reads cka_t2, so
    # re-deriving gaps_by_s from the sabotaged panel -- and the resulting verdict -- must be
    # byte-identical to before the sabotage.
    panel["linear"]["cka_t2"] = float("nan")
    gaps_by_s_after = {s: panel["linear"]["gap"] for s in cka.S_GRID}
    result_after = cka.per_d_verdict(gaps_by_s_after, thresholds_by_s, rule="rule naming S_GRID")

    assert result_before == result_after


def test_per_d_verdicts_are_independent():
    """A null at one d does not silently void another, and no pooled headline is invented
    (07.1's D-14 pattern, D8-13)."""
    thresholds_by_s = {10: (-0.05, 0.05), 20: (-0.05, 0.05), 50: (-0.05, 0.05)}
    gaps_d20 = {10: 0.2, 20: 0.2, 50: 0.2}
    gaps_d32 = {10: 0.0, 20: 0.0, 50: 0.0}
    gaps_d20_before = dict(gaps_d20)

    result_d32 = cka.per_d_verdict(gaps_d32, thresholds_by_s, rule="rule naming S_GRID")
    assert gaps_d20 == gaps_d20_before  # computing d=32's verdict did not mutate d=20's inputs

    result_d20 = cka.per_d_verdict(gaps_d20, thresholds_by_s, rule="rule naming S_GRID")
    assert result_d20["verdict"] == "CLEARS AT EVERY S"
    assert result_d32["verdict"] == "DOES NOT CLEAR"


def test_combine_seed_verdicts_requires_exactly_three():
    """**Deviation from the plan's literal "four distinct outcome strings" phrasing, recorded
    here and in 08-02-SUMMARY.md:** the plan's own must_haves truth for this function states
    "returns a terminal split outcome for one or two clearances WITHOUT UPGRADING BY MAJORITY
    VOTE" -- so a count of 1 and a count of 2 clearances must map to the SAME
    "SPLIT ACROSS SEEDS" outcome, not two different strings (upgrading 2-of-3 to a more
    favorable fourth string would BE the majority-vote upgrade D8-15 forbids). This mirrors
    linear_probe.combine_seed_verdicts' own three-outcome shape exactly. This test asserts the
    TRUE, ratified 3-outcome mapping across all four clearance counts (0, 1, 2, 3) -- the same
    class of plan-test-specification correction as 08-01-SUMMARY.md's
    test_double_centering_changes_the_answer.
    """
    clears, does_not = "CLEARS AT EVERY S", "DOES NOT CLEAR"
    rule = "rule naming SPLIT ACROSS SEEDS"

    with pytest.raises(ValueError):
        cka.combine_seed_verdicts({0: clears, 1: clears}, rule)
    with pytest.raises(ValueError):
        cka.combine_seed_verdicts({0: clears, 1: clears, 2: clears, 3: clears}, rule)

    r0 = cka.combine_seed_verdicts({0: does_not, 1: does_not, 2: does_not}, rule)
    r1 = cka.combine_seed_verdicts({0: clears, 1: does_not, 2: does_not}, rule)
    r2 = cka.combine_seed_verdicts({0: clears, 1: clears, 2: does_not}, rule)
    r3 = cka.combine_seed_verdicts({0: clears, 1: clears, 2: clears}, rule)

    assert r0["phase_verdict"] == "NO CLEARANCE IN ANY SEED"
    assert r1["phase_verdict"] == "SPLIT ACROSS SEEDS"
    assert r2["phase_verdict"] == "SPLIT ACROSS SEEDS"
    assert r3["phase_verdict"] == "CLEARS IN ALL THREE SEEDS"


def test_combine_seed_verdicts_requires_rule():
    with pytest.raises(RuntimeError):
        cka.combine_seed_verdicts(
            {0: "CLEARS AT EVERY S", 1: "CLEARS AT EVERY S", 2: "CLEARS AT EVERY S"}, ""
        )


def test_combine_seed_verdicts_rejects_invalid_verdict_value():
    with pytest.raises(ValueError):
        cka.combine_seed_verdicts({0: "CLEARS AT EVERY S", 1: "MAYBE", 2: "DOES NOT CLEAR"}, "rule")


def test_seed_pooling_raises():
    with pytest.raises(RuntimeError, match="05-03-DECISION"):
        cka.pooled_field_guard(["seed0", "seed1"])
    cka.pooled_field_guard(["seed0"])  # exactly one field never raises


def test_assert_preregistered_rejects_wrong_seed_handling_rule(monkeypatch):
    """D8-15/T-08-09: SEED_HANDLING_RULE is guarded by EXACT STRING EQUALITY, not truthiness --
    a non-empty string other than the ratified value must still fail."""
    for name, value in _PLAUSIBLE_FILLED_VALUES.items():
        monkeypatch.setattr(cka, name, value)
    monkeypatch.setattr(cka, "SEED_HANDLING_RULE", "pool_all_seeds_into_one_field")
    with pytest.raises(RuntimeError, match="SEED_HANDLING_RULE"):
        cka.assert_preregistered()


def test_assert_preregistered_rejects_verdict_rule_missing_s_grid(monkeypatch):
    """D8-09/T-08-10: VERDICT_RULE must NAME S_GRID -- a non-empty string that omits it must
    still fail."""
    for name, value in _PLAUSIBLE_FILLED_VALUES.items():
        monkeypatch.setattr(cka, name, value)
    monkeypatch.setattr(cka, "VERDICT_RULE", "clearance is required at every threshold point")
    with pytest.raises(RuntimeError, match="VERDICT_RULE"):
        cka.assert_preregistered()


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
    "S_GRID": (10, 20, 50),
    "N_TERTILES": 3,
    "DENSITY_K": 30,
    "DENSITY_FIELD_D": 20,
    "DENSITY_INPUT": "legacysurvey_ambient_768",
    "DENSITY_SIGN_CONVENTION": (
        "Relative density 1.0 / w, matching Phase 4's REGN-01 printed convention."
    ),
    "STRATIFICATION_RULE": (
        "Equal-count quantile bins on density rank; tertiles rank density-residualized "
        "curvature, not raw ||H||."
    ),
    "SENSITIVITY_GRID_RULE": (
        "S_GRID is a grid of THRESHOLDS, not point estimates; no headline S; clearance is "
        "required at every S."
    ),
    "N_PERMUTATIONS": 1000,
    "PERMUTATION_SEED": 20260827,
    "NULL_QUANTILE_PER_TAIL": 0.975,
    "NULL_KERNELS": ("linear", "rbf_sigma"),
    "TERTILE_STATISTIC_RULE": (
        "CKA(tertile 3) - CKA(tertile 1); the middle tertile is a shape diagnostic and gates "
        "nothing."
    ),
    "NULL_CONSTRUCTION_RULE": (
        "Permutes ||H|| tertile labels within density strata; not mknn.permutation_null's "
        "row-pairing shuffle; not a bootstrap CI."
    ),
    "MIDDLE_TERTILE_IS_NON_GATING": True,
    "D_SWEEP": (20, 25, 32),
    "SEED_FIELD_D": 25,
    "TORCH_INIT_SEEDS": (0, 1, 2),
    "VERDICT_RULE": (
        "Clearance is required at every S in S_GRID; there is no headline S."
    ),
    "SEED_HANDLING_RULE": "no_pooling_per_seed_verdicts",
    "SEED_VERDICT_COMBINATION_RULE": (
        "Unanimous 3-of-3 -> CLEARS IN ALL THREE SEEDS; zero -> NO CLEARANCE IN ANY SEED; one "
        "or two -> the terminal SPLIT ACROSS SEEDS, never upgraded by majority vote."
    ),
    "D32_IS_NON_GATING": True,
    "VALIDATION_LADDER_IS_NON_GATING": True,
    "N_REPEATS": 30,
    "NEGATIVE_CONTROL_FIELD": "h_norm_25",
    "PLANTED_EFFECT_GRID": (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50),
    "PLANTED_EFFECT_SEED": 20260827,
    "RECORD_STEM": "08_cka_alignment",
    "REPORTING_BLOCK_ROWS": (
        "d32_gap",
        "shuffled_h_false_positive_rate",
        "planted_effect_detection_floor",
        "realized_h_contrast_per_s",
        "sigma_rungs",
    ),
    "REPORTING_BLOCK_RULE": (
        "08-FINDINGS.md prints all five REPORTING_BLOCK_ROWS regardless of outcome, beside the "
        "headline, never in an appendix."
    ),
    "VERDICT_SENTENCE_RULE": (
        "The verdict sentence cannot be written without d=32's gap and the shuffled-||H|| "
        "false-positive rate in the same sentence."
    ),
}


def test_assert_preregistered_passes_when_all_constants_set(monkeypatch):
    for name, value in _PLAUSIBLE_FILLED_VALUES.items():
        monkeypatch.setattr(cka, name, value)
    cka.assert_preregistered()


# The UNSET sentinel `assert_preregistered` treats as "not filled", keyed by the value TYPE each
# `_PLAUSIBLE_FILLED_VALUES` entry carries -- `None` has no single UNSET counterpart of its own
# (a bare `None` already IS one of the three UNSET sentinels), so bool/int/float constants use
# `None` as their UNSET value.
_UNSET_SENTINEL_FOR_TYPE = {
    tuple: (),
    str: "",
}


def _unset_value_for(name: str):
    """The UNSET sentinel matching `name`'s plausible-filled-value type: `()` for tuples, `""`
    for strings, `None` for everything else (bool/int/float) -- the three sentinels
    `cka.assert_preregistered`'s generic UNSET check recognizes."""
    plausible = _PLAUSIBLE_FILLED_VALUES[name]
    return _UNSET_SENTINEL_FOR_TYPE.get(type(plausible), None)


@pytest.mark.parametrize("name", cka._REQUIRED_CONSTANTS)
def test_assert_preregistered_rejects_unset_constant(name, monkeypatch):
    """For each name in `_REQUIRED_CONSTANTS`, monkeypatch every OTHER required constant to a
    plausible filled value, EXPLICITLY monkeypatch the one under test to its UNSET sentinel (not
    its real, now-frozen module value -- 08-04 filled all 45 constants for real, so relying on
    the module's own state would no longer exercise the UNSET branch), then assert
    `assert_preregistered()` raises `RuntimeError` naming it. This is the mechanism that makes a
    constant added later without a guard entry fail this suite."""
    for other_name, filled_value in _PLAUSIBLE_FILLED_VALUES.items():
        if other_name != name:
            monkeypatch.setattr(cka, other_name, filled_value)
    monkeypatch.setattr(cka, name, _unset_value_for(name))
    with pytest.raises(RuntimeError, match=name):
        cka.assert_preregistered()


# --- 08-04: the freeze commit itself, its ancestry proof, and the two exact-content guards ----


def test_assert_preregistered_passes():
    """After the 08-04 freeze commit, `assert_preregistered()` returns without raising against
    the REAL, unmodified module state -- no monkeypatching. This is distinct from
    `test_assert_preregistered_passes_when_all_constants_set` above (which proves the guard's
    generic UNSET sweep is satisfiable in principle, using synthetic plausible values); this test
    proves the actual frozen constants in this file satisfy it."""
    cka.assert_preregistered()
    assert len(cka._REQUIRED_CONSTANTS) == 45


def test_freeze_commit_exists():
    assert _freeze_commit_exists(), (
        f"FREEZE_COMMIT_SHA={FREEZE_COMMIT_SHA!r} does not exist as a commit in this checkout's "
        "history."
    )


@pytest.mark.skipif(
    not _freeze_commit_is_strict_ancestor_of_head(),
    reason=(
        "freeze commit is not (yet) a STRICT ancestor of HEAD -- either it is absent from this "
        "checkout's history (e.g. a shallow clone), or HEAD IS the freeze commit itself (the "
        "expected state immediately after the freeze, before this test file's own commit "
        "lands). 08-05 onward re-runs the same ancestry check unconditionally at the moment a "
        "Phase 8 number is produced, which is where it actually bites."
    ),
)
def test_freeze_commit_is_a_strict_ancestor_of_head():
    """The precision requirement: a commit is its own ancestor, so `--is-ancestor` alone would
    pass even if a Phase 8 number were produced in the freeze commit itself. `git rev-list
    --count <freeze>..HEAD` must also be at least 1."""
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT_SHA, "HEAD"],
        cwd=_repo_root(),
    )
    assert is_ancestor.returncode == 0, "freeze commit is not an ancestor of HEAD at all"

    count_result = subprocess.run(
        ["git", "rev-list", "--count", f"{FREEZE_COMMIT_SHA}..HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    strict_distance = int(count_result.stdout.strip())
    assert strict_distance >= 1, (
        "freeze commit is not a STRICT ancestor of HEAD -- HEAD IS the freeze commit "
        "(strict_distance == 0), which would mean no number-producing commit exists yet"
    )


def test_seed_handling_rule_is_exact(monkeypatch):
    """D8-15/T-08-09: `SEED_HANDLING_RULE` is guarded by EXACT STRING EQUALITY, not truthiness --
    monkeypatching it to any OTHER non-empty string (not just an UNSET one) must still make
    `assert_preregistered` raise, so a future edit reintroducing seed pooling under a
    differently-worded rule string fails loudly rather than passing because the string happens
    to be non-empty."""
    monkeypatch.setattr(cka, "SEED_HANDLING_RULE", "pool_all_seeds_into_one_field")
    with pytest.raises(RuntimeError, match="SEED_HANDLING_RULE"):
        cka.assert_preregistered()


def test_reporting_block_rows_are_complete():
    """D8-21: `REPORTING_BLOCK_ROWS` names exactly the five required row identifiers, in order --
    a row silently dropped by a later edit would shrink the tuple below five without necessarily
    tripping any UNSET check."""
    assert cka.REPORTING_BLOCK_ROWS == (
        "d32_gap",
        "shuffled_h_false_positive_rate",
        "planted_effect_detection_floor",
        "realized_h_contrast_per_s",
        "sigma_rungs",
    )
