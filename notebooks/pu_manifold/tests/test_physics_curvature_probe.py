"""Unit tests for ``pu_manifold.physics_curvature_probe`` -- the Phase 9 statistics module.

This suite loads no Physics data, trains nothing, opens no HuggingFace dataset and reads
nothing from the output root -- every fixture is an in-memory numpy array built in the test
with a fixed ``np.random.default_rng`` seed. Load-bearing tests:
``test_controlled_partial_reproduces_colleague_numbers`` (the parity pin against
``09-COLLEAGUE-REANALYSIS.md``'s published numbers), ``test_oof_predictions_are_out_of_fold``
and ``test_oof_raises_on_incomplete_coverage`` (the out-of-fold structural proof),
``test_radial_decomposition_identity`` and ``test_radial_decomposition_on_analytic_sphere`` (the
decomposition's Pythagorean identity and its known-answer anchor), and
``test_assert_preregistered_rejects_unset_constant`` (the freeze guard's malformed-constant
sweep over every entry of ``_REQUIRED_CONSTANTS``).
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pu_manifold import physics_curvature_probe as pcp  # noqa: E402
from pu_manifold import linear_probe  # noqa: E402

# Tolerance literals -- test-local, NOT pre-registered constants (D9-18 enumerates what the
# freeze covers; tolerances are in none of those categories).
RTOL_DECOMPOSITION = 1e-10
ATOL_PARITY = 1e-3
ATOL_TARGET_RHO = 0.02
FRAC_SPHERE_TOLERANCE = 0.10


# --- freeze-commit ancestry scaffold (FREEZE_COMMIT_SHA wired by plan 09-05) -------------------

FREEZE_COMMIT_SHA = "5f7fbe27afb0ef2a76353b41fa5713e760bbeea5"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _freeze_commit_exists() -> bool:
    if not FREEZE_COMMIT_SHA:
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{FREEZE_COMMIT_SHA}^{{commit}}"],
        cwd=_repo_root(),
        capture_output=True,
    )
    return result.returncode == 0


def _freeze_commit_is_strict_ancestor_of_head() -> bool:
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


def test_freeze_commit_is_a_strict_ancestor_of_head():
    """09-05 filled FREEZE_COMMIT_SHA with the real freeze commit's SHA; this test now
    exercises the real ancestry proof (no longer skipped)."""
    assert _freeze_commit_is_strict_ancestor_of_head()


def test_freeze_commit_sha_is_full_lowercase_hex():
    """An abbreviation must never be pasted in later -- FREEZE_COMMIT_SHA must always be a full
    40-character lowercase hex string."""
    assert isinstance(FREEZE_COMMIT_SHA, str)
    assert len(FREEZE_COMMIT_SHA) == 40
    assert re.fullmatch(r"[0-9a-f]{40}", FREEZE_COMMIT_SHA)


# --- parity pin: reproduce the colleague's published numbers ---------------------------------


def test_controlled_partial_reproduces_colleague_numbers():
    """Builds a 512-row synthetic table via a Gaussian copula whose correlation matrix was
    solved (offline, via scipy.optimize, against ``cross_split_curvature.partial_spearman``'s
    OWN formula -- not a hand-derived approximation) to reproduce
    ``09-COLLEAGUE-REANALYSIS.md``'s published raw ``-0.4124`` and 3-control-controlled
    ``-0.2405`` values at d=16. The Cholesky factor below is that solved construction's fixed
    literal; this test itself performs no optimization -- it is a golden-array replay."""
    n = 512
    seed = 20260902
    # Cholesky factor of the (nearest-PSD) 5x5 correlation matrix over
    # (K_H_cross, r2_G, log_knn_radius, local_label_variance, local_evaluation_count), solved so
    # that raw spearman(K_H_cross, r2_G) and cross_split_curvature.partial_spearman(K_H_cross,
    # r2_G, controls=[log_knn_radius, local_label_variance, local_evaluation_count]) land within
    # 1e-6 of the colleague's published -0.4124 / -0.2405 (09-COLLEAGUE-REANALYSIS.md).
    L = np.array(
        [
            [1.00000050e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [-3.88734074e-01, 9.21350541e-01, 0.00000000e00, 0.00000000e00, 0.00000000e00],
            [7.99124856e-01, -3.40573434e-02, 6.00200434e-01, 0.00000000e00, 0.00000000e00],
            [-1.29725927e-04, -2.45341635e-05, 7.00928353e-04, 1.00000025e00, 0.00000000e00],
            [1.60176282e-04, 2.64183789e-04, -3.03714663e-04, 1.02129617e-04, 1.00000040e00],
        ]
    )
    rng = np.random.default_rng(seed)
    Zn = rng.normal(size=(n, 5)) @ L.T
    k_h_cross, r2_g, log_knn_radius, local_label_variance, local_evaluation_count = Zn.T

    from scipy.stats import spearmanr

    raw = float(spearmanr(k_h_cross, r2_g).statistic)
    controlled = pcp.controlled_partial(
        k_h_cross, r2_g,
        np.column_stack([log_knn_radius, local_label_variance, local_evaluation_count]),
    )

    assert abs(raw - (-0.4124)) < ATOL_PARITY, f"raw={raw}"
    assert abs(controlled - (-0.2405)) < ATOL_PARITY, f"controlled={controlled}"


# --- out-of-fold structure ---------------------------------------------------------------------


def test_oof_predictions_are_out_of_fold():
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(3)
    n, d_in = 200, 8
    X = rng.normal(size=(n, d_in))
    w = rng.normal(size=d_in)
    y = X @ w + 0.1 * rng.normal(size=n)
    alpha, n_folds, fold_seed = 10.0, 5, 42

    y_hat = pcp.oof_ridge_predictions(X, y, alpha, n_folds, fold_seed)
    assert np.all(np.isfinite(y_hat))

    # Recover the SAME fold assignment and rebuild y_hat by hand -- membership disjointness and
    # exact reproduction of oof_ridge_predictions's own construction.
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=fold_seed)
    manual_y_hat = np.full(n, np.nan)
    for train_idx, test_idx in kfold.split(X):
        assert set(test_idx.tolist()).isdisjoint(set(train_idx.tolist()))
        fit = linear_probe.fit_probe(
            X[train_idx], y[train_idx].reshape(-1, 1),
            alpha_grid=(float(alpha), float(alpha)), alpha_per_target=False, fit_intercept=True,
        )
        manual_y_hat[test_idx] = linear_probe.predict_probe(fit, X[test_idx]).ravel()
    assert np.allclose(y_hat, manual_y_hat, atol=1e-10)

    # A single whole-dataset fit is a materially different prediction from the OOF array.
    whole_fit = linear_probe.fit_probe(
        X, y.reshape(-1, 1), alpha_grid=(float(alpha), float(alpha)),
        alpha_per_target=False, fit_intercept=True,
    )
    whole_pred = linear_probe.predict_probe(whole_fit, X).ravel()
    assert not np.allclose(y_hat, whole_pred, atol=1e-6)


def test_oof_raises_on_incomplete_coverage(monkeypatch):
    """Injects a deliberately-defective KFold stand-in that never assigns the last row to any
    test fold, proving the structural guard actually fires if the fold machinery were ever
    broken -- a behavioural pin, not a source grep."""

    class _IncompleteKFold:
        def __init__(self, n_splits, shuffle, random_state):
            self.n_splits = n_splits

        def split(self, X):
            n = X.shape[0]
            idx = np.arange(n - 1)  # the last row is never covered
            fold_size = len(idx) // self.n_splits
            for i in range(self.n_splits):
                if i < self.n_splits - 1:
                    test_idx = idx[i * fold_size : (i + 1) * fold_size]
                else:
                    test_idx = idx[i * fold_size :]
                train_idx = np.setdiff1d(np.arange(n), test_idx)
                yield train_idx, test_idx

    import sklearn.model_selection

    monkeypatch.setattr(sklearn.model_selection, "KFold", _IncompleteKFold)

    rng = np.random.default_rng(4)
    n, d_in = 60, 4
    X = rng.normal(size=(n, d_in))
    y = rng.normal(size=n)
    with pytest.raises(ValueError, match="never received a held-out prediction"):
        pcp.oof_ridge_predictions(X, y, alpha=1.0, n_folds=5, fold_seed=0)


# --- radial/tangential decomposition ------------------------------------------------------------


def test_radial_decomposition_identity():
    rng = np.random.default_rng(5)
    n, out_dim = 300, 12
    H_vec = rng.normal(size=(n, out_dim))
    image = rng.normal(size=(n, out_dim)) + 3.0  # keep norms comfortably away from 0

    result = pcp.decompose_radial_tangential(H_vec, image, min_image_norm=1e-9)
    lhs = result["H_rad"] ** 2 + result["H_tan_norm"] ** 2
    rhs = result["H_norm"] ** 2
    np.testing.assert_allclose(lhs, rhs, rtol=RTOL_DECOMPOSITION, atol=1e-12)
    assert result["n_excluded_low_norm"] == 0


def test_radial_decomposition_on_analytic_sphere():
    """Anchor-at-known-answer: points on a unit d-sphere with an analytically exact radial mean
    curvature vector H = -d * u (u the outward unit normal), plus a small tangential
    perturbation, so both H_rad and a nonzero H_tan are exercised."""
    rng = np.random.default_rng(6)
    d = 5
    ambient = d + 1
    n = 400
    raw = rng.normal(size=(n, ambient))
    image = raw / np.linalg.norm(raw, axis=1, keepdims=True)  # points on the unit d-sphere

    tangential_noise = rng.normal(scale=0.05, size=(n, ambient))
    radial_component = np.einsum("ij,ij->i", tangential_noise, image)
    tangential_noise = tangential_noise - radial_component[:, None] * image  # project to tangent

    H_vec = -d * image + tangential_noise

    result = pcp.decompose_radial_tangential(H_vec, image, min_image_norm=1e-9)
    h_rad_median = float(np.median(result["H_rad"]))
    assert abs(h_rad_median - (-d)) <= FRAC_SPHERE_TOLERANCE * d


def test_radial_decomposition_excludes_low_norm_rows():
    rng = np.random.default_rng(7)
    n, out_dim = 50, 4
    H_vec = rng.normal(size=(n, out_dim))
    image = rng.normal(size=(n, out_dim)) + 3.0
    low_norm_rows = np.array([2, 5, 9])
    image[low_norm_rows] = 1e-12

    result = pcp.decompose_radial_tangential(H_vec, image, min_image_norm=1e-6)
    assert result["n_excluded_low_norm"] == len(low_norm_rows)
    assert np.all(np.isnan(result["H_rad"][low_norm_rows]))
    kept = np.setdiff1d(np.arange(n), low_norm_rows)
    assert np.all(np.isfinite(result["H_rad"][kept]))


# --- anchor draw -----------------------------------------------------------------------------


def test_anchor_indices_disjoint_and_sorted():
    idx = pcp.anchor_indices(
        n_rows=1000, split_seed=1, holdout_fraction=0.2, n_anchors=64, anchor_seed=2
    )
    assert np.intersect1d(idx["anchor_idx"], idx["train_idx"]).size == 0
    assert np.all(idx["anchor_idx"][:-1] < idx["anchor_idx"][1:])
    assert np.array_equal(idx["anchor_idx"], np.unique(idx["anchor_idx"]))
    assert np.all(np.isin(idx["anchor_idx"], idx["holdout_idx"]))


def test_anchor_indices_stable_across_d():
    """anchor_indices depends on no fitted model -- calling it twice with the same
    split/anchor parameters (as would happen once per d in a d-sweep) is bit-for-bit
    identical."""
    kwargs = dict(n_rows=1000, split_seed=1, holdout_fraction=0.2, n_anchors=64, anchor_seed=2)
    idx_d16 = pcp.anchor_indices(**kwargs)
    idx_d20 = pcp.anchor_indices(**kwargs)
    assert np.array_equal(idx_d16["anchor_idx"], idx_d20["anchor_idx"])
    assert np.array_equal(idx_d16["train_idx"], idx_d20["train_idx"])
    assert np.array_equal(idx_d16["holdout_idx"], idx_d20["holdout_idx"])


def test_anchor_indices_boundary_pool_sizes():
    # n_rows=100, holdout_fraction=0.1 -> round(100*0.1) = 10 holdout rows.
    idx_equal = pcp.anchor_indices(
        n_rows=100, split_seed=1, holdout_fraction=0.1, n_anchors=10, anchor_seed=2
    )
    assert idx_equal["anchor_idx"].shape[0] == 10
    assert np.array_equal(idx_equal["anchor_idx"], idx_equal["holdout_idx"])

    with pytest.raises(ValueError, match="anchor_indices"):
        pcp.anchor_indices(n_rows=100, split_seed=1, holdout_fraction=0.1, n_anchors=11, anchor_seed=2)


# --- local out-of-fold R2 panel ----------------------------------------------------------------


def test_constant_evaluation_count_column_is_harmless():
    rng = np.random.default_rng(8)
    n = 200
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    radius = rng.normal(size=n)
    variance = rng.normal(size=n)
    const_count = np.full(n, 17.0)

    three_control = pcp.controlled_partial(x, y, np.column_stack([radius, variance, const_count]))
    two_control = pcp.controlled_partial(x, y, np.column_stack([radius, variance]))
    assert abs(three_control - two_control) < 1e-12


def test_local_r2_masks_degenerate_neighbourhoods():
    rng = np.random.default_rng(9)
    n = 100
    y = rng.normal(size=n)
    y_hat = y + 0.1 * rng.normal(size=n)

    # Anchor 0: too few finite pairs (min_finite=5, only 2 finite neighbours).
    neighbour_idx = np.zeros((3, 6), dtype=int)
    neighbour_idx[0] = [0, 1, 2, 3, 4, 5]
    y = y.copy()
    y_hat = y_hat.copy()
    y[neighbour_idx[0][2:]] = np.nan  # only 2 finite pairs remain for anchor 0

    # Anchor 1: SST exactly zero (every finite y identical in the neighbourhood).
    neighbour_idx[1] = [10, 11, 12, 13, 14, 15]
    y[neighbour_idx[1]] = 5.0
    y_hat[neighbour_idx[1]] = rng.normal(size=6)

    # Anchor 2: a normal, well-behaved neighbourhood.
    neighbour_idx[2] = [20, 21, 22, 23, 24, 25]

    panel = pcp.local_r2_panel(y, y_hat, neighbour_idx, min_finite=5)
    assert np.isnan(panel["r2"][0])
    assert np.isnan(panel["r2"][1])
    assert np.isfinite(panel["r2"][2])
    assert panel["n_masked_anchors"] == 2


# --- Freedman-Lane null -----------------------------------------------------------------------


def test_freedman_lane_preserves_control_fit():
    """The FITTED component (the projection of ranked y onto the ranked-controls design) is
    unaffected by which permutation is drawn -- proved two ways: (1) an RNG whose
    ``.permutation`` is the identity exactly reproduces rankdata(y) on the finite mask (fit +
    unpermuted residual = the original ranks, by construction); (2) a golden re-derivation of
    the colleague's own formula (PATTERNS.md's transcribed excerpt), run with an
    identically-seeded RNG, is bit-identical to physics_curvature_probe's own output."""
    from scipy.stats import rankdata

    rng_data = np.random.default_rng(10)
    n = 150
    y = rng_data.normal(size=n)
    Z = rng_data.normal(size=(n, 2))

    class _IdentityPermutationRNG:
        def permutation(self, arr):
            return np.asarray(arr)

    y2_identity = pcp.freedman_lane_y(y, Z, _IdentityPermutationRNG())
    np.testing.assert_allclose(y2_identity, rankdata(y), rtol=0, atol=1e-10)

    def _golden(y, Z, rng):
        m = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        yr = rankdata(y[m])
        Zr = np.column_stack([rankdata(Z[m, j]) for j in range(Z.shape[1])])
        A = np.column_stack([np.ones(int(m.sum())), Zr])
        fit = A @ np.linalg.lstsq(A, yr, rcond=None)[0]
        resid = yr - fit
        y2 = y.copy()
        y2[m] = fit + rng.permutation(resid)
        return y2

    seed = 20260902
    golden = _golden(y, Z, np.random.default_rng(seed))
    from_module = pcp.freedman_lane_y(y, Z, np.random.default_rng(seed))
    np.testing.assert_allclose(golden, from_module, rtol=0, atol=1e-12)


def test_p_value_never_zero():
    null_draws = np.zeros(199)
    pv = pcp.p_value_from_null(5.0, null_draws)
    assert pv["floor_reached"] is True
    assert pv["p_display"].startswith("< ")
    assert pv["p"] > 0.0

    null_draws2 = np.concatenate([np.full(50, 10.0), np.zeros(149)])
    pv2 = pcp.p_value_from_null(5.0, null_draws2)
    assert pv2["floor_reached"] is False
    assert not pv2["p_display"].startswith("< ")


# --- verdict rules -----------------------------------------------------------------------------


def test_per_d_verdict_strict_boundaries(monkeypatch):
    monkeypatch.setattr(pcp, "PER_D_VERDICT_VALUES", ("FIRED", "NOT_FIRED"))
    assert pcp.per_d_verdict(rho=0.0, p_fwer=0.001, fwer_alpha=0.05) == "NOT_FIRED"
    assert pcp.per_d_verdict(rho=-0.1, p_fwer=0.05, fwer_alpha=0.05) == "NOT_FIRED"
    assert pcp.per_d_verdict(rho=-0.1, p_fwer=0.049, fwer_alpha=0.05) == "FIRED"
    with pytest.raises(ValueError):
        pcp.per_d_verdict(rho=float("nan"), p_fwer=0.01, fwer_alpha=0.05)


def test_phase_verdict_empty_map(monkeypatch):
    monkeypatch.setattr(pcp, "VERDICT_VALUES", ("EVERY_D", "SUBSET_OF_D", "DOES_NOT_REPLICATE"))
    monkeypatch.setattr(pcp, "PER_D_VERDICT_VALUES", ("FIRED", "NOT_FIRED"))
    assert pcp.phase_verdict({}) == "DOES_NOT_REPLICATE"
    assert pcp.phase_verdict({16: "FIRED", 20: "FIRED"}) == "EVERY_D"
    assert pcp.phase_verdict({16: "FIRED", 20: "NOT_FIRED"}) == "SUBSET_OF_D"
    assert pcp.phase_verdict({16: "NOT_FIRED", 20: "NOT_FIRED"}) == "DOES_NOT_REPLICATE"


def test_combine_seed_verdicts_requires_exactly_three():
    with pytest.raises(ValueError):
        pcp.combine_seed_verdicts(["HOLDS", "HOLDS"])
    with pytest.raises(ValueError):
        pcp.combine_seed_verdicts(["HOLDS", "HOLDS", "HOLDS", "HOLDS"])
    assert pcp.combine_seed_verdicts(["HOLDS", "HOLDS", "HOLDS"]) == "HOLDS"
    assert pcp.combine_seed_verdicts(["HOLDS", "HOLDS", "NO RELATIONSHIP"]) == "SPLIT ACROSS SEEDS"


# --- positive control --------------------------------------------------------------------------


def test_positive_control_guards_before_search():
    rng = np.random.default_rng(11)
    n = 50
    y = rng.normal(size=n)
    Z = rng.normal(size=(n, 3))

    with pytest.raises(ValueError, match="h_real"):
        pcp.plant_curvature_positive_control(
            np.full(n, 3.0), y, Z, target_rho=0.1, seed=1, n_bisect=10
        )

    h_with_nan = rng.normal(size=n)
    h_with_nan[0] = np.nan
    with pytest.raises(ValueError, match="h_real"):
        pcp.plant_curvature_positive_control(h_with_nan, y, Z, target_rho=0.1, seed=1, n_bisect=10)


def test_positive_control_hits_target_grid():
    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(size=n)
    h_real = base + rng.normal(scale=0.1, size=n)
    y = -0.6 * base + rng.normal(scale=1.0, size=n)
    Z = rng.normal(size=(n, 3))

    for target in (-0.05, -0.10, -0.20, -0.30, -0.40):
        result = pcp.plant_curvature_positive_control(
            h_real, y, Z, target_rho=target, seed=20260902, n_bisect=40
        )
        assert abs(result["achieved_controlled_partial"] - target) < ATOL_TARGET_RHO


# --- shuffled-label repeat core ------------------------------------------------------------------


def test_shuffled_label_repeat_holds_radius_fixed():
    rng = np.random.default_rng(12)
    n, n_anchors, k = 300, 20, 15
    X = rng.normal(size=(n, 6))
    y = rng.normal(size=n)
    neighbour_idx = rng.integers(0, n, size=(n_anchors, k))
    log_knn_radius = rng.normal(size=n_anchors)
    log_knn_radius_orig = log_knn_radius.copy()
    h_field = rng.normal(size=n_anchors)

    shuffle_rng = np.random.default_rng(13)
    result1 = pcp.shuffled_label_repeat(
        X, y, neighbour_idx, log_knn_radius, h_field, alpha=1.0, n_folds=5, fold_seed=0,
        min_finite=5, rng=shuffle_rng,
    )
    result2 = pcp.shuffled_label_repeat(
        X, y, neighbour_idx, log_knn_radius, h_field, alpha=1.0, n_folds=5, fold_seed=0,
        min_finite=5, rng=shuffle_rng,
    )
    assert np.array_equal(log_knn_radius, log_knn_radius_orig)
    assert not np.allclose(
        result1["local_label_variance"], result2["local_label_variance"], equal_nan=True
    )


# --- output root containment --------------------------------------------------------------------


def test_output_root_containment():
    root = pcp.resolve_output_root()
    good = root / "some_test_file.jsonl"
    pcp._assert_inside_output_root(good)  # must not raise

    bad = root.parent / "escaped_outside_root.jsonl"
    with pytest.raises(ValueError):
        pcp._assert_inside_output_root(bad)


# --- freeze guard: malformed-constant sweep ------------------------------------------------------


def _plausible_value(name: str):
    equality_map = {
        "SEED_HANDLING_RULE": pcp._REQUIRED_SEED_HANDLING_RULE,
        "CURVATURE_FIELD_FOR_VERDICT": pcp._REQUIRED_CURVATURE_FIELD_FOR_VERDICT,
        "ALPHA_SELECTION_RULE": pcp._REQUIRED_ALPHA_SELECTION_RULE,
        "OOF_IMPLEMENTATION_RULE": pcp._REQUIRED_OOF_IMPLEMENTATION_RULE,
        "NULL_CONSTRUCTION_RULE": pcp._REQUIRED_NULL_CONSTRUCTION_RULE,
        "CURVATURE_CONVENTION": "trace",
    }
    if name in equality_map:
        return equality_map[name]
    original = getattr(pcp, name)
    if isinstance(original, dict):
        return {"placeholder_key": 1}
    if isinstance(original, tuple):
        return (1,)
    if isinstance(original, str) and original:  # already filled (SWISS_ROLL_APPLICABILITY_RULE)
        return original
    return 1


def _unset_value(name: str):
    """The UNSET representation for ``name``'s own type -- used to force the constant UNDER
    TEST into an UNSET state even when its current module default is already filled (only
    ``SWISS_ROLL_APPLICABILITY_RULE`` is filled by default in this plan; every other required
    constant already defaults to an UNSET value, but explicitly re-asserting it here makes the
    test robust to either case rather than relying on the current default)."""
    original = getattr(pcp, name)
    if isinstance(original, dict):
        return {}
    if isinstance(original, tuple):
        return ()
    if isinstance(original, str):
        return ""
    return None


@pytest.mark.parametrize("name", pcp._REQUIRED_CONSTANTS)
def test_assert_preregistered_rejects_unset_constant(name):
    with pytest.MonkeyPatch.context() as mp:
        for other in pcp._REQUIRED_CONSTANTS:
            if other == name:
                continue
            mp.setattr(pcp, other, _plausible_value(other))
        mp.setattr(pcp, name, _unset_value(name))
        with pytest.raises(RuntimeError, match=re.escape(name)):
            pcp.assert_preregistered()


# --- 09-08 runner-mode tests -------------------------------------------------------------------
# The runner (09_physics_curvature_run.py) is loaded via importlib.util file path rather than
# imported as a package -- it lives under notebooks/diagnostics/, outside the pu_manifold
# package this test file's own package machinery resolves. Every loader/fit call is stubbed so
# no test downloads a HuggingFace dataset or trains a real autoencoder; only the runner's own
# control flow (gating order, record-row assembly, D_SWEEP ordering, pooled-key absence) is
# exercised.

_RUNNER_PATH = Path(__file__).resolve().parents[2] / "diagnostics" / "09_physics_curvature_run.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("_physics_curvature_run_test_mod", _RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verdict_mode_requires_both_gates(tmp_path, monkeypatch):
    """A record with rows but no `positive_control`/`shuffled_label` kind must refuse to verdict
    (T-09-53): `run_verdict` exits 2 naming both missing kinds."""
    module = _load_runner_module()
    monkeypatch.setenv(module.pcp.OUTPUT_ROOT_ENV_VAR, str(tmp_path))

    record_path = tmp_path / "09_scratch_verdict_gate_test.jsonl"
    with record_path.open("w") as fh:
        fh.write(json.dumps({"row_kind": "fit", "d": 16}) + "\n")
        fh.write(json.dumps({
            "row_kind": "partial", "d": 16, "label": "mag_r", "field": "H_tan_norm",
            "raw_rho": -0.4, "controlled_partial": -0.2,
        }) + "\n")

    args = module.build_arg_parser().parse_args([
        "--mode", "verdict",
        "--freeze-commit", module.FREEZE_COMMIT_SHA,
        "--record-path", str(record_path),
    ])
    with pytest.raises(SystemExit) as exc_info:
        module.run_verdict(args)
    assert exc_info.value.code == 2

    # No verdict row was ever appended on the refusal path -- the record is unchanged in kind.
    rows = [json.loads(line) for line in record_path.open()]
    assert not any(r.get("row_kind") == "verdict" for r in rows)
    assert not any(r.get("row_kind") == "environment" for r in rows)


def _stub_dsweep_dependencies(module, monkeypatch, n_rows=400, n_anchors=80, k_neighbours=30):
    """Wires synthetic, network-free stand-ins for every real read/fit `run_dsweep` calls, and
    shrinks the frozen draw counts (permutation/bootstrap/stratified-null) so the stubbed sweep
    finishes in well under a second -- this test exercises RECORD-ROW STRUCTURE, never a real
    statistic. `D_SWEEP` itself is left untouched (the real frozen `(16, 20, 25, 32)`) because the
    test under it asserts fit rows appear in exactly that order."""
    rng = np.random.default_rng(20260902)
    X_stub = rng.normal(size=(n_rows, module.pcp.AE_IN_DIM))

    def _fake_load_physics_embeddings(*args, **kwargs):
        return {
            "X": X_stub, "row_norm": np.ones(n_rows), "n_rows": n_rows,
            "n_features": module.pcp.AE_IN_DIM, "source_url": "stub", "normalization": "stub",
        }

    label_values = {col: rng.normal(size=n_rows) for col in module.pl.LABEL_COLUMN_MAP.values()}

    monkeypatch.setattr(module.pl, "load_physics_embeddings", _fake_load_physics_embeddings)
    monkeypatch.setattr(module.pl, "load_label_table", lambda *args, **kwargs: label_values)

    def _fake_fit_and_field_at_anchors(X, d, anchor_idx, **kwargs):
        n_anc = len(anchor_idx)
        out_dim = 8
        return {
            "H_vec": rng.normal(size=(n_anc, out_dim)),
            "image": rng.normal(size=(n_anc, out_dim)) + 3.0,
            "metric_condition_number": np.abs(rng.normal(size=n_anc)) + 1.0,
            "var_explained": 0.9, "wallclock_fit_s": 0.001, "wallclock_field_s": 0.001,
        }

    monkeypatch.setattr(module, "fit_and_field_at_anchors", _fake_fit_and_field_at_anchors)

    monkeypatch.setattr(module.pcp, "N_ANCHORS", n_anchors)
    monkeypatch.setattr(module.pcp, "K_NEIGHBOURS", k_neighbours)
    monkeypatch.setattr(module.pcp, "HOLDOUT_FRACTION", 0.5)
    monkeypatch.setattr(module.pcp, "MIN_FINITE_NEIGHBOURS", 5)
    monkeypatch.setattr(module.pcp, "N_PERMUTATIONS", 5)
    monkeypatch.setattr(module.pcp, "N_BOOTSTRAP", 5)
    monkeypatch.setattr(module.pcp, "STRATIFIED_NULL_DRAWS", 5)


def _run_stubbed_dsweep(tmp_path, monkeypatch):
    module = _load_runner_module()
    monkeypatch.setenv(module.pcp.OUTPUT_ROOT_ENV_VAR, str(tmp_path))
    _stub_dsweep_dependencies(module, monkeypatch)

    record_path = tmp_path / "09_scratch_dsweep_test.jsonl"
    args = module.build_arg_parser().parse_args([
        "--mode", "dsweep", "--freeze-commit", module.FREEZE_COMMIT_SHA,
        "--record-path", str(record_path), "--threads", "1",
    ])
    ok = module.run_dsweep(args)
    assert ok is True
    rows = [json.loads(line) for line in record_path.open()]
    return module, record_path, rows


def test_dsweep_records_follow_d_sweep_order(tmp_path, monkeypatch):
    module, record_path, rows = _run_stubbed_dsweep(tmp_path, monkeypatch)

    fit_ds = [r["d"] for r in rows if r.get("row_kind") == "fit"]
    assert fit_ds == list(module.pcp.D_SWEEP)

    # Every other per-d row kind (anchor_summary, partial, null/stratified, bootstrap) must also
    # appear in D_SWEEP order (never interleaved out of order across d).
    for row_kind in ("anchor_summary", "partial", "bootstrap"):
        ds_seen_in_order = [r["d"] for r in rows if r.get("row_kind") == row_kind]
        # non-decreasing when filtered to the first-seen d ordering matches D_SWEEP's own order
        distinct_in_order = list(dict.fromkeys(ds_seen_in_order))
        assert distinct_in_order == list(module.pcp.D_SWEEP), row_kind


def test_no_pooled_headline_statistic(tmp_path, monkeypatch):
    """No row the dsweep-then-verdict path writes may carry a key naming a statistic pooled
    across `d`, other than the `fwer_global` null row -- which IS a null construction and is
    labelled as one (`null_type == "fwer_global"`), never a headline number."""
    module, record_path, dsweep_rows = _run_stubbed_dsweep(tmp_path, monkeypatch)

    # Satisfy run_verdict's own two-gates precondition with minimal synthetic gate rows so the
    # verdict path actually runs and its own appended row can be inspected too.
    with record_path.open("a") as fh:
        fh.write(json.dumps({
            "row_kind": "positive_control", "target_magnitude": 0.05, "cleared": True,
        }) + "\n")
        fh.write(json.dumps({"row_kind": "shuffled_label", "cleared": False}) + "\n")

    args = module.build_arg_parser().parse_args([
        "--mode", "verdict", "--freeze-commit", module.FREEZE_COMMIT_SHA,
        "--record-path", str(record_path),
    ])
    ok = module.run_verdict(args)
    assert ok is True

    rows = [json.loads(line) for line in record_path.open()]
    pooled_key_pattern = re.compile(r"pooled|across.?d\b|headline.?across", re.IGNORECASE)
    for row in rows:
        is_labelled_envelope = row.get("null_type") == "fwer_global"
        for key in row:
            if pooled_key_pattern.search(key) and not is_labelled_envelope:
                pytest.fail(
                    f"row_kind={row.get('row_kind')!r} carries an unlabelled pooled-looking key "
                    f"{key!r} (only the labelled fwer_global null row may carry a cross-d key)"
                )


# --- `--mode seeds` (Wave B, D9-17) --------------------------------------------------------------


def test_seeds_mode_refuses_untriggered_d(tmp_path):
    """`_triggered_d_values` reads the scope from the record's own `verdict` row and returns
    only the `d` values whose per-`d` verdict fired -- a `d` Wave A did not fire at is excluded,
    never passed through as a CLI argument an operator could widen after seeing a result
    (T-09-63/`WAVE_B_TRIGGER_RULE`)."""
    module = _load_runner_module()
    record_path = tmp_path / "09_scratch_triggered_scope_test.jsonl"
    with record_path.open("w") as fh:
        fh.write(json.dumps({
            "row_kind": "verdict",
            "per_d_verdicts": {
                "16": module.pcp.PER_D_VERDICT_VALUES[1],
                "20": module.pcp.PER_D_VERDICT_VALUES[0],
                "25": module.pcp.PER_D_VERDICT_VALUES[1],
                "32": module.pcp.PER_D_VERDICT_VALUES[0],
            },
        }) + "\n")

    triggered = module._triggered_d_values(record_path)
    assert triggered == [20, 32]
    assert 16 not in triggered
    assert 25 not in triggered

    # No verdict row at all -- e.g. Wave A's --mode verdict was never run -- is likewise an empty
    # scope, never an error, so run_seeds's own empty-scope branch handles it identically.
    empty_record_path = tmp_path / "09_scratch_no_verdict_row_test.jsonl"
    empty_record_path.write_text(json.dumps({"row_kind": "fit", "d": 16}) + "\n")
    assert module._triggered_d_values(empty_record_path) == []
    assert module._triggered_d_values(tmp_path / "09_scratch_does_not_exist.jsonl") == []


def test_seeds_mode_records_wave_b_not_triggered(tmp_path, monkeypatch):
    """When Wave A's own verdict row fires at zero `d` values, `--mode seeds` must append exactly
    one row carrying `wave_b == "WAVE_B_NOT_TRIGGERED"` and exit (return `True`) without fitting
    anything -- a complete, distinguishable outcome, never a silent no-op (T-09-65). No
    autoencoder call is stubbed here; `fit_and_field_at_anchors` and `pl.load_physics_embeddings`
    are left wired to their real (network/training) implementations, so if this branch ever fell
    through to a real fit, this test would hang or raise rather than pass -- that absence of a
    stub is itself part of the proof that the empty-scope path never reaches them."""
    module = _load_runner_module()
    monkeypatch.setenv(module.pcp.OUTPUT_ROOT_ENV_VAR, str(tmp_path))

    record_path = tmp_path / "09_scratch_not_triggered_test.jsonl"
    with record_path.open("w") as fh:
        fh.write(json.dumps({
            "row_kind": "verdict",
            "per_d_verdicts": {
                "16": module.pcp.PER_D_VERDICT_VALUES[1],
                "20": module.pcp.PER_D_VERDICT_VALUES[1],
                "25": module.pcp.PER_D_VERDICT_VALUES[1],
                "32": module.pcp.PER_D_VERDICT_VALUES[1],
            },
        }) + "\n")

    args = module.build_arg_parser().parse_args([
        "--mode", "seeds", "--freeze-commit", module.FREEZE_COMMIT_SHA,
        "--record-path", str(record_path),
    ])
    ok = module.run_seeds(args)
    assert ok is True

    rows = [json.loads(line) for line in record_path.open()]
    nt_rows = [r for r in rows if r.get("wave_b") == "WAVE_B_NOT_TRIGGERED"]
    assert len(nt_rows) == 1
    assert nt_rows[0]["row_kind"] == "seed_cell_verdict"
    assert not any(r.get("row_kind") == "seed_fit" for r in rows)
    assert not any(r.get("row_kind") == "seed_partial" for r in rows)


def test_seed_cell_verdict_never_upgrades_a_split():
    """T-09-61: two `PER_D_VERDICT_VALUES[0]` ("cleared") plus one `PER_D_VERDICT_VALUES[1]`
    ("not-cleared") must combine to the terminal split value, never an upgrade to unanimous
    clearance -- and two entries and four entries must both raise. Exercises
    `combine_seed_verdicts` in the exact vocabulary `run_seeds` actually passes it
    (`PER_D_VERDICT_VALUES`), distinct from `test_combine_seed_verdicts_requires_exactly_three`'s
    generic-string exercise above."""
    cleared = pcp.PER_D_VERDICT_VALUES[0]
    not_cleared = pcp.PER_D_VERDICT_VALUES[1]
    assert pcp.combine_seed_verdicts([cleared, cleared, not_cleared]) == "SPLIT ACROSS SEEDS"
    assert pcp.combine_seed_verdicts([cleared, not_cleared, not_cleared]) == "SPLIT ACROSS SEEDS"
    assert pcp.combine_seed_verdicts([cleared, cleared, cleared]) == cleared
    assert pcp.combine_seed_verdicts([not_cleared, not_cleared, not_cleared]) == not_cleared
    with pytest.raises(ValueError):
        pcp.combine_seed_verdicts([cleared, cleared])
    with pytest.raises(ValueError):
        pcp.combine_seed_verdicts([cleared, cleared, cleared, cleared])
