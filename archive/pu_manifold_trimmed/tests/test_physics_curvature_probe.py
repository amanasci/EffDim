# Tests removed from notebooks/pu_manifold/tests/test_physics_curvature_probe.py in the paper-closure Stage 5: they
# exercise definitions archived under archive/pu_manifold_trimmed/. Verbatim, original order; the
# imports and aliases they use are those of the original test file.


# --- removed from tests/test_physics_curvature_probe.py:32-32 ---
ATOL_TARGET_RHO = 0.02


# --- removed from tests/test_physics_curvature_probe.py:426-432 ---
def test_combine_seed_verdicts_requires_exactly_three():
    with pytest.raises(ValueError):
        pcp.combine_seed_verdicts(["HOLDS", "HOLDS"])
    with pytest.raises(ValueError):
        pcp.combine_seed_verdicts(["HOLDS", "HOLDS", "HOLDS", "HOLDS"])
    assert pcp.combine_seed_verdicts(["HOLDS", "HOLDS", "HOLDS"]) == "HOLDS"
    assert pcp.combine_seed_verdicts(["HOLDS", "HOLDS", "NO RELATIONSHIP"]) == "SPLIT ACROSS SEEDS"


# --- removed from tests/test_physics_curvature_probe.py:435-435 ---
# --- positive control --------------------------------------------------------------------------


# --- removed from tests/test_physics_curvature_probe.py:438-452 ---
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


# --- removed from tests/test_physics_curvature_probe.py:455-467 ---
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


# --- removed from tests/test_physics_curvature_probe.py:470-470 ---
# --- shuffled-label repeat core ------------------------------------------------------------------


# --- removed from tests/test_physics_curvature_probe.py:473-495 ---
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


# --- removed from tests/test_physics_curvature_probe.py:563-579 ---
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
