# Final whole-branch review (after fa139de): tests removed from curvature-experiment/tests/test_pu_manifold.py -- they
# exercise definitions archived in the final review. Line numbers at fa139de. Verbatim, original
# order; the imports and aliases they use are those of the original test file.


# --- removed from tests/test_pu_manifold.py:67-83 ---
# --- joblib_cache --------------------------------------------------------------------------


def test_joblib_cache_round_trip_is_bit_identical_and_computes_once():
    cfg = {"seed": 2}
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"embedding_": np.arange(12).reshape(4, 3).astype(np.float64)}

    first = cache_mod.joblib_cache("stem_joblib", cfg, compute)
    second = cache_mod.joblib_cache("stem_joblib", cfg, compute)

    assert calls["count"] == 1
    assert np.array_equal(first["embedding_"], second["embedding_"])



# --- removed from tests/test_pu_manifold.py:151-179 ---
# --- assert_alignment ----------------------------------------------------------------------


def test_assert_alignment_passes_on_synthetic_aligned_pair():
    rng = np.random.default_rng(7)
    n = 200
    hsc_raw = rng.standard_normal((n, subsample_mod.N_FEATURES))
    ls_raw = hsc_raw + 0.01 * rng.standard_normal((n, subsample_mod.N_FEATURES))
    hsc, _ = subsample_mod.l2_normalize(hsc_raw)
    legacysurvey, _ = subsample_mod.l2_normalize(ls_raw)
    row_indices = np.arange(n)

    stats = subsample_mod.assert_alignment(hsc, legacysurvey, row_indices, seed=7)
    assert stats["z"] > subsample_mod.ALIGNMENT_MARGIN_Z
    assert "row_indices_sha256" in stats


def test_assert_alignment_raises_on_off_by_one_negative_control():
    rng = np.random.default_rng(7)
    n = 200
    hsc_raw = rng.standard_normal((n, subsample_mod.N_FEATURES))
    ls_raw = hsc_raw + 0.01 * rng.standard_normal((n, subsample_mod.N_FEATURES))
    hsc, _ = subsample_mod.l2_normalize(hsc_raw)
    legacysurvey, _ = subsample_mod.l2_normalize(ls_raw)
    legacysurvey_rolled = np.roll(legacysurvey, 1, axis=0)
    row_indices = np.arange(n)

    with pytest.raises(ValueError):
        subsample_mod.assert_alignment(hsc, legacysurvey_rolled, row_indices, seed=7)
