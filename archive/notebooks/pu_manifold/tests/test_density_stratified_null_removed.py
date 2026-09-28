# Removed from notebooks/pu_manifold/tests/test_density_stratified_null.py in the paper-closure
# Stage 2: these load notebooks/diagnostics/07.1_density_stratified_null_run.py (archived). Verbatim.
def test_torch_init_seed_is_restored_after_a_failed_fit(runner_071):
    """T-07.1-19: when the fit callable raises, fit_field_at_seed's seed-scoping helper still
    restores cc.TORCH_INIT_SEED to its entry value in a `finally` block -- the sealed module's
    attribute is never left mutated however the call ends."""
    entry_seed = cc.TORCH_INIT_SEED
    assert entry_seed == 0

    class _RaisingRunner:
        @staticmethod
        def fit_and_field(*args, **kwargs):
            raise RuntimeError("simulated fit failure")

    with pytest.raises(RuntimeError, match="simulated fit failure"):
        runner_071.fit_field_at_seed(_RaisingRunner(), np.zeros((10, 3)), seed=1, n_rows=10)

    assert cc.TORCH_INIT_SEED == entry_seed, (
        "cc.TORCH_INIT_SEED was left mutated after fit_field_at_seed's callee raised -- the "
        "`finally` restore did not run or did not restore the correct value."
    )


def test_fit_field_at_seed_halts_if_entry_seed_has_drifted(runner_071):
    """fit_field_at_seed asserts cc.TORCH_INIT_SEED equals Phase 7's frozen 0 on entry -- a
    drifted entry value halts rather than silently fitting under an unregistered seed."""
    entry_seed = cc.TORCH_INIT_SEED
    assert entry_seed == 0
    cc.TORCH_INIT_SEED = 99
    try:
        with pytest.raises(RuntimeError, match="drifted"):
            runner_071.fit_field_at_seed(object(), np.zeros((10, 3)), seed=1, n_rows=10)
    finally:
        cc.TORCH_INIT_SEED = entry_seed
    assert cc.TORCH_INIT_SEED == entry_seed


_RUNNER_071_PATH = (
    Path(__file__).resolve().parents[2] / "diagnostics" / "07.1_density_stratified_null_run.py"
)


@pytest.fixture(scope="module")
def runner_071():
    """Loads the 07.1 runner script as a module by file path -- it is not a package member (it
    lives under `notebooks/diagnostics/`, a sibling directory to `notebooks/pu_manifold/`).
    Matches `test_crossmodal_curvature_run.py`'s existing `runner` fixture pattern rather than
    inventing a second one. Module-level code only sets thread-related env vars and imports
    (pure numpy, no torch); it does not run `main()` (guarded by `if __name__ == "__main__"`),
    so import has no side effects beyond that."""
    spec = importlib.util.spec_from_file_location(
        "density_stratified_null_run_under_test", _RUNNER_071_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- runner: append_record_row rejects a raw numpy value ---------------------------------------


def test_record_row_rejects_raw_numpy(runner_071, tmp_path):
    record_path = tmp_path / "scratch.jsonl"
    with pytest.raises(TypeError):
        runner_071.append_record_row({"x": np.float64(1.0)}, record_path)
    with pytest.raises(TypeError):
        runner_071.append_record_row({"x": np.array([1, 2, 3])}, record_path)


def _frozen_artifacts_available() -> bool:
    fields_path = cache.cache_path("07_crossmodal_curvature_fields", "npz")
    record_path = cache.cache_path("07_crossmodal_curvature", "jsonl")
    subsample_cands = glob.glob(str(cache.CACHE_DIR / "subsample_*.npz"))
    return fields_path.exists() and record_path.exists() and len(subsample_cands) > 0


# --- D-07: recomputed partial reproduces Phase 7's frozen record at all three d ----------------


@pytest.mark.skipif(
    not _frozen_artifacts_available(),
    reason="frozen Phase 7 cache artifacts (fields npz / record jsonl / subsample npz) are "
    "absent in this checkout -- they are gitignored per CLAUDE.md and not always present.",
)
def test_recomputed_partial_matches_frozen_record(runner_071):
    mknn_arr, density, X_hsc, X_ls, subsample_file = runner_071.recompute_mknn_and_density()

    record_path = cache.cache_path("07_crossmodal_curvature", "jsonl")
    frozen_by_d = {}
    with record_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("row_kind") == "sweep":
                frozen_by_d[row["d"]] = row["partial_rho_density_controlled"]
    assert set(frozen_by_d.keys()) == {20, 25, 32}

    for d, frozen_value in frozen_by_d.items():
        h = runner_071.load_frozen_field(d)
        recomputed = cross_split_curvature.partial_spearman(h, mknn_arr, controls=density)
        assert np.isclose(
            recomputed, frozen_value,
            rtol=dsn.PARTIAL_REFERENCE_RTOL, atol=dsn.PARTIAL_REFERENCE_ATOL,
        ), f"d={d}: recomputed {recomputed!r} vs frozen record {frozen_value!r}"
        assert np.isclose(
            recomputed, dsn.FROZEN_PARTIAL_REFERENCE[d],
            rtol=dsn.PARTIAL_REFERENCE_RTOL, atol=dsn.PARTIAL_REFERENCE_ATOL,
        ), f"d={d}: recomputed {recomputed!r} vs dsn.FROZEN_PARTIAL_REFERENCE {dsn.FROZEN_PARTIAL_REFERENCE[d]!r}"
