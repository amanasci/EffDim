# Removed from notebooks/pu_manifold/tests/test_physics_curvature_probe.py in paper-closure Stage 5a (original lines 565-798, contiguous):
# these tests exercise 09_physics_curvature_run.py's freeze gate and dsweep/verdict/seeds modes, archived under archive/notebooks/diagnostics/trimmed/.

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


def test_runner_gate_rejects_superseded_freeze_sha(capsys):
    """`_strict_ancestor_or_exit` must refuse the superseded 09-05 freeze SHA (CR-01's exact-
    equality check), even though it is a genuine ancestor of HEAD."""
    module = _load_runner_module()
    assert module.FREEZE_COMMIT_SHA == FREEZE_COMMIT_SHA
    with pytest.raises(SystemExit) as exc_info:
        module._strict_ancestor_or_exit(SUPERSEDED_FREEZE_SHA)
    assert exc_info.value.code == 1
    assert "does not equal the known freeze commit" in capsys.readouterr().err


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
