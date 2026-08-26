"""Focused guard tests for the Phase 7 runner script
(`notebooks/diagnostics/07_crossmodal_curvature_run.py`).

WR-04 recorded that the runner carried zero automated coverage. This file targets exactly the
three CR-01/CR-02/CR-03 guard-strength gaps `07-REVIEW.md` found (and this commit fixes) --
not broad runner coverage (CLAUDE.md: keep things simple first):

- CR-01: `--freeze-commit` must resolve to EXACTLY `FREEZE_COMMIT_SHA`, not merely be some
  earlier ancestor of HEAD.
- CR-02: `--mode positive-control` must pass through the same strict-ancestor gate
  `--mode dsweep` uses before it can write any row.
- CR-03: `--threads`/`--smoke-rows`/`--max-epochs` must be recognized in both `--flag value`
  and `--flag=value` argv forms.

Loads no PU data, trains nothing, reads no cache. Importing the runner module pulls in torch
(module-level `import torch`), matching this repo's existing test suite's dependency on the
project virtualenv (`.venv/bin/python`, per CLAUDE.md / environment notes).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "diagnostics" / "07_crossmodal_curvature_run.py"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def runner():
    """Loads the runner script as a module by file path (it is not a package member -- it
    lives under `notebooks/diagnostics/`, a sibling directory to `notebooks/pu_manifold/`).
    Module-level code only sets thread-related env vars and imports; it does not run `main()`
    (guarded by `if __name__ == "__main__"`), so import has no side effects beyond that."""
    spec = importlib.util.spec_from_file_location("crossmodal_curvature_run_under_test", _RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- CR-01: --freeze-commit must equal FREEZE_COMMIT_SHA, not merely be an ancestor --------


def test_strict_ancestor_or_exit_accepts_the_real_freeze_commit(runner):
    """The exact reproducibility invocation (`--freeze-commit f032745...`) must still pass --
    FREEZE_COMMIT_SHA resolves to itself, so both the equality check and the strict-ancestor
    check hold. No SystemExit means success."""
    runner._strict_ancestor_or_exit(runner.FREEZE_COMMIT_SHA)


def test_strict_ancestor_or_exit_rejects_a_wrong_but_genuine_ancestor(runner):
    """CR-01's exact failure shape: a SHA that IS a real, strict git ancestor of HEAD (so the
    pre-fix ancestry-only check would have silently passed it) but is NOT the freeze commit.
    `HEAD~1` (resolved to a full SHA) satisfies exactly this: a genuine ancestor, wrong
    freeze."""
    wrong_but_ancestor = subprocess.run(
        ["git", "rev-parse", "HEAD~1"],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert wrong_but_ancestor != runner.FREEZE_COMMIT_SHA, (
        "test fixture assumption broken: HEAD~1 must not equal the freeze commit itself"
    )
    with pytest.raises(SystemExit) as excinfo:
        runner._strict_ancestor_or_exit(wrong_but_ancestor)
    assert excinfo.value.code == 1


def test_strict_ancestor_or_exit_rejects_missing_value(runner):
    with pytest.raises(SystemExit) as excinfo:
        runner._strict_ancestor_or_exit(None)
    assert excinfo.value.code == 1


# --- CR-02: --mode positive-control must be gated identically to --mode dsweep -------------


def test_run_positive_control_calls_the_strict_ancestor_gate_before_touching_field_npz(
    runner, monkeypatch
):
    """Proves ordering: the gate must run before `run_positive_control` ever inspects
    `args.field_npz`. We monkeypatch the gate to record that it was called (without exiting)
    and then let execution continue into the (expected) FileNotFoundError raised because no
    `--field-npz` was supplied -- if the gate were not wired in, this test would still reach
    the same FileNotFoundError, but `calls` would stay empty, so this test would fail on the
    `assert calls` line rather than passing vacuously."""
    calls = []

    def _fake_gate(freeze_commit):
        calls.append(freeze_commit)

    monkeypatch.setattr(runner, "_strict_ancestor_or_exit", _fake_gate)
    args = SimpleNamespace(field_npz=None)

    with pytest.raises(FileNotFoundError):
        runner.run_positive_control(args)

    assert calls == [runner.FREEZE_COMMIT_SHA], (
        "run_positive_control must call _strict_ancestor_or_exit(FREEZE_COMMIT_SHA) before "
        "reaching the field_npz check"
    )


def test_run_positive_control_propagates_a_failed_gate_before_any_write(runner, monkeypatch):
    """If the gate fails (SystemExit), run_positive_control must not proceed past it -- in
    particular it must not reach the field_npz check (which would raise a different
    exception, FileNotFoundError, if the gate call were missing or misplaced)."""

    def _failing_gate(freeze_commit):
        sys.exit(1)

    monkeypatch.setattr(runner, "_strict_ancestor_or_exit", _failing_gate)
    args = SimpleNamespace(field_npz=None)

    with pytest.raises(SystemExit) as excinfo:
        runner.run_positive_control(args)
    assert excinfo.value.code == 1


# --- CR-03: --flag value and --flag=value must parse equivalently --------------------------


@pytest.mark.parametrize("flag", ["--threads", "--smoke-rows", "--max-epochs"])
def test_flag_value_from_argv_accepts_space_separated_form(runner, flag):
    argv = ["prog", flag, "42"]
    assert runner._flag_value_from_argv(flag, argv) == "42"


@pytest.mark.parametrize("flag", ["--threads", "--smoke-rows", "--max-epochs"])
def test_flag_value_from_argv_accepts_equals_form(runner, flag):
    argv = ["prog", f"{flag}=42"]
    assert runner._flag_value_from_argv(flag, argv) == "42"


@pytest.mark.parametrize("flag", ["--threads", "--smoke-rows", "--max-epochs"])
def test_flag_value_from_argv_returns_none_when_absent(runner, flag):
    argv = ["prog", "--mode", "dsweep"]
    assert runner._flag_value_from_argv(flag, argv) is None


def test_flag_value_from_argv_space_and_equals_forms_agree(runner):
    """The two forms argparse itself treats as equivalent must be recognized as equivalent
    here too -- this is the exact regression CR-03 named: a raw `"--max-epochs" in sys.argv`
    scan returns True for the space form and False for the `=` form of the identical
    request."""
    space_form = ["prog", "--max-epochs", "5"]
    equals_form = ["prog", "--max-epochs=5"]
    assert (
        runner._flag_value_from_argv("--max-epochs", space_form)
        == runner._flag_value_from_argv("--max-epochs", equals_form)
        == "5"
    )
