"""Smoke guard for the supplementary runner `notebooks/diagnostics/09_colleague_estimator_run.py`.

Runs `--mode smoke` in a subprocess against the colleague's read-only checkout (his code
imported unchanged, our sealed statistics) with a temporary output root and record path, and
asserts exit 0 and the final `SMOKE PASS` line. Skips cleanly when no checkout is available:
the path is taken from `EFFDIM_COLLEAGUE_ROOT` when set, else from the known local/host
locations below. Loads no Physics data; the smoke mode is entirely synthetic.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

_RUNNER_PATH = Path(__file__).resolve().parents[2] / "diagnostics" / "09_colleague_estimator_run.py"

_KNOWN_COLLEAGUE_ROOTS = (
    "/mnt/ssd-cluster/effdim/colleague-curvature-experiments",
    "/tmp/claude-1000/-home-akagi-Documents-Projects-EffDim/7b8f80e3-f8dc-495d-afa1-9f6c727c75a6/"
    "scratchpad/colleague-curvature-experiments",
)


def _colleague_root():
    candidates = [os.environ.get("EFFDIM_COLLEAGUE_ROOT")] + list(_KNOWN_COLLEAGUE_ROOTS)
    for c in candidates:
        if c and (Path(c) / "experiments").is_dir():
            return c
    return None


def test_smoke_mode_passes_in_a_subprocess(tmp_path):
    root = _colleague_root()
    if root is None:
        pytest.skip("colleague checkout not present (set EFFDIM_COLLEAGUE_ROOT to run)")

    record_path = tmp_path / "09_scratch_colleague_smoke.jsonl"
    env = dict(os.environ)
    env["EFFDIM_09_OUTPUT_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable, str(_RUNNER_PATH), "--mode", "smoke", "--colleague-root", root,
            "--record-path", str(record_path), "--threads", "2", "--smoke-permutations", "50",
        ],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "SMOKE PASS" in result.stdout.splitlines()[-1]
    assert record_path.exists()
