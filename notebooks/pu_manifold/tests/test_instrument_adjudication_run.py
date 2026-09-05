"""Smoke guard for `notebooks/diagnostics/09_instrument_adjudication_run.py`.

Runs `--mode smoke` in a subprocess against the colleague's read-only checkout (tiny in-sphere
fixture, exact autodiff truth, both instruments, both noise levels) with a temporary record path,
and asserts exit 0 and the final `SMOKE PASS` line. Skips cleanly when no checkout is available:
`EFFDIM_COLLEAGUE_ROOT` when set, else the known local/host locations below. Loads no Physics
data; the smoke mode is entirely synthetic.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

_RUNNER_PATH = Path(__file__).resolve().parents[2] / "diagnostics" / "09_instrument_adjudication_run.py"

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

    record_path = tmp_path / "09_scratch_adjudication_smoke.jsonl"
    result = subprocess.run(
        [
            sys.executable, str(_RUNNER_PATH), "--mode", "smoke", "--colleague-root", root,
            "--record-path", str(record_path), "--threads", "2",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.stdout.splitlines()[-1] == "SMOKE PASS"
    assert record_path.exists()
    # one environment row + (ours, his) x (noise 0, patch)
    assert sum(1 for _ in record_path.open()) == 5
