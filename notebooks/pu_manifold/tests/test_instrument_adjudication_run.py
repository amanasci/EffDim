"""Smoke guard for `notebooks/diagnostics/09_instrument_adjudication_run.py`.

Runs `--mode smoke` in a subprocess (tiny in-sphere fixture, exact autodiff truth, both noise
levels) with a temporary record path, and asserts exit 0 and the final `SMOKE PASS` line. Loads
no Physics data; the smoke mode is entirely synthetic.
"""
import subprocess
import sys
from pathlib import Path

_RUNNER_PATH = Path(__file__).resolve().parents[2] / "diagnostics" / "09_instrument_adjudication_run.py"


def test_smoke_mode_passes_in_a_subprocess(tmp_path):
    record_path = tmp_path / "09_scratch_adjudication_smoke.jsonl"
    result = subprocess.run(
        [
            sys.executable, str(_RUNNER_PATH), "--mode", "smoke",
            "--record-path", str(record_path), "--threads", "2",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.stdout.splitlines()[-1] == "SMOKE PASS"
    assert record_path.exists()
    # one environment row + ours x (noise 0, patch)
    assert sum(1 for _ in record_path.open()) == 3
