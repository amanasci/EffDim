"""appendix_gen.py regenerates main.tex byte-for-byte from the records (needs the record cache)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("EFFDIM_CACHE_DIR") or PAPER.parent / "curvature-experiment" / ".cache")
needs_records = pytest.mark.skipif(not (CACHE / "09_physics_probe_facing_split.jsonl").exists(),
                                   reason="record cache not available")


@needs_records
def test_appendix_gen_reproduces_main_tex(tmp_path):
    sandbox = tmp_path / "paper"
    shutil.copytree(PAPER / "generate", sandbox / "generate")
    shutil.copytree(PAPER / "latex", sandbox / "latex")
    shutil.copy(PAPER / "records.py", sandbox / "records.py")
    env = dict(os.environ, EFFDIM_CACHE_DIR=str(CACHE))
    subprocess.run([sys.executable, str(sandbox / "generate" / "appendix_gen.py")], check=True, env=env, cwd=tmp_path)
    assert (sandbox / "latex" / "main.tex").read_bytes() == (PAPER / "latex" / "main.tex").read_bytes()
